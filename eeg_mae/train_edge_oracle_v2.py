"""Train the advanced multi-layer DINO edge oracle v2."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from .advanced_edge import DINOEdgeOracleV2, advanced_edge_loss
from .build_pyramid_edge_targets import PYRAMID_KEYS
from .pyramid_edge import PyramidEdgeOutput, load_pyramid_edge_decoder
from .train_eeg_pyramid_edges import concatenate
from .train_pyramid_edge_oracle import metrics as pyramid_metrics


def arguments():
    p = argparse.ArgumentParser(description="Advanced multi-layer DINO edge oracle")
    p.add_argument("--rich-bank", default="data/derived/rich_dino_color_v2")
    p.add_argument("--target-bank", default="data/derived/pyramid_edge_targets_v1.pt")
    p.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    p.add_argument("--baseline-bank", default="data/derived/visual_targets_dinov2s_192.pt")
    p.add_argument("--baseline-decoder", default="runs/dino_pyramid_edge_oracle_v1/best.pt")
    p.add_argument("--output", default="runs/dino_edge_oracle_v2")
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--eval-batch-size", type=int, default=12)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--feature-noise", type=float, default=0.012)
    p.add_argument("--token-dropout", type=float, default=0.04)
    p.add_argument("--layer-dropout", type=float, default=0.10)
    p.add_argument("--mask-ratio", type=float, default=0.35)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--panel-images", type=int, default=6)
    p.add_argument("--seed", type=int, default=20260808)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def split_names(path):
    groups = {"train": set(), "val": set(), "test": set()}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["subject_id"]) == 0 and row["split"] in groups:
                groups[row["split"]].add(row["image_file"])
    return {key: sorted(value) for key, value in groups.items()}


class RichDINOEdgeDataset(Dataset):
    def __init__(self, root, names, target):
        self.manifest = json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
        if not self.manifest.get("complete"):
            raise RuntimeError("rich DINO bank is incomplete")
        rich_lookup = {name: i for i, name in enumerate(self.manifest["filenames"])}
        target_lookup = {name: i for i, name in enumerate(target["filenames"])}
        self.rows = [
            (rich_lookup[name], target_lookup[name], name)
            for name in names
            if name in rich_lookup
            and name in target_lookup
            and bool(target["valid"][target_lookup[name]])
        ]
        root = Path(root)
        self.patches = np.load(root / "patches.npy", mmap_mode="r")
        self.cls = np.load(root / "cls.npy", mmap_mode="r")
        self.rgb = np.load(root / "rgb.npy", mmap_mode="r")
        self.target = target

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        rich, target, name = self.rows[index]
        return {
            "patches": torch.from_numpy(np.array(self.patches[rich], copy=True)),
            "cls": torch.from_numpy(np.array(self.cls[rich], copy=True)),
            "rgb": torch.from_numpy(np.array(self.rgb[rich], copy=True)).float() / 255,
            "filename": name,
            **{key: self.target[key][target].float() for key in PYRAMID_KEYS},
        }


class LegacyEdgeOracle:
    def __init__(self, bank_path, decoder_path, device):
        self.bank = torch.load(bank_path, map_location="cpu", weights_only=False, mmap=True)
        self.lookup = {name: i for i, name in enumerate(self.bank["filenames"])}
        self.model, _ = load_pyramid_edge_decoder(decoder_path, device)
        self.device = device

    @torch.no_grad()
    def __call__(self, names):
        indices = [self.lookup[name] for name in names]
        grid = F.normalize(self.bank["dino_grid"][indices].float().to(self.device), dim=-1)
        return self.model(grid)


def loader(dataset, batch, shuffle, seed):
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def augment(patches, cls, args):
    if args.feature_noise:
        patches = patches + args.feature_noise * patches.float().std(
            (2, 3, 4), keepdim=True
        ).clamp_min(1e-3) * torch.randn_like(patches)
        cls = cls + args.feature_noise * cls.float().std(2, keepdim=True).clamp_min(
            1e-3
        ) * torch.randn_like(cls)
    if args.token_dropout:
        patches = patches * (
            torch.rand((*patches.shape[:4], 1), device=patches.device) >= args.token_dropout
        )
    if args.layer_dropout:
        keep = (
            torch.rand((len(patches), patches.shape[1], 1, 1, 1), device=patches.device)
            >= args.layer_dropout
        )
        empty = ~keep.flatten(1).any(1)
        if empty.any():
            keep[empty, -1] = True
        patches = patches * keep
        cls = cls * keep[:, :, 0, 0]
    return patches, cls


@torch.no_grad()
def update_ema(ema, model, decay):
    for target, source in zip(ema.parameters(), model.parameters()):
        target.mul_(decay).add_(source, alpha=1 - decay)
    for target, source in zip(ema.buffers(), model.buffers()):
        target.copy_(source)


def train_epoch(model, ema, data, optimizer, scaler, device, args):
    model.train()
    totals = {}
    batches = 0
    for batch in data:
        patches, cls = augment(batch["patches"].to(device), batch["cls"].to(device), args)
        target = {key: batch[key].to(device) for key in PYRAMID_KEYS}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            output = model(patches, cls, mask_ratio=args.mask_ratio)
            loss, parts = advanced_edge_loss(output, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 4.0)
        scaler.step(optimizer)
        scaler.update()
        update_ema(ema, model, args.ema_decay)
        batches += 1
        values = {"loss": loss, **parts}
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
    return {key: value / max(batches, 1) for key, value in totals.items()}


@torch.no_grad()
def predict(model, data, device):
    model.eval()
    outputs = []
    targets = {key: [] for key in PYRAMID_KEYS}
    for batch in data:
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            output = model(batch["patches"].to(device), batch["cls"].to(device)).pyramid
        outputs.append(
            PyramidEdgeOutput(
                **{
                    field: getattr(output, field).float().cpu()
                    for field in PyramidEdgeOutput.__dataclass_fields__
                }
            )
        )
        for key in PYRAMID_KEYS:
            targets[key].append(batch[key].float())
    return concatenate(outputs), {key: torch.cat(value) for key, value in targets.items()}


def shuffled_output(output, seed):
    order = torch.randperm(
        len(output.shape_logits16), generator=torch.Generator().manual_seed(seed)
    )
    return PyramidEdgeOutput(
        **{field: getattr(output, field)[order] for field in PyramidEdgeOutput.__dataclass_fields__}
    )


def evaluate(model, data, device, seed):
    output, target = predict(model, data, device)
    return pyramid_metrics(output, target), pyramid_metrics(shuffled_output(output, seed), target)


@torch.no_grad()
def evaluate_legacy(model, data, seed):
    outputs = []
    targets = {key: [] for key in PYRAMID_KEYS}
    for batch in data:
        value = model(list(batch["filename"]))
        outputs.append(
            PyramidEdgeOutput(
                **{
                    field: getattr(value, field).float().cpu()
                    for field in PyramidEdgeOutput.__dataclass_fields__
                }
            )
        )
        for key in PYRAMID_KEYS:
            targets[key].append(batch[key].float())
    output = concatenate(outputs)
    target = {key: torch.cat(value) for key, value in targets.items()}
    return pyramid_metrics(output, target), pyramid_metrics(shuffled_output(output, seed), target)


def score(value, wrong):
    keys = (
        "semantic_tolerant_f1",
        "crisp_tolerant_f1",
        "perceptual_pearson",
        "distance_correlation",
    )
    specificity = sum(value[key] - wrong[key] for key in keys)
    return (
        value["shape_iou"]
        + value["semantic_tolerant_f1"]
        + 1.2 * value["crisp_tolerant_f1"]
        + value["perceptual_pearson"]
        + value["distance_correlation"]
        + 0.9 * specificity
        + 0.25 * value["orientation_cosine"]
        - 0.2 * value["confidence_mae"]
    )


def gray(value, tile=112):
    array = (value.squeeze().float().clamp(0, 1) * 255).byte().numpy()
    return Image.fromarray(array, "L").resize((tile, tile), Image.Resampling.NEAREST).convert("RGB")


def rgb_image(value, tile=112):
    array = (value.permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
    return Image.fromarray(array).resize((tile, tile), Image.Resampling.NEAREST)


@torch.no_grad()
def save_panel(model, dataset, legacy, device, path, count):
    samples = [dataset[i] for i in range(min(count, len(dataset)))]
    if not samples:
        return
    patches = torch.stack([s["patches"] for s in samples]).to(device)
    cls = torch.stack([s["cls"] for s in samples]).to(device)
    with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
        output = model(patches, cls).pyramid
    base = legacy([s["filename"] for s in samples]) if legacy else None
    headers = (
        "image",
        "semantic GT",
        "semantic v1",
        "semantic v2",
        "crisp GT",
        "crisp v1",
        "crisp v2",
        "uncertainty v2",
    )
    tile, label = 112, 22
    canvas = Image.new("RGB", (len(headers) * tile, (len(samples) + 1) * (tile + label)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, header in enumerate(headers):
        draw.text((column * tile + 3, 3), header, fill="black")
    for row, sample in enumerate(samples):
        y = (row + 1) * (tile + label)
        draw.text((3, y - label + 3), sample["filename"], fill="black")
        zero = torch.zeros_like(output.semantic_logits32[row].cpu())
        zero64 = torch.zeros_like(output.crisp_logits64[row].cpu())
        images = (
            rgb_image(sample["rgb"], tile),
            gray(sample["semantic32"], tile),
            gray(base.semantic_logits32.sigmoid().cpu()[row] if base else zero, tile),
            gray(output.semantic_logits32.sigmoid().cpu()[row], tile),
            gray(sample["crisp64"], tile),
            gray(base.crisp_logits64.sigmoid().cpu()[row] if base else zero64, tile),
            gray(output.crisp_logits64.sigmoid().cpu()[row], tile),
            gray(output.uncertainty64.cpu()[row], tile),
        )
        for column, image in enumerate(images):
            canvas.paste(image, (column * tile, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    args = arguments()
    if args.smoke:
        args.epochs = 1
        args.batch_size = 2
        args.eval_batch_size = 4
        args.panel_images = 2
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    target = torch.load(args.target_bank, map_location="cpu", weights_only=False, mmap=True)
    manifest = json.loads((Path(args.rich_bank) / "manifest.json").read_text(encoding="utf-8"))
    groups = split_names(args.index)
    datasets = {
        key: RichDINOEdgeDataset(args.rich_bank, names, target) for key, names in groups.items()
    }
    if args.smoke:
        available = RichDINOEdgeDataset(args.rich_bank, manifest["filenames"], target)
        if any(len(value) == 0 for value in datasets.values()):
            a = max(2, int(0.7 * len(available)))
            b = max(a + 1, int(0.85 * len(available)))
            datasets = {
                "train": Subset(available, range(a)),
                "val": Subset(available, range(a, b)),
                "test": Subset(available, range(b, len(available))),
            }
        datasets = {
            key: Subset(value, range(min(16, len(value)))) for key, value in datasets.items()
        }
    loaders = {
        key: loader(
            value,
            args.batch_size if key == "train" else args.eval_batch_size,
            key == "train",
            args.seed,
        )
        for key, value in datasets.items()
    }
    if not len(datasets["train"]) or not len(datasets["val"]):
        raise RuntimeError("train or validation split is empty")
    model = DINOEdgeOracleV2(
        input_dim=int(manifest["hidden_dim"]), layers=len(manifest["layers"])
    ).to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=2e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    legacy = (
        LegacyEdgeOracle(args.baseline_bank, args.baseline_decoder, device)
        if Path(args.baseline_bank).exists() and Path(args.baseline_decoder).exists()
        else None
    )
    baseline_score = None
    if legacy:
        base, wrong = evaluate_legacy(legacy, loaders["val"], args.seed)
        baseline_score = score(base, wrong)
        (out / "baseline_validation.json").write_text(
            json.dumps({"score": baseline_score, "validation": base, "shuffled": wrong}, indent=2),
            encoding="utf-8",
        )
        print(
            f"legacy edge oracle score={baseline_score:.3f} semantic={base['semantic_tolerant_f1']:.3f} crisp={base['crisp_tolerant_f1']:.3f}",
            flush=True,
        )
    start = 0
    best = -math.inf
    stale = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start = state["epoch"] + 1
        best = state.get("best_score", best)
        stale = state.get("stale", 0)
    history = out / "history.jsonl"
    if not args.resume:
        history.unlink(missing_ok=True)
    panel_dataset = (
        datasets["val"].dataset if isinstance(datasets["val"], Subset) else datasets["val"]
    )
    for epoch in range(start, args.epochs):
        train = train_epoch(model, ema, loaders["train"], optimizer, scaler, device, args)
        validation, wrong = evaluate(ema, loaders["val"], device, args.seed + epoch)
        current = score(validation, wrong)
        scheduler.step(current)
        improved = current > best + 1e-5
        stale = 0 if improved else stale + 1
        record = {
            "epoch": epoch,
            "score": current,
            "beats_legacy_oracle": baseline_score is None or current > baseline_score,
            "train": train,
            "validation": validation,
            "shuffled": wrong,
            "lr": optimizer.param_groups[0]["lr"],
        }
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "model_config": model.config(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": max(best, current),
            "stale": stale,
            "record": record,
            "rich_bank_manifest": manifest,
        }
        torch.save(payload, out / "last.pt")
        if improved:
            best = current
            torch.save(payload, out / "best.pt")
        save_panel(
            ema,
            panel_dataset,
            legacy,
            device,
            out / "panels" / f"epoch_{epoch:03d}.png",
            args.panel_images,
        )
        print(
            f"epoch={epoch:03d} loss={train['loss']:.3f} score={current:.3f} semantic={validation['semantic_tolerant_f1']:.3f} crisp={validation['crisp_tolerant_f1']:.3f} orientation={validation['orientation_cosine']:.3f}",
            flush=True,
        )
        if stale >= args.patience:
            break
    selected = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    ema.load_state_dict(selected["ema"])
    ema.to(device)
    test, wrong = evaluate(ema, loaders["test"], device, args.seed + 999)
    baseline_test = evaluate_legacy(legacy, loaders["test"], args.seed + 999)[0] if legacy else None
    report = {
        "checkpoint_epoch": selected["epoch"],
        "oracle": test,
        "legacy_oracle": baseline_test,
        "shuffled": wrong,
        "test_images": len(datasets["test"]),
        "untouched_full_test": not args.smoke,
    }
    (out / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
