"""Train the high-capacity multi-layer DINO color oracle."""

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

from ..models.advanced_color import (
    DINOColorOracleV2,
    PatchDiscriminator,
    advanced_color_loss,
    color_artifact_metrics,
    rgb_to_oklab,
)
from ..models.color import ColorOutput, color_descriptor, color_metrics, load_color_decoder


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Advanced multi-layer DINO color oracle")
    p.add_argument("--rich-bank", default="data/derived/rich_dino_color_v2")
    p.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    p.add_argument("--output", default="runs/eeg_color_encoder/training/color_oracle")
    p.add_argument("--baseline-bank", default="data/derived/visual_targets_dinov2s_192.pt")
    p.add_argument(
        "--baseline-decoder", default="runs/eeg_color_encoder/components/color_decoder.pt"
    )
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--discriminator-lr", type=float, default=8e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--feature-noise", type=float, default=0.015)
    p.add_argument("--token-dropout", type=float, default=0.04)
    p.add_argument("--layer-dropout", type=float, default=0.10)
    p.add_argument("--adversarial-start", type=int, default=10)
    p.add_argument("--adversarial-weight", type=float, default=0.025)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--panel-images", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260807)
    p.add_argument("--device", default="cuda")
    p.add_argument("--chroma-gradient-weight", type=float, default=0.45)
    p.add_argument("--hard-pixel-weight", type=float, default=0.60)
    p.add_argument("--hard-pixel-fraction", type=float, default=0.08)
    p.add_argument("--spurious-chroma-edge-weight", type=float, default=0.65)
    p.add_argument("--gamut-weight", type=float, default=0.30)
    p.add_argument("--resume", default=None)
    p.add_argument("--finetune-from", default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def split_names(path: Path) -> dict[str, list[str]]:
    groups = {"train": set(), "val": set(), "test": set()}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["subject_id"]) == 0 and row["split"] in groups:
                groups[row["split"]].add(row["image_file"])
    return {k: sorted(v) for k, v in groups.items()}


class RichDINOColorDataset(Dataset):
    def __init__(self, root: Path, names: list[str]) -> None:
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if not self.manifest.get("complete"):
            raise RuntimeError(
                f"rich bank is incomplete: {self.manifest.get('processed')}/{self.manifest.get('images')}"
            )
        lookup = {name: i for i, name in enumerate(self.manifest["filenames"])}
        self.rows = [(lookup[name], name) for name in names if name in lookup]
        self.patches = np.load(root / "patches.npy", mmap_mode="r")
        self.cls = np.load(root / "cls.npy", mmap_mode="r")
        self.rgb = np.load(root / "rgb.npy", mmap_mode="r")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row, name = self.rows[index]
        # Explicit copies keep PyTorch away from read-only mmap storage during collation.
        return {
            "patches": torch.from_numpy(np.array(self.patches[row], copy=True)),
            "cls": torch.from_numpy(np.array(self.cls[row], copy=True)),
            "rgb": torch.from_numpy(np.array(self.rgb[row], copy=True)).float() / 255,
            "filename": name,
        }


class LegacyOracle:
    def __init__(self, bank_path: str, decoder_path: str, device):
        self.bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        self.lookup = {name: i for i, name in enumerate(self.bank["filenames"])}
        self.model, _ = load_color_decoder(decoder_path, device)
        self.device = device

    @torch.no_grad()
    def __call__(self, names: list[str], size: int) -> torch.Tensor:
        indices = [self.lookup[name] for name in names]
        grid = F.normalize(self.bank["dino_grid"][indices].float().to(self.device), dim=-1)
        glob = F.normalize(self.bank["dino_global"][indices].float().to(self.device), dim=-1)
        rgb = self.model(grid, glob).rgb32.float()
        return F.interpolate(rgb, (size, size), mode="bilinear", align_corners=False)


def make_loader(dataset, batch, shuffle, seed):
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def augment(patches, cls, noise, token_dropout, layer_dropout):
    if noise:
        patch_scale = patches.float().std((2, 3, 4), keepdim=True).clamp_min(1e-3)
        cls_scale = cls.float().std(2, keepdim=True).clamp_min(1e-3)
        patches = patches + noise * patch_scale * torch.randn_like(patches)
        cls = cls + noise * cls_scale * torch.randn_like(cls)
    if token_dropout:
        keep = torch.rand((*patches.shape[:4], 1), device=patches.device) >= token_dropout
        patches = patches * keep
    if layer_dropout:
        keep = (
            torch.rand((len(patches), patches.shape[1], 1, 1, 1), device=patches.device)
            >= layer_dropout
        )
        # Never erase all layers of a sample.
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


def train_epoch(
    model, ema, discriminator, data, g_optimizer, d_optimizer, scaler, device, args, epoch
):
    model.train()
    discriminator.train()
    totals = {}
    batches = 0
    adversarial = epoch >= args.adversarial_start
    for batch in data:
        patches = batch["patches"].to(device, non_blocking=True)
        cls = batch["cls"].to(device, non_blocking=True)
        rgb = batch["rgb"].to(device, non_blocking=True)
        patches, cls = augment(
            patches, cls, args.feature_noise, args.token_dropout, args.layer_dropout
        )
        g_optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            output = model(patches, cls)
            reconstruction, parts = advanced_color_loss(
                output,
                rgb,
                args.chroma_gradient_weight,
                args.hard_pixel_weight,
                args.spurious_chroma_edge_weight,
                args.gamut_weight,
                args.hard_pixel_fraction,
            )
        d_loss = rgb.new_zeros(())
        if adversarial:
            d_optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
                real_score = discriminator(rgb)
                fake_score = discriminator(output.rgb.detach())
                d_loss = F.relu(1 - real_score).mean() + F.relu(1 + fake_score).mean()
            scaler.scale(d_loss).backward()
            scaler.unscale_(d_optimizer)
            nn.utils.clip_grad_norm_(discriminator.parameters(), 2.0)
            scaler.step(d_optimizer)
            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
                adversarial_loss = -discriminator(output.rgb).mean()
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)
        else:
            adversarial_loss = rgb.new_zeros(())
        loss = reconstruction + args.adversarial_weight * adversarial_loss
        scaler.scale(loss).backward()
        scaler.unscale_(g_optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 4.0)
        scaler.step(g_optimizer)
        scaler.update()
        update_ema(ema, model, args.ema_decay)
        batches += 1
        values = {
            "loss": loss,
            "reconstruction": reconstruction,
            "adversarial": adversarial_loss,
            "discriminator": d_loss,
            **parts,
        }
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
    return {key: value / max(batches, 1) for key, value in totals.items()}


@torch.no_grad()
def predict(model, data, device, hard=False):
    model.eval()
    predictions = []
    oklabs = []
    linear = []
    targets = []
    names = []
    palettes = []
    for batch in data:
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            output = model(batch["patches"].to(device), batch["cls"].to(device), hard_chroma=hard)
        predictions.append(output.rgb.float().cpu())
        oklabs.append(output.oklab.float().cpu())
        linear.append(output.linear_rgb.float().cpu())
        targets.append(batch["rgb"].float())
        palettes.append(output.palette.float().cpu())
        names.extend(batch["filename"])
    return (
        torch.cat(predictions),
        torch.cat(oklabs),
        torch.cat(linear),
        torch.cat(targets),
        torch.cat(palettes),
        names,
    )


def evaluate(model, data, device, train_mean, seed=None, hard=False):
    prediction, oklab, linear, target, _, _ = predict(model, data, device, hard)
    descriptor = color_descriptor(prediction)
    output = ColorOutput(prediction, descriptor)
    if seed is not None:
        order = torch.randperm(len(prediction), generator=torch.Generator().manual_seed(seed))
        output = ColorOutput(prediction[order], descriptor[order])
        oklab = oklab[order]
        linear = linear[order]
    value = color_metrics(output, target, color_descriptor(target), train_mean)
    value.update(color_artifact_metrics(oklab, target, linear))
    return value


@torch.no_grad()
def evaluate_legacy(legacy, data, train_mean, seed=None):
    predictions = []
    targets = []
    for batch in data:
        predictions.append(legacy(list(batch["filename"]), batch["rgb"].shape[-1]).cpu())
        targets.append(batch["rgb"].float())
    prediction, target = torch.cat(predictions), torch.cat(targets)
    descriptor = color_descriptor(prediction)
    if seed is not None:
        order = torch.randperm(len(prediction), generator=torch.Generator().manual_seed(seed))
        prediction, descriptor = prediction[order], descriptor[order]
    value = color_metrics(
        ColorOutput(prediction, descriptor), target, color_descriptor(target), train_mean
    )
    value.update(color_artifact_metrics(rgb_to_oklab(prediction), target))
    return value


def selection_score(value, shuffled):
    specificity = (
        value["palette_pearson"]
        - shuffled["palette_pearson"]
        + value["descriptor_cosine"]
        - shuffled["descriptor_cosine"]
    )
    return (
        1.4 * value["palette_pearson"]
        + value["descriptor_cosine"]
        + value["rgb_gain_vs_train_mean"]
        + specificity
        - value["rgb_mae"]
        - 1.5 * value["chroma_mae"]
        + 0.2 * min(value["between_image_std_ratio"], 1.0)
        - 2.0 * value.get("chroma_edge_artifact", 0)
        - 0.35 * value.get("oklab_delta_p95", 0)
    )


def dataset_rgb_mean(dataset: Dataset, batch_size: int = 128) -> torch.Tensor:
    total = None
    images = 0
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        rgb = batch["rgb"].double()
        if total is None:
            total = torch.zeros(1, *rgb.shape[1:], dtype=torch.float64)
        total += rgb.sum(0, keepdim=True)
        images += len(rgb)
    if total is None:
        raise RuntimeError("cannot compute mean of an empty dataset")
    return (total / images).float()


def image(value, tile):
    array = (value.permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
    return Image.fromarray(array).resize((tile, tile), Image.Resampling.NEAREST)


def palette_image(palette, tile):
    from ..models.advanced_color import oklab_to_rgb

    side = round(len(palette) ** 0.5)
    value = palette[: side * side].reshape(side, side, 3).permute(2, 0, 1)[None]
    rgb = oklab_to_rgb(value)[0]
    return image(rgb, tile)


@torch.no_grad()
def save_panel(model, dataset, legacy, device, path, count):
    samples = [dataset[i] for i in range(min(count, len(dataset)))]
    if not samples:
        return
    patches = torch.stack([s["patches"] for s in samples]).to(device)
    cls = torch.stack([s["cls"] for s in samples]).to(device)
    model.eval()
    with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
        soft = model(patches, cls)
        hard = model(patches, cls, hard_chroma=True)
    names = [s["filename"] for s in samples]
    baseline = (
        legacy(names, soft.rgb.shape[-1]).cpu() if legacy else torch.zeros_like(soft.rgb.cpu())
    )
    headers = (
        "target",
        "baseline decoder",
        "color oracle soft",
        "color oracle argmax",
        "learned palette",
    )
    tile, label = 128, 22
    canvas = Image.new("RGB", (len(headers) * tile, (len(samples) + 1) * (tile + label)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, header in enumerate(headers):
        draw.text((column * tile + 3, 3), header, fill="black")
    for row, sample in enumerate(samples):
        y = (row + 1) * (tile + label)
        draw.text((3, y - label + 3), sample["filename"], fill="black")
        values = (
            image(sample["rgb"], tile),
            image(baseline[row], tile),
            image(soft.rgb[row].cpu(), tile),
            image(hard.rgb[row].cpu(), tile),
            palette_image(soft.palette[row].cpu(), tile),
        )
        for column, value in enumerate(values):
            canvas.paste(value, (column * tile, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    args = arguments()
    if args.resume and args.finetune_from:
        raise ValueError("use either --resume or --finetune-from, not both")
    if args.smoke:
        args.epochs = 1
        args.batch_size = 2
        args.eval_batch_size = 4
        args.panel_images = 2
        args.adversarial_start = 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    manifest = json.loads((Path(args.rich_bank) / "manifest.json").read_text(encoding="utf-8"))
    groups = split_names(Path(args.index))
    datasets = {
        key: RichDINOColorDataset(Path(args.rich_bank), names) for key, names in groups.items()
    }
    if args.smoke:
        available = RichDINOColorDataset(Path(args.rich_bank), manifest["filenames"])
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
        key: make_loader(
            value,
            args.batch_size if key == "train" else args.eval_batch_size,
            key == "train",
            args.seed,
        )
        for key, value in datasets.items()
    }
    sample_dataset = datasets["train"]
    if len(sample_dataset) == 0:
        raise RuntimeError("training split is empty")
    model = DINOColorOracleV2(
        input_dim=int(manifest["hidden_dim"]),
        layers=len(manifest["layers"]),
        output_size=int(manifest["rgb_size"]),
    ).to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    discriminator = PatchDiscriminator().to(device)
    g_optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    d_optimizer = torch.optim.AdamW(
        discriminator.parameters(), lr=args.discriminator_lr, weight_decay=0, betas=(0.0, 0.99)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        g_optimizer, mode="max", factor=0.5, patience=4, min_lr=2e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    legacy = None
    if Path(args.baseline_bank).exists() and Path(args.baseline_decoder).exists():
        legacy = LegacyOracle(args.baseline_bank, args.baseline_decoder, device)
    # Mean is computed strictly from training rows.
    train_rgb = dataset_rgb_mean(sample_dataset)
    baseline_validation = None
    baseline_score = None
    if legacy:
        baseline_validation = evaluate_legacy(legacy, loaders["val"], train_rgb)
        baseline_shuffled = evaluate_legacy(legacy, loaders["val"], train_rgb, args.seed)
        baseline_score = selection_score(baseline_validation, baseline_shuffled)
        (out / "baseline_validation.json").write_text(
            json.dumps(
                {
                    "score": baseline_score,
                    "validation": baseline_validation,
                    "shuffled": baseline_shuffled,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"legacy oracle score={baseline_score:.3f} rgb={baseline_validation['rgb_mae']:.3f} "
            f"palette={baseline_validation['palette_pearson']:.3f}",
            flush=True,
        )
    start = 0
    best = -math.inf
    stale = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        discriminator.load_state_dict(state["discriminator"])
        g_optimizer.load_state_dict(state["g_optimizer"])
        d_optimizer.load_state_dict(state["d_optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start = state["epoch"] + 1
        best = state.get("best_score", best)
        stale = state.get("stale", 0)
    elif args.finetune_from:
        state = torch.load(args.finetune_from, map_location="cpu", weights_only=False)
        source = state.get("ema", state["model"])
        model.load_state_dict(source)
        ema.load_state_dict(source)
        print(
            f"anti-artifact finetune initialized from {args.finetune_from} epoch={state.get('epoch')}",
            flush=True,
        )
    history = out / "history.jsonl"
    if not args.resume:
        history.unlink(missing_ok=True)
    base_dataset = (
        datasets["val"].dataset if isinstance(datasets["val"], Subset) else datasets["val"]
    )
    if args.finetune_from:
        source_validation = evaluate(ema, loaders["val"], device, train_rgb)
        source_shuffled = evaluate(ema, loaders["val"], device, train_rgb, args.seed - 1)
        best = selection_score(source_validation, source_shuffled)
        initial = {
            "epoch": -1,
            "source_epoch": state.get("epoch"),
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "model_config": model.config(),
            "discriminator": discriminator.state_dict(),
            "g_optimizer": g_optimizer.state_dict(),
            "d_optimizer": d_optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": best,
            "stale": 0,
            "finetune_from": args.finetune_from,
            "validation": source_validation,
            "rich_bank_manifest": manifest,
        }
        torch.save(initial, out / "best.pt")
        (out / "source_validation.json").write_text(
            json.dumps(
                {"score": best, "validation": source_validation, "shuffled": source_shuffled},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"source anti-artifact score={best:.3f} artifact={source_validation['chroma_edge_artifact']:.5f} "
            f"p95={source_validation['oklab_delta_p95']:.4f}",
            flush=True,
        )
    for epoch in range(start, args.epochs):
        train = train_epoch(
            model,
            ema,
            discriminator,
            loaders["train"],
            g_optimizer,
            d_optimizer,
            scaler,
            device,
            args,
            epoch,
        )
        validation = evaluate(ema, loaders["val"], device, train_rgb)
        shuffled = evaluate(ema, loaders["val"], device, train_rgb, args.seed + epoch)
        hard = evaluate(ema, loaders["val"], device, train_rgb, hard=True)
        current = selection_score(validation, shuffled)
        scheduler.step(current)
        improved = current > best + 1e-5
        stale = 0 if improved else stale + 1
        record = {
            "epoch": epoch,
            "score": current,
            "beats_legacy_oracle": baseline_score is None or current > baseline_score,
            "train": train,
            "validation": validation,
            "validation_hard": hard,
            "shuffled": shuffled,
            "lr": g_optimizer.param_groups[0]["lr"],
        }
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "model_config": model.config(),
            "discriminator": discriminator.state_dict(),
            "g_optimizer": g_optimizer.state_dict(),
            "d_optimizer": d_optimizer.state_dict(),
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
            base_dataset,
            legacy,
            device,
            out / "panels" / f"epoch_{epoch:03d}.png",
            args.panel_images,
        )
        print(
            f"epoch={epoch:03d} loss={train['loss']:.3f} score={current:.3f} rgb={validation['rgb_mae']:.3f} palette={validation['palette_pearson']:.3f} psnr={validation['psnr']:.2f}",
            flush=True,
        )
        if stale >= args.patience:
            break
    selected = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    ema.load_state_dict(selected["ema"])
    ema.to(device)
    test = evaluate(ema, loaders["test"], device, train_rgb)
    shuffled = evaluate(ema, loaders["test"], device, train_rgb, args.seed + 999)
    hard = evaluate(ema, loaders["test"], device, train_rgb, hard=True)
    baseline_test = evaluate_legacy(legacy, loaders["test"], train_rgb) if legacy else None
    report = {
        "checkpoint_epoch": selected["epoch"],
        "oracle": test,
        "oracle_hard": hard,
        "legacy_oracle": baseline_test,
        "shuffled": shuffled,
        "test_images": len(datasets["test"]),
        "untouched_full_test": not args.smoke,
    }
    (out / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
