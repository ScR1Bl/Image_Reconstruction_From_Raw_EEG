from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ..data.build_pyramid_edge_targets import PYRAMID_KEYS
from ..models.model_utils import correlation_loss
from ..models.pyramid_edge import PyramidEdgeDecoder, PyramidEdgeOutput, pyramid_edge_loss


def arguments():
    parser = argparse.ArgumentParser(description="Train visual-grid oracle for pyramid edges")
    parser.add_argument("--visual-bank", default="data/derived/visual_targets_dinov2s_192.pt")
    parser.add_argument("--target-bank", default="data/derived/pyramid_edge_targets_v1.pt")
    parser.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    parser.add_argument(
        "--images-zip", default="data/things_eeg2_osf/image_set/training_images.zip"
    )
    parser.add_argument("--output", default="runs/eeg_edge_encoder/training/edge_decoder")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--panel-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def split_names(index_path, split):
    result = set()
    with Path(index_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["subject_id"]) == 0 and row["split"] == split:
                result.add(row["image_file"])
    return result


class PyramidDataset(Dataset):
    def __init__(self, visual, target, names, visual_grid=None):
        visual_lookup = {name: index for index, name in enumerate(visual["filenames"])}
        target_lookup = {name: index for index, name in enumerate(target["filenames"])}
        self.records = [
            (visual_lookup[name], target_lookup[name], name)
            for name in sorted(names)
            if name in visual_lookup
            and name in target_lookup
            and bool(target["valid"][target_lookup[name]])
        ]
        # The full float32 bank is hundreds of MB.  Share one normalized copy
        # across train/validation/test instead of materialising it three times.
        self.grid = (
            visual_grid
            if visual_grid is not None
            else F.normalize(visual["dino_grid"].float(), dim=-1)
        )
        self.target = target

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        vi, ti, name = self.records[index]
        return {
            "grid": self.grid[vi],
            "filename": name,
            **{key: self.target[key][ti].float() for key in PYRAMID_KEYS},
        }


def tolerant_f1(prediction, target, tolerance=1):
    prediction, target = prediction.bool(), target.bool()
    kernel = 2 * tolerance + 1
    pred_near = F.max_pool2d(prediction.float(), kernel, 1, tolerance).bool()
    target_near = F.max_pool2d(target.float(), kernel, 1, tolerance).bool()
    matched_pred = (prediction & target_near).sum((1, 2, 3)).float()
    matched_target = (target & pred_near).sum((1, 2, 3)).float()
    precision = matched_pred / prediction.sum((1, 2, 3)).clamp_min(1)
    recall = matched_target / target.sum((1, 2, 3)).clamp_min(1)
    return (2 * precision * recall / (precision + recall).clamp_min(1e-6)).mean()


@torch.no_grad()
def metrics(output, target):
    shape = output.shape_logits16.sigmoid() >= 0.5
    shape_target = target["shape16"] >= 0.5
    intersection = (shape & shape_target).sum((1, 2, 3)).float()
    union = (shape | shape_target).sum((1, 2, 3)).clamp_min(1)
    semantic = output.semantic_logits32.sigmoid() >= 0.30
    semantic_target = target["semantic32"] >= 0.30
    crisp = output.crisp_logits64.sigmoid() >= 0.30
    crisp_target = target["crisp64"] >= 0.5
    weight = F.max_pool2d(target["crisp64"], 5, 1, 2) * (0.25 + target["confidence64"])
    orientation = (
        (output.orientation64 * target["orientation64"]).sum(1, keepdim=True).abs() * weight
    ).sum() / weight.sum().clamp_min(1)
    return {
        "shape_iou": float((intersection / union).mean()),
        "semantic_tolerant_f1": float(tolerant_f1(semantic, semantic_target, 1)),
        "crisp_tolerant_f1": float(tolerant_f1(crisp, crisp_target, 1)),
        "perceptual_pearson": float(
            1 - correlation_loss(output.perceptual_logits64.sigmoid(), target["perceptual64"])
        ),
        "distance_correlation": float(
            1 - correlation_loss(output.distance16, target["distance16"])
        ),
        "confidence_mae": float(F.l1_loss(output.confidence64, target["confidence64"])),
        "orientation_cosine": float(orientation),
    }


def concatenate(outputs):
    fields = tuple(PyramidEdgeOutput.__dataclass_fields__)
    return type(outputs[0])(
        **{field: torch.cat([getattr(value, field) for value in outputs]) for field in fields}
    )


@torch.no_grad()
def evaluate(model, loader, device, shuffle=False, seed=0):
    model.eval()
    outputs, targets = [], {key: [] for key in PYRAMID_KEYS}
    for batch in loader:
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            value = model(batch["grid"].to(device))
        outputs.append(
            type(value)(**{field: tensor.float().cpu() for field, tensor in vars(value).items()})
        )
        for key in PYRAMID_KEYS:
            targets[key].append(batch[key].float())
    if not outputs:
        return {"empty": True}
    output = concatenate(outputs)
    target = {key: torch.cat(value) for key, value in targets.items()}
    if shuffle:
        order = torch.randperm(
            len(target["shape16"]), generator=torch.Generator().manual_seed(seed)
        )
        output = type(output)(**{field: tensor[order] for field, tensor in vars(output).items()})
    return metrics(output, target)


def train_epoch(model, loader, optimizer, device):
    model.train()
    totals = {}
    batches = 0
    for batch in loader:
        target = {key: batch[key].to(device) for key in PYRAMID_KEYS}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            output = model(batch["grid"].to(device))
            loss, parts = pyramid_edge_loss(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        values = {
            "loss": float(loss.detach()),
            **{key: float(value.detach()) for key, value in parts.items()},
        }
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


def gray(value, size=128):
    array = (value.squeeze().float().clamp(0, 1) * 255).to(torch.uint8).numpy()
    return Image.fromarray(array, "L").resize((size, size), Image.Resampling.NEAREST).convert("RGB")


def source_images(samples, images_zip, size):
    with zipfile.ZipFile(images_zip) as archive:
        members = {Path(member).name: member for member in archive.namelist()}
        result = []
        for sample in samples:
            with Image.open(io.BytesIO(archive.read(members[sample["filename"]]))) as image:
                result.append(image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS))
    return result


@torch.no_grad()
def save_panel(model, dataset, device, path, count, images_zip):
    samples = [dataset[index] for index in range(min(count, len(dataset)))]
    model.eval()
    output = model(torch.stack([sample["grid"] for sample in samples]).to(device))
    originals = source_images(samples, images_zip, 128)
    headers = (
        "original GT",
        "shape GT",
        "shape oracle",
        "semantic GT",
        "semantic oracle",
        "crisp GT",
        "crisp oracle",
        "confidence GT",
        "confidence oracle",
    )
    tile, label = 128, 22
    canvas = Image.new("RGB", (len(headers) * tile, (len(samples) + 1) * (tile + label)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, header in enumerate(headers):
        draw.text((column * tile + 3, 3), header, fill="black")
    for row, sample in enumerate(samples, 1):
        y = row * (tile + label)
        draw.text((3, y - label + 3), sample["filename"], fill="black")
        images = (
            originals[row - 1],
            gray(sample["shape16"]),
            gray(output.shape_logits16.sigmoid().cpu()[row - 1]),
            gray(sample["semantic32"]),
            gray(output.semantic_logits32.sigmoid().cpu()[row - 1]),
            gray(sample["crisp64"]),
            gray(output.crisp_logits64.sigmoid().cpu()[row - 1]),
            gray(sample["confidence64"]),
            gray(output.confidence64.cpu()[row - 1]),
        )
        for column, image in enumerate(images):
            canvas.paste(image, (column * tile, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    args = arguments()
    if args.smoke:
        args.epochs = 2
        args.patience = 2
        args.panel_images = 4
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    visual = torch.load(args.visual_bank, map_location="cpu", weights_only=False)
    target = torch.load(args.target_bank, map_location="cpu", weights_only=False)
    visual_grid = F.normalize(visual["dino_grid"].float(), dim=-1)
    datasets = {
        split: PyramidDataset(visual, target, split_names(args.index, split), visual_grid)
        for split in ("train", "val", "test")
    }
    if args.smoke:
        # A partial target bank may not cover every official split. Keep the
        # smoke test meaningful by using deterministic disjoint subsets.
        available = sorted(set(target["filenames"]))
        if any(len(dataset) == 0 for dataset in datasets.values()):
            cuts = (max(1, int(0.7 * len(available))), max(2, int(0.85 * len(available))))
            groups = {
                "train": available[: cuts[0]],
                "val": available[cuts[0] : cuts[1]],
                "test": available[cuts[1] :],
            }
            datasets = {
                key: PyramidDataset(visual, target, value, visual_grid)
                for key, value in groups.items()
            }
    loaders = {
        key: DataLoader(
            value,
            batch_size=args.batch_size if key == "train" else args.eval_batch_size,
            shuffle=key == "train",
            num_workers=0,
            drop_last=False,
        )
        for key, value in datasets.items()
    }
    model = PyramidEdgeDecoder(input_dim=visual["dino_grid"].shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    history = output_dir / "history.jsonl"
    history.unlink(missing_ok=True)
    best, stale = -math.inf, 0
    for epoch in range(args.epochs):
        begun = time.perf_counter()
        train = train_epoch(model, loaders["train"], optimizer, device)
        val = evaluate(model, loaders["val"], device)
        shuffled = evaluate(model, loaders["val"], device, True, args.seed + epoch)
        score = (
            val.get("shape_iou", 0)
            + val.get("semantic_tolerant_f1", 0)
            + val.get("crisp_tolerant_f1", 0)
            + val.get("perceptual_pearson", 0)
            + val.get("distance_correlation", 0)
        )
        improved = score > best + 1e-6
        stale = 0 if improved else stale + 1
        scheduler.step(score)
        record = {
            "epoch": epoch,
            "seconds": time.perf_counter() - begun,
            "score": score,
            "train": train,
            "validation": val,
            "validation_shuffled": shuffled,
            "lr": optimizer.param_groups[0]["lr"],
        }
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "model_config": model.config(),
            "record": record,
            "best_score": max(best, score),
        }
        torch.save(payload, output_dir / "last.pt")
        if improved:
            best = score
            torch.save(payload, output_dir / "best.pt")
        save_panel(
            model,
            datasets["val"],
            device,
            output_dir / "panels" / f"epoch_{epoch:03d}.png",
            args.panel_images,
            args.images_zip,
        )
        print(
            f"epoch={epoch:03d} loss={train['loss']:.3f} shape={val.get('shape_iou', 0):.3f} "
            f"semantic={val.get('semantic_tolerant_f1', 0):.3f} "
            f"crisp={val.get('crisp_tolerant_f1', 0):.3f}",
            flush=True,
        )
        if stale >= args.patience:
            break
    state = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    report = {
        "checkpoint_epoch": state["epoch"],
        "oracle": evaluate(model, loaders["test"], device),
        "shuffled": evaluate(model, loaders["test"], device, True, args.seed + 999),
        "test_images": len(datasets["test"]),
        "untouched_full_test": not args.smoke,
    }
    (output_dir / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
