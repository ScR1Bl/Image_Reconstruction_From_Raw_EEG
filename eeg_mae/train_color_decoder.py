"""Train the image-feature-to-color oracle used by the EEG color encoder."""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, Dataset

from .color import ColorDecoder, ColorOutput, color_loss, color_metrics


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DINO-to-low-frequency-color oracle")
    parser.add_argument("--visual-bank", default="data/derived/visual_targets_dinov2s_192.pt")
    parser.add_argument("--rgb-bank", default="data/derived/rich_visual_targets_32.pt")
    parser.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    parser.add_argument("--output", default="runs/eeg_color_encoder/training/color_decoder")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--feature-noise", type=float, default=0.025)
    parser.add_argument("--feature-dropout", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--panel-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def split_names(index_path: Path) -> dict[str, list[str]]:
    groups = {"train": set(), "val": set(), "test": set()}
    with index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["subject_id"]) == 0 and row["split"] in groups:
                groups[row["split"]].add(row["image_file"])
    return {key: sorted(value) for key, value in groups.items()}


class ColorDataset(Dataset):
    def __init__(self, visual: dict, rich: dict, names: list[str]) -> None:
        visual_lookup = {name: index for index, name in enumerate(visual["filenames"])}
        rich_lookup = {name: index for index, name in enumerate(rich["filenames"])}
        self.rows = [
            (visual_lookup[name], rich_lookup[name], name)
            for name in names
            if name in visual_lookup and name in rich_lookup
        ]
        self.grid = visual["dino_grid"]
        self.global_feature = visual["dino_global"]
        self.descriptor = visual["color"]
        self.rgb = rich["rgb"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        visual_index, rich_index, name = self.rows[index]
        return {
            "grid": self.grid[visual_index].float(),
            "global": self.global_feature[visual_index].float(),
            "descriptor": self.descriptor[visual_index].float(),
            "rgb": self.rgb[rich_index].float(),
            "filename": name,
        }


def loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )


def augment_features(
    grid: torch.Tensor, global_feature: torch.Tensor, noise: float, dropout: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if noise:
        grid = grid + noise * torch.randn_like(grid)
        global_feature = global_feature + noise * torch.randn_like(global_feature)
    if dropout:
        keep = torch.rand((*grid.shape[:3], 1), device=grid.device) >= dropout
        grid = grid * keep
    return F.normalize(grid, dim=-1), F.normalize(global_feature, dim=-1)


def train_epoch(model, data, optimizer, device, noise, dropout):
    model.train()
    totals = {}
    batches = 0
    for batch in data:
        grid = batch["grid"].to(device)
        global_feature = batch["global"].to(device)
        rgb = batch["rgb"].to(device)
        descriptor = batch["descriptor"].to(device)
        grid, global_feature = augment_features(grid, global_feature, noise, dropout)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            output = model(grid, global_feature)
            loss, parts = color_loss(output, rgb, descriptor)
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


@torch.no_grad()
def evaluate(model, data, device, train_mean, shuffle_seed: int | None = None):
    model.eval()
    predictions, descriptors, rgbs = [], [], []
    for batch in data:
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            output = model(
                F.normalize(batch["grid"].to(device), dim=-1),
                F.normalize(batch["global"].to(device), dim=-1),
            )
        predictions.append(output.rgb32.float().cpu())
        descriptors.append(output.descriptor.float().cpu())
        rgbs.append(batch["rgb"].float())
    output = ColorOutput(torch.cat(predictions), torch.cat(descriptors))
    rgb = torch.cat(rgbs)
    descriptor = torch.cat([batch["descriptor"].float() for batch in data])
    if shuffle_seed is not None:
        order = torch.randperm(len(rgb), generator=torch.Generator().manual_seed(shuffle_seed))
        output = ColorOutput(output.rgb32[order], output.descriptor[order])
    return color_metrics(output, rgb, descriptor, train_mean)


def selection_score(metrics: dict[str, float]) -> float:
    return (
        metrics["palette_pearson"]
        + metrics["descriptor_cosine"]
        + 2.0 * metrics["descriptor_top1"]
        + metrics.get("rgb_gain_vs_train_mean", 0)
        - metrics["rgb_mae"]
        - 1.5 * metrics["chroma_mae"]
        + 0.25 * min(metrics["between_image_std_ratio"], 1.0)
    )


@torch.no_grad()
def save_panel(model, dataset, device, path: Path, count: int) -> None:
    samples = [dataset[index] for index in range(min(count, len(dataset)))]
    if not samples:
        return
    grid = torch.stack([sample["grid"] for sample in samples]).to(device)
    global_feature = torch.stack([sample["global"] for sample in samples]).to(device)
    model.eval()
    output = model(F.normalize(grid, dim=-1), F.normalize(global_feature, dim=-1))
    tile, label = 160, 22
    canvas = Image.new("RGB", (2 * tile, len(samples) * (tile + label)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, sample in enumerate(samples):
        y = row * (tile + label)
        draw.text((3, y + 3), str(sample["filename"]), fill="black")
        for column, tensor in enumerate((sample["rgb"], output.rgb32[row].cpu())):
            array = (tensor.permute(1, 2, 0).float().clamp(0, 1) * 255).byte().numpy()
            image = Image.fromarray(array).resize((tile, tile), Image.Resampling.NEAREST)
            canvas.paste(image, (column * tile, y + label))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    args = arguments()
    if args.smoke:
        args.epochs = 1
        args.batch_size = 4
        args.eval_batch_size = 8
        args.panel_images = 2
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    visual = torch.load(args.visual_bank, map_location="cpu", weights_only=False)
    rich = torch.load(args.rgb_bank, map_location="cpu", weights_only=False)
    groups = split_names(Path(args.index))
    datasets = {key: ColorDataset(visual, rich, names) for key, names in groups.items()}
    if args.smoke:
        datasets = {
            key: torch.utils.data.Subset(value, range(min(32, len(value))))
            for key, value in datasets.items()
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
    train_indices = [row[1] for row in ColorDataset(visual, rich, groups["train"]).rows]
    train_mean = rich["rgb"][train_indices].float().mean(0, keepdim=True)
    model = ColorDecoder(
        grid_dim=visual["dino_grid"].shape[-1],
        global_dim=visual["dino_global"].shape[-1],
        descriptor_dim=visual["color"].shape[-1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=3e-6
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    history = output_dir / "history.jsonl"
    history.unlink(missing_ok=True)
    best, stale = -math.inf, 0
    for epoch in range(args.epochs):
        train = train_epoch(
            model, loaders["train"], optimizer, device, args.feature_noise, args.feature_dropout
        )
        validation = evaluate(model, loaders["val"], device, train_mean)
        shuffled = evaluate(model, loaders["val"], device, train_mean, args.seed + epoch)
        score = selection_score(validation)
        improved = score > best + 1e-6
        stale = 0 if improved else stale + 1
        scheduler.step(score)
        record = {
            "epoch": epoch,
            "score": score,
            "train": train,
            "validation": validation,
            "validation_shuffled": shuffled,
            "lr": optimizer.param_groups[0]["lr"],
        }
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "model_config": model.config(),
            "best_score": max(best, score),
            "train_mean": train_mean,
            "record": record,
        }
        torch.save(payload, output_dir / "last.pt")
        if improved:
            best = score
            torch.save(payload, output_dir / "best.pt")
        base_dataset = (
            datasets["val"].dataset
            if isinstance(datasets["val"], torch.utils.data.Subset)
            else datasets["val"]
        )
        save_panel(
            model,
            base_dataset,
            device,
            output_dir / "panels" / f"epoch_{epoch:03d}.png",
            args.panel_images,
        )
        print(
            f"epoch={epoch:03d} loss={train['loss']:.3f} score={score:.3f} "
            f"palette={validation['palette_pearson']:.3f} "
            f"gain={validation['rgb_gain_vs_train_mean']:.3f}",
            flush=True,
        )
        if stale >= args.patience:
            break
    state = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    report = {
        "checkpoint_epoch": state["epoch"],
        "oracle": evaluate(model, loaders["test"], device, train_mean),
        "shuffled": evaluate(model, loaders["test"], device, train_mean, args.seed + 999),
        "test_images": len(datasets["test"]),
        "untouched_full_test": not args.smoke,
    }
    (output_dir / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
