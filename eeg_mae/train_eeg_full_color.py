"""Train raw EEG to predict the full 16x16 antiring color latent."""

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
from torch.utils.data import DataLoader, Dataset

from .advanced_color import (
    FullColorLatent,
    advanced_color_loss,
    color_artifact_metrics,
)
from .color import ColorOutput, color_descriptor, color_metrics
from .full_color_eeg import EEGFullColorLatentPredictor
from .model_utils import grid_structure_losses
from .official_200way_benchmark import cached_array
from .semantic_encoder import load_semantic_encoder
from .train_color_oracle_v2 import selection_score, update_ema
from .train_compact_color_tokens import load_teacher


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw EEG -> full 16x16 antiring color latent")
    parser.add_argument("--semantic-encoder", default="runs/eeg_semantic_encoder_v2/best.pt")
    parser.add_argument("--spatial-init", default="runs/eeg_color_encoder_v1/best.pt")
    parser.add_argument("--latent-bank", default="data/derived/full_color_latents_antiring")
    parser.add_argument("--rich-bank", default="data/derived/rich_dino_color_v2")
    parser.add_argument("--renderer", default="runs/dino_color_oracle_v2_antiring/best.pt")
    parser.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    parser.add_argument("--archives", default="data/things_eeg2_osf/preprocessed")
    parser.add_argument("--cache", default="data/derived/eeg_float32_cache")
    parser.add_argument("--output", default="runs/eeg_full_color_antiring")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--spatial-lr", type=float, default=4e-6)
    parser.add_argument("--weight-decay", type=float, default=0.015)
    parser.add_argument("--render-weight", type=float, default=0.70)
    parser.add_argument("--hard-sample-weight", type=float, default=0.45)
    parser.add_argument("--rank-weight", type=float, default=0.20)
    parser.add_argument("--ema-decay", type=float, default=0.998)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--panel-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def split_records(index_path: Path, split: str, filenames: list[str]) -> list[tuple[int, int, str]]:
    lookup = {name: index for index, name in enumerate(filenames)}
    records = {}
    with index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["subject_id"]) != 0 or row["split"] != split:
                continue
            name, condition = row["image_file"], int(row["image_index"])
            if name in lookup:
                records[condition] = (condition, lookup[name], name)
    return [records[key] for key in sorted(records)]


class EEGFullColorDataset(Dataset):
    def __init__(self, arrays, records, latent_root: Path, rgb_root: Path) -> None:
        self.arrays, self.records = arrays, records
        self.grid = np.load(latent_root / "grid.npy", mmap_mode="r")
        self.global_feature = np.load(latent_root / "global.npy", mmap_mode="r")
        rgb_manifest = json.loads((rgb_root / "manifest.json").read_text(encoding="utf-8"))
        rgb_lookup = {name: index for index, name in enumerate(rgb_manifest["filenames"])}
        self.rgb_rows = [rgb_lookup[record[2]] for record in records]
        self.rgb = np.load(rgb_root / "rgb.npy", mmap_mode="r")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        condition, latent_index, name = self.records[index]
        eeg = np.stack(
            [np.asarray(array[condition], dtype=np.float32).mean(0) for array in self.arrays]
        )
        return {
            "eeg": torch.from_numpy(eeg),
            "grid": torch.from_numpy(np.array(self.grid[latent_index], copy=True)),
            "global": torch.from_numpy(np.array(self.global_feature[latent_index], copy=True)),
            "rgb": torch.from_numpy(np.array(self.rgb[self.rgb_rows[index]], copy=True)).float()
            / 255,
            "filename": name,
        }


def latent_statistics(root: Path, indices: list[int]) -> dict[str, torch.Tensor]:
    grid = np.load(root / "grid.npy", mmap_mode="r")
    global_feature = np.load(root / "global.npy", mmap_mode="r")
    grid_sum = np.zeros(grid.shape[-1], np.float64)
    grid_square = np.zeros_like(grid_sum)
    grid_count = 0
    global_sum = np.zeros(global_feature.shape[-1], np.float64)
    global_square = np.zeros_like(global_sum)
    for start in range(0, len(indices), 128):
        chosen = indices[start : start + 128]
        local = np.asarray(grid[chosen], dtype=np.float32)
        glob = np.asarray(global_feature[chosen], dtype=np.float32)
        grid_sum += local.sum((0, 1, 2), dtype=np.float64)
        grid_square += np.square(local, dtype=np.float32).sum((0, 1, 2), dtype=np.float64)
        grid_count += np.prod(local.shape[:3])
        global_sum += glob.sum(0, dtype=np.float64)
        global_square += np.square(glob, dtype=np.float32).sum(0, dtype=np.float64)
    grid_mean = grid_sum / grid_count
    global_mean = global_sum / len(indices)
    grid_std = np.sqrt(np.maximum(grid_square / grid_count - grid_mean**2, 1e-5))
    global_std = np.sqrt(np.maximum(global_square / len(indices) - global_mean**2, 1e-5))
    return {
        "grid_mean": torch.from_numpy(grid_mean).float(),
        "grid_std": torch.from_numpy(grid_std).float(),
        "global_mean": torch.from_numpy(global_mean).float(),
        "global_std": torch.from_numpy(global_std).float(),
    }


def training_rgb_mean(dataset: EEGFullColorDataset) -> torch.Tensor:
    total = np.zeros(dataset.rgb.shape[1:], np.float64)
    for start in range(0, len(dataset.rgb_rows), 256):
        rows = dataset.rgb_rows[start : start + 256]
        total += np.asarray(dataset.rgb[rows], dtype=np.float32).sum(0, dtype=np.float64)
    return torch.from_numpy(total / (255.0 * len(dataset.rgb_rows))).float()[None]


def normalize_targets(batch, stats, device):
    grid = batch["grid"].float().to(device)
    glob = batch["global"].float().to(device)
    return (
        (grid - stats["grid_mean"]) / stats["grid_std"],
        (glob - stats["global_mean"]) / stats["global_std"],
    )


def denormalize(prediction, stats) -> FullColorLatent:
    return FullColorLatent(
        prediction.grid * stats["grid_std"] + stats["grid_mean"],
        prediction.global_feature * stats["global_std"] + stats["global_mean"],
    )


def contrastive_loss(pred_grid, pred_global, true_grid, true_global, temperature=0.10):
    prediction = F.normalize(torch.cat((pred_grid.mean((1, 2)), pred_global), -1), dim=-1)
    target = F.normalize(torch.cat((true_grid.mean((1, 2)), true_global), -1), dim=-1)
    logits = prediction @ target.T / temperature
    labels = torch.arange(len(logits), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def train_epoch(model, ema, renderer, loader, optimizer, scaler, stats, device, args):
    model.train()
    renderer.eval()
    totals = {}
    batches = 0
    for batch in loader:
        eeg, rgb = (
            batch["eeg"].to(device, non_blocking=True),
            batch["rgb"].to(device, non_blocking=True),
        )
        target_grid, target_global = normalize_targets(batch, stats, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            prediction = model.forward_ensemble(eeg)
            grid_direct = F.smooth_l1_loss(prediction.grid, target_grid, beta=0.10)
            global_direct = F.smooth_l1_loss(prediction.global_feature, target_global, beta=0.10)
            grid_cosine = (1 - F.cosine_similarity(prediction.grid, target_grid, dim=-1)).mean()
            global_cosine = (
                1 - F.cosine_similarity(prediction.global_feature, target_global, dim=-1)
            ).mean()
            residual, gradient = grid_structure_losses(prediction.grid, target_grid)
            contrastive = (
                contrastive_loss(
                    prediction.grid, prediction.global_feature, target_grid, target_global
                )
                if len(eeg) > 1
                else grid_direct * 0
            )
            sample_error = (prediction.grid - target_grid).abs().mean((1, 2, 3)) + 0.5 * (
                prediction.global_feature - target_global
            ).abs().mean(1)
            hard_count = max(1, math.ceil(0.20 * len(sample_error)))
            hard = sample_error.topk(hard_count).values.mean()
            latent = denormalize(prediction, stats)
            rendered = renderer.decode_full_latent(latent)
            render_loss, render_parts = advanced_color_loss(rendered, rgb)
            if len(eeg) > 1 and args.rank_weight:
                wrong_latent = FullColorLatent(
                    torch.roll(latent.grid, 1, 0), torch.roll(latent.global_feature, 1, 0)
                )
                wrong = renderer.decode_full_latent(wrong_latent)
                correct_error = (rendered.rgb - rgb).abs().mean((1, 2, 3))
                wrong_error = (wrong.rgb - rgb).abs().mean((1, 2, 3))
                ranking = F.relu(0.015 + correct_error - wrong_error).mean()
            else:
                ranking = grid_direct * 0
            loss = (
                grid_direct
                + 0.55 * global_direct
                + 0.45 * grid_cosine
                + 0.30 * global_cosine
                + 0.45 * residual
                + 0.25 * gradient
                + 0.20 * contrastive
                + args.hard_sample_weight * hard
                + args.render_weight * render_loss
                + args.rank_weight * ranking
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        update_ema(ema, model, args.ema_decay)
        batches += 1
        values = {
            "loss": loss,
            "grid_direct": grid_direct,
            "global_direct": global_direct,
            "grid_cosine_loss": grid_cosine,
            "global_cosine_loss": global_cosine,
            "residual": residual,
            "gradient": gradient,
            "contrastive": contrastive,
            "hard_samples": hard,
            "ranking": ranking,
            "render": render_loss,
            **{f"render_{key}": value for key, value in render_parts.items()},
        }
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
    return {key: value / max(batches, 1) for key, value in totals.items()}


@torch.no_grad()
def predict(model, renderer, loader, stats, device, hard=False):
    model.eval()
    renderer.eval()
    predictions = []
    oklabs = []
    linear = []
    targets = []
    grid_cosines = []
    global_cosines = []
    names = []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        target_grid, target_global = normalize_targets(batch, stats, device)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            output = model.forward_ensemble(eeg)
            rendered = renderer.decode_full_latent(denormalize(output, stats), hard_chroma=hard)
        predictions.append(rendered.rgb.float().cpu())
        oklabs.append(rendered.oklab.float().cpu())
        linear.append(rendered.linear_rgb.float().cpu())
        targets.append(batch["rgb"].float())
        grid_cosines.append(
            F.cosine_similarity(output.grid.float(), target_grid.float(), dim=-1).mean().cpu()
        )
        global_cosines.append(
            F.cosine_similarity(output.global_feature.float(), target_global.float(), dim=-1)
            .mean()
            .cpu()
        )
        names.extend(batch["filename"])
    return (
        torch.cat(predictions),
        torch.cat(oklabs),
        torch.cat(linear),
        torch.cat(targets),
        float(torch.stack(grid_cosines).mean()),
        float(torch.stack(global_cosines).mean()),
        names,
    )


def evaluate(model, renderer, loader, stats, device, train_mean, seed=None, hard=False):
    prediction, oklab, linear, target, grid_cosine, global_cosine, _ = predict(
        model, renderer, loader, stats, device, hard
    )
    descriptor = color_descriptor(prediction)
    if seed is not None:
        order = torch.randperm(len(prediction), generator=torch.Generator().manual_seed(seed))
        prediction, descriptor, oklab, linear = (
            prediction[order],
            descriptor[order],
            oklab[order],
            linear[order],
        )
    values = color_metrics(
        ColorOutput(prediction, descriptor), target, color_descriptor(target), train_mean
    )
    values.update(color_artifact_metrics(oklab, target, linear))
    values.update({"latent_grid_cosine": grid_cosine, "latent_global_cosine": global_cosine})
    return values


def score(metrics, shuffled):
    return (
        selection_score(metrics, shuffled)
        + 0.70 * metrics["latent_grid_cosine"]
        + 0.40 * metrics["latent_global_cosine"]
    )


def tensor_image(value, tile):
    array = (value.permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
    return Image.fromarray(array).resize((tile, tile), Image.Resampling.NEAREST)


@torch.no_grad()
def save_panel(model, renderer, dataset, stats, device, path, count):
    samples = [dataset[index] for index in range(min(count, len(dataset)))]
    if not samples:
        return
    eeg = torch.stack([sample["eeg"] for sample in samples]).to(device)
    true_grid = torch.stack([sample["grid"] for sample in samples]).float().to(device)
    true_global = torch.stack([sample["global"] for sample in samples]).float().to(device)
    model.eval()
    prediction = model.forward_ensemble(eeg)
    soft = renderer.decode_full_latent(denormalize(prediction, stats))
    hard = renderer.decode_full_latent(denormalize(prediction, stats), hard_chroma=True)
    oracle = renderer.decode_full_latent(FullColorLatent(true_grid, true_global))
    headers = ("target", "full oracle latent", "EEG full soft", "EEG full argmax")
    tile, label = 128, 22
    canvas = Image.new("RGB", (4 * tile, (len(samples) + 1) * (tile + label)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, header in enumerate(headers):
        draw.text((column * tile + 3, 3), header, fill="black")
    for row, sample in enumerate(samples):
        y = (row + 1) * (tile + label)
        draw.text((3, y - label + 3), sample["filename"], fill="black")
        values = (sample["rgb"], oracle.rgb[row].cpu(), soft.rgb[row].cpu(), hard.rgb[row].cpu())
        for column, value in enumerate(values):
            canvas.paste(tensor_image(value, tile), (column * tile, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def move_stats(stats, device):
    return {
        "grid_mean": stats["grid_mean"].to(device)[None, None, None],
        "grid_std": stats["grid_std"].to(device)[None, None, None],
        "global_mean": stats["global_mean"].to(device)[None],
        "global_std": stats["global_std"].to(device)[None],
    }


def main() -> None:
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
    latent_root = Path(args.latent_bank)
    manifest = json.loads((latent_root / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise RuntimeError("full color latent bank is incomplete")
    renderer, renderer_payload = load_teacher(Path(args.renderer), device)
    for parameter in renderer.parameters():
        parameter.requires_grad_(False)
    if renderer_payload.get("epoch") != manifest.get("teacher_epoch"):
        raise ValueError("latent bank and renderer checkpoint epochs differ")
    arrays = [
        cached_array(Path(args.archives), Path(args.cache), subject, "training")
        for subject in range(10)
    ]
    records = {
        split: split_records(Path(args.index), split, manifest["filenames"])
        for split in ("train", "val", "test")
    }
    if args.smoke and any(len(value) == 0 for value in records.values()):
        all_records = []
        for split in ("train", "val", "test"):
            all_records.extend(records[split])
        unique = {record[0]: record for record in all_records}
        all_records = [unique[key] for key in sorted(unique)]
        first = max(2, int(0.70 * len(all_records)))
        second = max(first + 1, int(0.85 * len(all_records)))
        records = {
            "train": all_records[:first],
            "val": all_records[first:second],
            "test": all_records[second:],
        }
    if args.smoke:
        records = {key: value[:16] for key, value in records.items()}
    if any(len(value) == 0 for value in records.values()):
        raise RuntimeError("train/val/test split is empty")
    datasets = {
        key: EEGFullColorDataset(arrays, value, latent_root, Path(args.rich_bank))
        for key, value in records.items()
    }
    loaders = {
        key: DataLoader(
            value,
            batch_size=args.batch_size if key == "train" else args.eval_batch_size,
            shuffle=key == "train",
            num_workers=0,
            drop_last=key == "train",
        )
        for key, value in datasets.items()
    }
    train_indices = [record[1] for record in records["train"]]
    raw_stats = latent_statistics(latent_root, train_indices)
    stats = move_stats(raw_stats, device)
    train_mean = training_rgb_mean(datasets["train"])
    semantic_encoder = load_semantic_encoder(args.semantic_encoder, device)
    model = EEGFullColorLatentPredictor(semantic_encoder, latent_dim=manifest["global_dim"]).to(
        device
    )
    initialization = None
    if args.spatial_init and Path(args.spatial_init).exists():
        initialization = model.initialize_from_spatial(
            torch.load(args.spatial_init, map_location="cpu", weights_only=False)
        )
        print(f"spatial initialization: {initialization}", flush=True)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    spatial = [
        model.spatial_norm_weight,
        model.spatial_norm_bias,
        model.spatial_weight,
        model.spatial_bias,
    ]
    spatial_ids = {id(parameter) for parameter in spatial}
    other = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in spatial_ids
    ]
    optimizer = torch.optim.AdamW(
        [{"params": spatial, "lr": args.spatial_lr}, {"params": other, "lr": args.lr}],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=3e-7
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    start, best, stale = 0, -math.inf, 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        for key in raw_stats:
            raw_stats[key] = state["latent_stats"][key]
        stats = move_stats(raw_stats, device)
        start = state["epoch"] + 1
        best, stale = state.get("best_score", best), state.get("stale", 0)
    history = output / "history.jsonl"
    if not args.resume:
        history.unlink(missing_ok=True)
    if start == 0:
        baseline = evaluate(ema, renderer, loaders["val"], stats, device, train_mean)
        baseline_shuffled = evaluate(
            ema, renderer, loaders["val"], stats, device, train_mean, args.seed
        )
        best = score(baseline, baseline_shuffled)
        initial = {
            "epoch": -1,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "model_config": model.config(),
            "best_score": best,
            "stale": 0,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "latent_stats": raw_stats,
            "renderer": args.renderer,
            "latent_bank_manifest": manifest,
            "initialization": initialization,
            "record": {"validation": baseline, "shuffled": baseline_shuffled},
        }
        torch.save(initial, output / "best.pt")
        print(f"mean-latent baseline score={best:.3f} rgb={baseline['rgb_mae']:.3f}", flush=True)
    for epoch in range(start, args.epochs):
        train = train_epoch(
            model, ema, renderer, loaders["train"], optimizer, scaler, stats, device, args
        )
        validation = evaluate(ema, renderer, loaders["val"], stats, device, train_mean)
        shuffled = evaluate(
            ema, renderer, loaders["val"], stats, device, train_mean, args.seed + epoch + 1
        )
        hard = evaluate(ema, renderer, loaders["val"], stats, device, train_mean, hard=True)
        current = score(validation, shuffled)
        improved = current > best + 1e-5
        stale = 0 if improved else stale + 1
        scheduler.step(current)
        record = {
            "epoch": epoch,
            "score": current,
            "train": train,
            "validation": validation,
            "validation_hard": hard,
            "shuffled": shuffled,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
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
            "latent_stats": raw_stats,
            "renderer": args.renderer,
            "renderer_epoch": renderer_payload.get("epoch"),
            "latent_bank_manifest": manifest,
            "initialization": initialization,
        }
        torch.save(payload, output / "last.pt")
        if improved:
            best = current
            torch.save(payload, output / "best.pt")
        save_panel(
            ema,
            renderer,
            datasets["val"],
            stats,
            device,
            output / "panels" / f"epoch_{epoch:03d}.png",
            args.panel_images,
        )
        print(
            f"epoch={epoch:03d} loss={train['loss']:.3f} score={current:.3f} "
            f"grid={validation['latent_grid_cosine']:.3f} global={validation['latent_global_cosine']:.3f} "
            f"rgb={validation['rgb_mae']:.3f} palette={validation['palette_pearson']:.3f}",
            flush=True,
        )
        if stale >= args.patience:
            break
    selected = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    ema.load_state_dict(selected["ema"])
    ema.to(device)
    test = evaluate(ema, renderer, loaders["test"], stats, device, train_mean)
    shuffled = evaluate(ema, renderer, loaders["test"], stats, device, train_mean, args.seed + 999)
    report = {
        "checkpoint_epoch": selected["epoch"],
        "eeg_full_color": test,
        "shuffled": shuffled,
        "test_images": len(datasets["test"]),
        "untouched_full_test": not args.smoke,
        "latent_interface": {"grid": manifest["grid_shape"], "global": manifest["global_dim"]},
        "renderer_epoch": renderer_payload.get("epoch"),
    }
    (output / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
