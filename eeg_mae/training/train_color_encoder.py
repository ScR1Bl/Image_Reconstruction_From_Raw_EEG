"""Train the independent raw-EEG color encoder by direct task supervision."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..evaluation.official_200way_benchmark import cached_array
from ..models.advanced_color import FullColorLatent, advanced_color_loss, color_artifact_metrics
from ..models.color import (
    ColorOutput,
    color_descriptor,
    color_metrics,
    rendered_descriptor_contrastive,
    rgb_to_oklab,
)
from ..models.color_specialist import EEGColorSpecialist
from ..models.model_utils import variance_covariance_regularization
from ..models.task_adapter import cvar, shuffled_ranking_loss
from .color_training import (
    EEGColorDataset,
    load_color_oracle,
    split_records,
    training_rgb_mean,
)
from .train_color_oracle import selection_score, update_ema


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent color-specialized raw EEG encoder")
    parser.add_argument("--latent-bank", default="data/derived/full_color_latents_antiring")
    parser.add_argument("--rich-bank", default="data/derived/rich_dino_color_v2")
    parser.add_argument(
        "--oracle-renderer", default="runs/eeg_color_encoder/components/color_oracle.pt"
    )
    parser.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    parser.add_argument("--archives", default="data/things_eeg2_osf/preprocessed")
    parser.add_argument("--cache", default="data/derived/eeg_float32_cache")
    parser.add_argument("--output", default="runs/eeg_color_encoder")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.015)
    parser.add_argument("--patience", type=int, default=9)
    parser.add_argument("--ema-decay", type=float, default=0.998)
    parser.add_argument("--rank-margin", type=float, default=0.035)
    parser.add_argument("--panel-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-from", default=None, help="warm-start matching EEG encoder weights")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def color_sample_error(rgb: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction_oklab, target_oklab = rgb_to_oklab(rgb), rgb_to_oklab(target)
    chroma = (prediction_oklab[:, 1:] - target_oklab[:, 1:]).abs().mean((1, 2, 3))
    palette = (
        (F.adaptive_avg_pool2d(rgb, 4) - F.adaptive_avg_pool2d(target, 4)).abs().mean((1, 2, 3))
    )
    pixel = (rgb - target).abs().mean((1, 2, 3))
    return 1.3 * chroma + 0.8 * palette + 0.25 * pixel


def rare_chroma_loss(logits: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
    target = F.adaptive_avg_pool2d(rgb_to_oklab(target_rgb)[:, 1:], 16)
    bins = logits.shape[2]
    indices = ((target + 0.4) / 0.8 * (bins - 1)).round().long().clamp(0, bins - 1)
    magnitude = target.square().sum(1).sqrt()
    weights = 1 + 2 * (magnitude / 0.25).clamp(0, 1)
    losses = [
        F.cross_entropy(logits[:, channel], indices[:, channel], reduction="none")
        for channel in range(2)
    ]
    return sum((loss * weights).mean() for loss in losses) / 2


def train_epoch(model, ema, oracle, loader, optimizer, scaler, device, args):
    model.train()
    totals = {}
    batches = 0
    for batch in loader:
        eeg, target = (
            batch["eeg"].to(device, non_blocking=True),
            batch["rgb"].to(device, non_blocking=True),
        )
        true_latent = FullColorLatent(
            batch["grid"].float().to(device), batch["global"].float().to(device)
        )
        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            teacher = oracle.decode_full_latent(true_latent)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            output = model(eeg)
            reconstruction, parts = advanced_color_loss(output.image, target)
            oracle_chroma = F.smooth_l1_loss(
                output.image.oklab[:, 1:], teacher.oklab[:, 1:], beta=0.012
            )
            oracle_palette = F.smooth_l1_loss(
                F.adaptive_avg_pool2d(output.image.rgb, 4),
                F.adaptive_avg_pool2d(teacher.rgb, 4),
                beta=0.025,
            )
            descriptor_target = color_descriptor(target)
            descriptor = F.smooth_l1_loss(output.condition.descriptor, descriptor_target, beta=0.03)
            contrastive = (
                rendered_descriptor_contrastive(output.image.rgb, descriptor_target)
                if len(eeg) > 1
                else reconstruction * 0
            )
            error = color_sample_error(output.image.rgb, target)
            task = error.mean()
            hard = cvar(error)
            order = torch.roll(torch.arange(len(target), device=device), 1)
            wrong_error = color_sample_error(output.image.rgb, target[order])
            ranking = (
                shuffled_ranking_loss(error, wrong_error, args.rank_margin)
                if len(eeg) > 1
                else task * 0
            )
            rare = rare_chroma_loss(output.image.chroma_logits16, target)
            target_std = target.flatten(1).std(0, unbiased=False).mean()
            output_std = output.image.rgb.flatten(1).std(0, unbiased=False).mean()
            diversity = F.relu(0.72 * target_std - output_std)
            representation = output.condition.grid16.mean((1, 2))
            variance, covariance = variance_covariance_regularization(representation)
            spatial_spread = F.relu(0.35 - output.condition.grid16.std((1, 2)).mean())
            loss = (
                reconstruction
                + 0.55 * oracle_chroma
                + 0.30 * oracle_palette
                + 0.30 * descriptor
                + 0.15 * contrastive
                + 0.45 * task
                + 0.65 * hard
                + 0.40 * ranking
                + 0.22 * rare
                + 0.40 * diversity
                + 0.20 * variance
                + 0.04 * covariance
                + 0.10 * spatial_spread
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 4.0)
        scaler.step(optimizer)
        scaler.update()
        update_ema(ema, model, args.ema_decay)
        batches += 1
        values = {
            "loss": loss,
            "task": task,
            "hard": hard,
            "ranking": ranking,
            "oracle_chroma": oracle_chroma,
            "oracle_palette": oracle_palette,
            "descriptor_direct": descriptor,
            "contrastive": contrastive,
            "rare_chroma": rare,
            "diversity": diversity,
            "representation_variance": variance,
            "representation_covariance": covariance,
            "spatial_spread": spatial_spread,
            **parts,
        }
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
    return {key: value / max(1, batches) for key, value in totals.items()}


@torch.no_grad()
def predict(model, loader, device, hard=False):
    model.eval()
    predictions = []
    targets = []
    oklabs = []
    linear = []
    names = []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            output = model(eeg, hard_chroma=hard)
        predictions.append(output.image.rgb.float().cpu())
        oklabs.append(output.image.oklab.float().cpu())
        linear.append(output.image.linear_rgb.float().cpu())
        targets.append(batch["rgb"].float())
        names.extend(batch["filename"])
    return (
        torch.cat(predictions),
        torch.cat(oklabs),
        torch.cat(linear),
        torch.cat(targets),
        names,
    )


def image_metrics(prediction, target, train_mean, oklab=None, linear=None):
    values = color_metrics(
        ColorOutput(prediction, color_descriptor(prediction)),
        target,
        color_descriptor(target),
        train_mean,
    )
    values.update(
        color_artifact_metrics(rgb_to_oklab(prediction) if oklab is None else oklab, target, linear)
    )
    return values


def evaluate(model, loader, device, train_mean, seed=None, hard=False):
    prediction, oklab, linear, target, _ = predict(model, loader, device, hard)
    clean = image_metrics(prediction, target, train_mean, oklab, linear)
    if seed is None:
        return clean
    order = torch.randperm(len(prediction), generator=torch.Generator().manual_seed(seed))
    shuffled = image_metrics(prediction[order], target, train_mean, oklab[order], linear[order])
    return clean, shuffled


def encoder_score(value, shuffled):
    return selection_score(value, shuffled)


def to_image(value, tile):
    array = (value.permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
    return Image.fromarray(array).resize((tile, tile), Image.Resampling.NEAREST)


@torch.no_grad()
def save_panel(model, oracle, dataset, device, path, count):
    samples = [dataset[index] for index in range(min(count, len(dataset)))]
    if not samples:
        return
    eeg = torch.stack([sample["eeg"] for sample in samples]).to(device)
    grid = torch.stack([sample["grid"] for sample in samples]).float().to(device)
    glob = torch.stack([sample["global"] for sample in samples]).float().to(device)
    model.eval()
    soft = model(eeg)
    hard = model(eeg, hard_chroma=True)
    teacher = oracle.decode_full_latent(FullColorLatent(grid, glob))
    headers = (
        "target",
        "color direct",
        "encoder argmax",
        "oracle",
        "error x2",
    )
    tile, label = 128, 22
    canvas = Image.new("RGB", (len(headers) * tile, (len(samples) + 1) * (tile + label)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, header in enumerate(headers):
        draw.text((column * tile + 3, 3), header, fill="black")
    for row, sample in enumerate(samples):
        y = (row + 1) * (tile + label)
        draw.text((3, y - label + 3), sample["filename"], fill="black")
        difference = (soft.image.rgb[row] - sample["rgb"].to(device)).abs().mul(2).clamp(0, 1)
        values = (
            sample["rgb"],
            soft.image.rgb[row].cpu(),
            hard.image.rgb[row].cpu(),
            teacher.rgb[row].cpu(),
            difference.cpu(),
        )
        for column, value in enumerate(values):
            canvas.paste(to_image(value, tile), (column * tile, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


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
    arrays = [
        cached_array(Path(args.archives), Path(args.cache), subject, "training")
        for subject in range(10)
    ]
    records = {
        split: split_records(Path(args.index), split, manifest["filenames"])
        for split in ("train", "val", "test")
    }
    if args.smoke:
        records = {key: value[:16] for key, value in records.items()}
    datasets = {
        key: EEGColorDataset(arrays, value, latent_root, Path(args.rich_bank))
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
    train_mean = training_rgb_mean(datasets["train"])
    oracle, oracle_payload = load_color_oracle(Path(args.oracle_renderer), device)
    for parameter in oracle.parameters():
        parameter.requires_grad_(False)
    model = EEGColorSpecialist().to(device)
    if args.init_from and not args.resume:
        warm = torch.load(args.init_from, map_location="cpu", weights_only=False)
        source = warm.get("ema", warm["model"])
        current = model.state_dict()
        matching = {
            key: value
            for key, value in source.items()
            if key.startswith("encoder.") and key in current and current[key].shape == value.shape
        }
        model.load_state_dict(matching, strict=False)
        print(f"warm-started {len(matching)} color EEG tensors from {args.init_from}", flush=True)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=2e-6
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
        start, best, stale = state["epoch"] + 1, state["best_score"], state.get("stale", 0)
    history = output / "history.jsonl"
    if not args.resume:
        history.unlink(missing_ok=True)
    initial, initial_shuffled = evaluate(ema, loaders["val"], device, train_mean, args.seed)
    initial_score = encoder_score(initial, initial_shuffled)
    if start == 0:
        best = initial_score
        payload = {
            "epoch": -1,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "model_config": model.config(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": best,
            "stale": 0,
            "oracle_renderer": args.oracle_renderer,
            "latent_bank_manifest": manifest,
            "architecture": "independent_direct_color_v1",
            "record": {"validation": initial, "shuffled": initial_shuffled},
        }
        torch.save(payload, output / "best.pt")
        print(
            f"initial encoder={best:.3f} rgb={initial['rgb_mae']:.3f}",
            flush=True,
        )
    for epoch in range(start, args.epochs):
        train = train_epoch(model, ema, oracle, loaders["train"], optimizer, scaler, device, args)
        validation, shuffled = evaluate(
            ema, loaders["val"], device, train_mean, args.seed + epoch + 1
        )
        hard = evaluate(ema, loaders["val"], device, train_mean, hard=True)
        current = encoder_score(validation, shuffled)
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
            "oracle_renderer": args.oracle_renderer,
            "oracle_epoch": oracle_payload.get("epoch"),
            "latent_bank_manifest": manifest,
            "architecture": "independent_direct_color_v1",
        }
        torch.save(payload, output / "last.pt")
        if improved:
            best = current
            torch.save(payload, output / "best.pt")
        save_panel(
            ema,
            oracle,
            datasets["val"],
            device,
            output / "panels" / f"epoch_{epoch:03d}.png",
            args.panel_images,
        )
        print(
            f"epoch={epoch:03d} loss={train['loss']:.3f} score={current:.3f} "
            f"rgb={validation['rgb_mae']:.3f} "
            f"chroma={validation['chroma_mae']:.3f} palette={validation['palette_pearson']:.3f} "
            f"var={train['representation_variance']:.3f}",
            flush=True,
        )
        if stale >= args.patience:
            break
    selected = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    ema.load_state_dict(selected["ema"])
    ema.to(device)
    test, shuffled = evaluate(ema, loaders["test"], device, train_mean, args.seed + 999)
    report = {
        "checkpoint_epoch": selected["epoch"],
        "color_encoder": test,
        "shuffled": shuffled,
        "validation_selected_score": selected["best_score"],
        "test_images": len(datasets["test"]),
        "untouched_full_test": not args.smoke,
        "inference_inputs": ["preprocessed_eeg"],
        "architecture": "independent_direct_color_v1",
    }
    (output / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
