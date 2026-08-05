"""Train the compact scene/color token interface from oracle_v2_antiring."""

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
from torch.utils.data import Subset

from .advanced_color import (
    DINOColorOracleV2,
    advanced_color_loss,
    color_artifact_metrics,
    oklab_to_rgb,
)
from .color import ColorOutput, color_descriptor, color_metrics, rgb_to_oklab
from .compact_color import CompactColorAutoencoder, CompactColorTokens
from .train_color_oracle_v2 import (
    RichDINOColorDataset,
    augment,
    dataset_rgb_mean,
    make_loader,
    selection_score,
    split_names,
    update_ema,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill oracle_v2_antiring into compact EEG-predictable color tokens"
    )
    parser.add_argument("--rich-bank", default="data/derived/rich_dino_color_v2")
    parser.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    parser.add_argument("--teacher", default="runs/dino_color_oracle_v2_antiring/best.pt")
    parser.add_argument("--output", default="runs/compact_color_tokens_antiring")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--tokenizer-depth", type=int, default=2)
    parser.add_argument("--color-tokens", type=int, default=16)
    parser.add_argument("--feature-noise", type=float, default=0.01)
    parser.add_argument("--feature-token-dropout", type=float, default=0.02)
    parser.add_argument("--feature-layer-dropout", type=float, default=0.05)
    parser.add_argument("--latent-noise", type=float, default=0.025)
    parser.add_argument("--latent-dropout", type=float, default=0.025)
    parser.add_argument("--distill-weight", type=float, default=0.65)
    parser.add_argument("--scene-weight", type=float, default=0.50)
    parser.add_argument("--swap-weight", type=float, default=0.15)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--panel-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_teacher(path: Path, device: torch.device) -> tuple[DINOColorOracleV2, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = DINOColorOracleV2(**payload["model_config"])
    model.load_state_dict(payload.get("ema", payload["model"]), strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def token_variance_loss(tokens: CompactColorTokens) -> torch.Tensor:
    values = torch.cat((tokens.scene.flatten(0, 2), tokens.color.flatten(0, 1)), 0).float()
    deviation = values.std(0, unbiased=False)
    return F.relu(0.35 - deviation).mean()


def distillation_loss(student, teacher) -> torch.Tensor:
    rgb = F.smooth_l1_loss(student.rgb, teacher.rgb, beta=0.03)
    lightness = F.smooth_l1_loss(student.oklab[:, :1], teacher.oklab[:, :1], beta=0.02)
    chroma = F.smooth_l1_loss(student.oklab[:, 1:], teacher.oklab[:, 1:], beta=0.012)
    descriptor = F.smooth_l1_loss(student.descriptor, teacher.descriptor, beta=0.025)
    return rgb + 0.45 * lightness + 1.5 * chroma + 0.35 * descriptor


def scene_structure_loss(scene_lightness: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
    target = F.adaptive_avg_pool2d(rgb_to_oklab(target_rgb)[:, :1], scene_lightness.shape[-2:])
    pixel = F.smooth_l1_loss(scene_lightness, target, beta=0.025)
    gradients = F.l1_loss(
        scene_lightness[..., 1:, :] - scene_lightness[..., :-1, :],
        target[..., 1:, :] - target[..., :-1, :],
    ) + F.l1_loss(
        scene_lightness[..., :, 1:] - scene_lightness[..., :, :-1],
        target[..., :, 1:] - target[..., :, :-1],
    )
    return pixel + 0.30 * gradients


def train_epoch(model, ema, teacher, loader, optimizer, scaler, device, args, epoch):
    model.train()
    totals = {}
    batches = 0
    robustness = min(1.0, (epoch + 1) / 8.0)
    for batch in loader:
        clean_patches = batch["patches"].to(device, non_blocking=True)
        clean_cls = batch["cls"].to(device, non_blocking=True)
        rgb = batch["rgb"].to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            teacher_output = teacher(clean_patches, clean_cls)
        patches, cls = augment(
            clean_patches,
            clean_cls,
            args.feature_noise,
            args.feature_token_dropout,
            args.feature_layer_dropout,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            forward = model(
                patches,
                cls,
                token_noise=robustness * args.latent_noise,
                token_dropout=robustness * args.latent_dropout,
            )
            reconstruction, parts = advanced_color_loss(forward.image, rgb)
            distilled = distillation_loss(forward.image, teacher_output)
            scene = scene_structure_loss(forward.scene_lightness, rgb)
            variance = token_variance_loss(forward.tokens)
            if len(rgb) > 1 and args.swap_weight:
                wrong_tokens = CompactColorTokens(
                    forward.decoded_tokens.scene, torch.roll(forward.decoded_tokens.color, 1, 0)
                )
                wrong, _ = model.decode(wrong_tokens)
                target_chroma = rgb_to_oklab(rgb)[:, 1:]
                correct_error = (forward.image.oklab[:, 1:] - target_chroma).abs().mean((1, 2, 3))
                wrong_error = (wrong.oklab[:, 1:] - target_chroma).abs().mean((1, 2, 3))
                swap = F.relu(0.008 + correct_error - wrong_error).mean()
            else:
                swap = rgb.new_zeros(())
            loss = (
                reconstruction
                + args.distill_weight * distilled
                + args.scene_weight * scene
                + args.swap_weight * swap
                + 0.03 * variance
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 4.0)
        scaler.step(optimizer)
        scaler.update()
        update_ema(ema, model, args.ema_decay)
        values = {
            "loss": loss,
            "reconstruction": reconstruction,
            "distillation": distilled,
            "scene": scene,
            "color_swap_ranking": swap,
            "token_variance": variance,
            **parts,
        }
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


@torch.no_grad()
def predict(model, loader, device, hard=False):
    model.eval()
    predictions = []
    oklabs = []
    linear = []
    targets = []
    names = []
    color_swapped = []
    for batch in loader:
        patches, cls = batch["patches"].to(device), batch["cls"].to(device)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            forward = model(patches, cls, hard_chroma=hard)
            if len(patches) > 1:
                swapped, _ = model.decode(
                    CompactColorTokens(
                        forward.tokens.scene, torch.roll(forward.tokens.color, 1, 0)
                    ),
                    hard,
                )
            else:
                swapped = forward.image
        predictions.append(forward.image.rgb.float().cpu())
        oklabs.append(forward.image.oklab.float().cpu())
        linear.append(forward.image.linear_rgb.float().cpu())
        color_swapped.append(swapped.rgb.float().cpu())
        targets.append(batch["rgb"].float())
        names.extend(batch["filename"])
    return (
        torch.cat(predictions),
        torch.cat(oklabs),
        torch.cat(linear),
        torch.cat(color_swapped),
        torch.cat(targets),
        names,
    )


def metrics_from_prediction(prediction, oklab, linear, target, train_mean):
    output = ColorOutput(prediction, color_descriptor(prediction))
    values = color_metrics(output, target, color_descriptor(target), train_mean)
    values.update(color_artifact_metrics(oklab, target, linear))
    return values


@torch.no_grad()
def evaluate(model, loader, device, train_mean, seed=0, hard=False):
    prediction, oklab, linear, color_swapped, target, _ = predict(model, loader, device, hard)
    clean = metrics_from_prediction(prediction, oklab, linear, target, train_mean)
    order = torch.randperm(len(prediction), generator=torch.Generator().manual_seed(seed))
    shuffled = metrics_from_prediction(
        prediction[order], oklab[order], linear[order], target, train_mean
    )
    swapped_oklab = rgb_to_oklab(color_swapped)
    color_swap = metrics_from_prediction(color_swapped, swapped_oklab, None, target, train_mean)
    return clean, shuffled, color_swap


@torch.no_grad()
def evaluate_teacher(model, loader, device, train_mean):
    predictions = []
    oklabs = []
    linear = []
    targets = []
    for batch in loader:
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            output = model(batch["patches"].to(device), batch["cls"].to(device))
        predictions.append(output.rgb.float().cpu())
        oklabs.append(output.oklab.float().cpu())
        linear.append(output.linear_rgb.float().cpu())
        targets.append(batch["rgb"].float())
    return metrics_from_prediction(
        torch.cat(predictions), torch.cat(oklabs), torch.cat(linear), torch.cat(targets), train_mean
    )


def image(value: torch.Tensor, tile: int) -> Image.Image:
    array = (value.permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
    return Image.fromarray(array).resize((tile, tile), Image.Resampling.NEAREST)


def palette_image(palette: torch.Tensor, tile: int) -> Image.Image:
    side = round(len(palette) ** 0.5)
    value = palette[: side * side].reshape(side, side, 3).permute(2, 0, 1)[None]
    return image(oklab_to_rgb(value)[0], tile)


@torch.no_grad()
def save_panel(model, teacher, dataset, device, path, count):
    samples = [dataset[index] for index in range(min(count, len(dataset)))]
    if not samples:
        return
    patches = torch.stack([sample["patches"] for sample in samples]).to(device)
    cls = torch.stack([sample["cls"] for sample in samples]).to(device)
    model.eval()
    with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
        reference = teacher(patches, cls)
        soft = model(patches, cls)
        hard = model(patches, cls, hard_chroma=True)
    headers = ("target", "oracle v2 antiring", "compact soft", "compact argmax", "compact palette")
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
            image(reference.rgb[row].cpu(), tile),
            image(soft.image.rgb[row].cpu(), tile),
            image(hard.image.rgb[row].cpu(), tile),
            palette_image(soft.image.palette[row].cpu(), tile),
        )
        for column, value in enumerate(values):
            canvas.paste(value, (column * tile, y))
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
    rich_root = Path(args.rich_bank)
    manifest = json.loads((rich_root / "manifest.json").read_text(encoding="utf-8"))
    groups = split_names(Path(args.index))
    datasets = {key: RichDINOColorDataset(rich_root, names) for key, names in groups.items()}
    if args.smoke:
        available = RichDINOColorDataset(rich_root, manifest["filenames"])
        if any(len(value) == 0 for value in datasets.values()):
            first = max(2, int(0.70 * len(available)))
            second = max(first + 1, int(0.85 * len(available)))
            datasets = {
                "train": Subset(available, range(first)),
                "val": Subset(available, range(first, second)),
                "test": Subset(available, range(second, len(available))),
            }
        datasets = {
            key: Subset(value, range(min(12, len(value)))) for key, value in datasets.items()
        }
    if any(len(value) == 0 for value in datasets.values()):
        raise RuntimeError("train/val/test split cannot be empty")
    loaders = {
        key: make_loader(
            value,
            args.batch_size if key == "train" else args.eval_batch_size,
            key == "train",
            args.seed,
        )
        for key, value in datasets.items()
    }
    teacher, teacher_payload = load_teacher(Path(args.teacher), device)
    teacher_config = teacher_payload["model_config"]
    if (
        teacher_config["input_dim"] != int(manifest["hidden_dim"])
        or teacher_config["layers"] != len(manifest["layers"])
        or teacher_config["output_size"] != int(manifest["rgb_size"])
    ):
        raise ValueError("teacher checkpoint and rich DINO bank are incompatible")
    model = CompactColorAutoencoder(
        input_dim=int(manifest["hidden_dim"]),
        layers=len(manifest["layers"]),
        width=args.width,
        heads=args.heads,
        tokenizer_depth=args.tokenizer_depth,
        color_tokens=args.color_tokens,
        chroma_bins=teacher_config["chroma_bins"],
        output_size=int(manifest["rgb_size"]),
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
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    train_mean = dataset_rgb_mean(datasets["train"])
    teacher_validation = evaluate_teacher(teacher, loaders["val"], device, train_mean)
    (output / "teacher_validation.json").write_text(
        json.dumps(
            {"checkpoint_epoch": teacher_payload.get("epoch"), "metrics": teacher_validation},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"teacher epoch={teacher_payload.get('epoch')} rgb={teacher_validation['rgb_mae']:.4f} "
        f"chroma={teacher_validation['chroma_mae']:.4f} psnr={teacher_validation['psnr']:.2f}",
        flush=True,
    )

    start, best, stale = 0, -math.inf, 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        if "scheduler" in state:
            scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start, best, stale = (
            state["epoch"] + 1,
            state.get("best_score", best),
            state.get("stale", 0),
        )
    history = output / "history.jsonl"
    if not args.resume:
        history.unlink(missing_ok=True)
    for epoch in range(start, args.epochs):
        train = train_epoch(
            model, ema, teacher, loaders["train"], optimizer, scaler, device, args, epoch
        )
        validation, shuffled, color_swapped = evaluate(
            ema, loaders["val"], device, train_mean, args.seed + epoch
        )
        hard, _, _ = evaluate(ema, loaders["val"], device, train_mean, args.seed + epoch, hard=True)
        current = selection_score(validation, shuffled)
        scheduler.step(current)
        improved = current > best + 1e-5
        stale = 0 if improved else stale + 1
        retention = {
            "rgb_mae_ratio": validation["rgb_mae"] / max(teacher_validation["rgb_mae"], 1e-8),
            "chroma_mae_ratio": validation["chroma_mae"]
            / max(teacher_validation["chroma_mae"], 1e-8),
            "psnr_delta": validation["psnr"] - teacher_validation["psnr"],
            "color_token_chroma_effect": color_swapped["chroma_mae"] - validation["chroma_mae"],
        }
        record = {
            "epoch": epoch,
            "score": current,
            "train": train,
            "validation": validation,
            "validation_hard": hard,
            "shuffled": shuffled,
            "color_tokens_shuffled": color_swapped,
            "teacher_retention": retention,
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
            "teacher": str(Path(args.teacher)),
            "teacher_epoch": teacher_payload.get("epoch"),
            "rich_bank_manifest": manifest,
        }
        torch.save(payload, output / "last.pt")
        if improved:
            best = current
            torch.save(payload, output / "best.pt")
        save_panel(
            ema,
            teacher,
            datasets["val"],
            device,
            output / "panels" / f"epoch_{epoch:03d}.png",
            args.panel_images,
        )
        print(
            f"epoch={epoch:03d} loss={train['loss']:.3f} score={current:.3f} "
            f"rgb={validation['rgb_mae']:.4f} chroma={validation['chroma_mae']:.4f} "
            f"psnr={validation['psnr']:.2f} color_effect={retention['color_token_chroma_effect']:.4f}",
            flush=True,
        )
        if stale >= args.patience:
            break

    selected = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    ema.load_state_dict(selected["ema"])
    ema.to(device)
    test, shuffled, color_swapped = evaluate(
        ema, loaders["test"], device, train_mean, args.seed + 999
    )
    teacher_test = evaluate_teacher(teacher, loaders["test"], device, train_mean)
    report = {
        "checkpoint_epoch": selected["epoch"],
        "compact": test,
        "teacher": teacher_test,
        "shuffled": shuffled,
        "color_tokens_shuffled": color_swapped,
        "test_images": len(datasets["test"]),
        "untouched_full_test": not args.smoke,
        "token_interface": {"scene": [8, 8, args.width], "color": [args.color_tokens, args.width]},
    }
    (output / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
