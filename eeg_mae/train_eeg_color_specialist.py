"""Train a color-only EEG specialist on top of the stable adapter-v2 baseline."""

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

from .advanced_color import FullColorLatent, advanced_color_loss, color_artifact_metrics
from .color import (
    ColorOutput,
    color_descriptor,
    color_metrics,
    load_color_decoder,
    rendered_descriptor_contrastive,
    rgb_to_oklab,
)
from .color_specialist import EEGColorSpecialist
from .official_200way_benchmark import cached_array
from .semantic_edge import EEGDINOGridPredictor
from .semantic_encoder import load_semantic_encoder
from .task_adapter import TaskResidualAdapter, cvar, improvement_loss, shuffled_ranking_loss
from .train_color_oracle_v2 import selection_score, update_ema
from .train_compact_color_tokens import load_teacher
from .train_eeg_color import semantic_ensemble
from .train_eeg_full_color import EEGFullColorDataset, split_records, training_rgb_mean


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent color-specialized raw EEG encoder")
    parser.add_argument("--adapter-v2", default="runs/eeg_color_encoder/components/adapter.pt")
    parser.add_argument(
        "--semantic-encoder", default="runs/eeg_color_encoder/components/semantic_encoder.pt"
    )
    parser.add_argument(
        "--color-decoder", default="runs/eeg_color_encoder/components/color_decoder.pt"
    )
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
    parser.add_argument("--margin", type=float, default=0.008)
    parser.add_argument("--rank-margin", type=float, default=0.035)
    parser.add_argument("--panel-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


class FrozenAdapterV2:
    def __init__(
        self, path: Path, semantic_path: str, decoder_path: str, device: torch.device
    ) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        encoder = load_semantic_encoder(semantic_path, device)
        self.predictor = EEGDINOGridPredictor(encoder, **payload["predictor_config"]).to(device)
        self.predictor.load_state_dict(payload["predictor"], strict=True)
        self.adapter = TaskResidualAdapter(**payload["adapter_config"]).to(device)
        self.adapter.load_state_dict(payload["adapter"], strict=True)
        self.decoder, self.decoder_payload = load_color_decoder(decoder_path, device)
        self.payload = payload
        for model in (self.predictor, self.adapter, self.decoder):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        grid, contexts = self.predictor.forward_ensemble(eeg)
        global_feature = semantic_ensemble(self.predictor, eeg)
        adapted = self.adapter(grid, contexts, global_feature)
        return self.decoder(adapted.grid, adapted.global_feature).rgb32.float()


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


def residual_smoothness(delta: torch.Tensor) -> torch.Tensor:
    return (delta[..., 1:, :] - delta[..., :-1, :]).abs().mean() + (
        delta[..., :, 1:] - delta[..., :, :-1]
    ).abs().mean()


def train_epoch(model, ema, baseline, oracle, loader, optimizer, scaler, device, args):
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
            base32 = baseline(eeg)
            base = F.interpolate(base32, target.shape[-2:], mode="bilinear", align_corners=False)
            teacher = oracle.decode_full_latent(true_latent)
            base_error = color_sample_error(base, target)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            output = model(eeg, base32)
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
            improve = improvement_loss(error, base_error, args.margin)
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
            smooth = residual_smoothness(output.chroma_delta16)
            delta = output.chroma_delta16.square().mean()
            alpha_open = F.relu(0.06 - output.alpha16.mean())
            loss = (
                reconstruction
                + 0.55 * oracle_chroma
                + 0.30 * oracle_palette
                + 0.30 * descriptor
                + 0.15 * contrastive
                + 0.45 * task
                + 0.65 * hard
                + 1.6 * improve
                + 0.40 * ranking
                + 0.22 * rare
                + 0.40 * diversity
                + 0.04 * smooth
                + 0.004 * delta
                + 0.08 * alpha_open
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
            "improvement": improve,
            "ranking": ranking,
            "oracle_chroma": oracle_chroma,
            "oracle_palette": oracle_palette,
            "descriptor_direct": descriptor,
            "contrastive": contrastive,
            "rare_chroma": rare,
            "diversity": diversity,
            "residual_smoothness": smooth,
            "residual_energy": delta,
            "alpha": output.alpha16.mean(),
            **parts,
        }
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
    return {key: value / max(1, batches) for key, value in totals.items()}


@torch.no_grad()
def predict(model, baseline, loader, device, hard=False):
    model.eval()
    predictions = []
    baselines = []
    targets = []
    oklabs = []
    linear = []
    names = []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            base32 = baseline(eeg)
            output = model(eeg, base32, hard_chroma=hard)
        predictions.append(output.image.rgb.float().cpu())
        oklabs.append(output.image.oklab.float().cpu())
        linear.append(output.image.linear_rgb.float().cpu())
        baselines.append(
            F.interpolate(
                base32.float(), output.image.rgb.shape[-2:], mode="bilinear", align_corners=False
            ).cpu()
        )
        targets.append(batch["rgb"].float())
        names.extend(batch["filename"])
    return (
        torch.cat(predictions),
        torch.cat(oklabs),
        torch.cat(linear),
        torch.cat(baselines),
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


def evaluate(model, baseline, loader, device, train_mean, seed=None, hard=False):
    prediction, oklab, linear, base, target, _ = predict(model, baseline, loader, device, hard)
    clean = image_metrics(prediction, target, train_mean, oklab, linear)
    baseline_metrics = image_metrics(base, target, train_mean)
    if seed is None:
        return clean, baseline_metrics
    order = torch.randperm(len(prediction), generator=torch.Generator().manual_seed(seed))
    shuffled = image_metrics(prediction[order], target, train_mean, oklab[order], linear[order])
    return clean, baseline_metrics, shuffled


def specialist_score(value, shuffled, adapter):
    """Prefer measurable visual improvement over descriptor-only gains."""
    return (
        selection_score(value, shuffled)
        + 2.0 * (adapter["rgb_mae"] - value["rgb_mae"])
        + 3.0 * (adapter["chroma_mae"] - value["chroma_mae"])
        + 0.8 * (value["palette_pearson"] - adapter["palette_pearson"])
        - 2.0 * max(0.0, value["chroma_edge_artifact"] - adapter["chroma_edge_artifact"])
    )


def strictly_beats_adapter(value, adapter, score_value, adapter_score):
    return (
        score_value > adapter_score
        and value["rgb_mae"] < adapter["rgb_mae"]
        and value["chroma_mae"] < adapter["chroma_mae"]
        and value["palette_pearson"] > adapter["palette_pearson"]
        and value["chroma_edge_artifact"] <= adapter["chroma_edge_artifact"] + 5e-5
    )


def to_image(value, tile):
    array = (value.permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
    return Image.fromarray(array).resize((tile, tile), Image.Resampling.NEAREST)


@torch.no_grad()
def save_panel(model, baseline, oracle, dataset, device, path, count):
    samples = [dataset[index] for index in range(min(count, len(dataset)))]
    if not samples:
        return
    eeg = torch.stack([sample["eeg"] for sample in samples]).to(device)
    grid = torch.stack([sample["grid"] for sample in samples]).float().to(device)
    glob = torch.stack([sample["global"] for sample in samples]).float().to(device)
    base32 = baseline(eeg)
    base = F.interpolate(base32, (64, 64), mode="bilinear", align_corners=False)
    model.eval()
    soft = model(eeg, base32)
    hard = model(eeg, base32, hard_chroma=True)
    teacher = oracle.decode_full_latent(FullColorLatent(grid, glob))
    headers = (
        "target",
        "adapter v2",
        "color specialist",
        "specialist argmax",
        "oracle",
        "difference x4",
    )
    tile, label = 128, 22
    canvas = Image.new("RGB", (len(headers) * tile, (len(samples) + 1) * (tile + label)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, header in enumerate(headers):
        draw.text((column * tile + 3, 3), header, fill="black")
    for row, sample in enumerate(samples):
        y = (row + 1) * (tile + label)
        draw.text((3, y - label + 3), sample["filename"], fill="black")
        difference = (soft.image.rgb[row] - base[row]).abs().mul(4).clamp(0, 1)
        values = (
            sample["rgb"],
            base[row].cpu(),
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
    train_mean = training_rgb_mean(datasets["train"])
    baseline = FrozenAdapterV2(
        Path(args.adapter_v2), args.semantic_encoder, args.color_decoder, device
    )
    oracle, oracle_payload = load_teacher(Path(args.oracle_renderer), device)
    for parameter in oracle.parameters():
        parameter.requires_grad_(False)
    model = EEGColorSpecialist().to(device)
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
    initial, adapter_validation, initial_shuffled = evaluate(
        ema, baseline, loaders["val"], device, train_mean, args.seed
    )
    initial_score = specialist_score(initial, initial_shuffled, adapter_validation)
    # Evaluate the frozen adapter with its own shuffled output for an apples-to-apples threshold.
    _, _, _, adapter_images, targets, _ = predict(ema, baseline, loaders["val"], device)
    order = torch.randperm(len(adapter_images), generator=torch.Generator().manual_seed(args.seed))
    adapter_shuffled = image_metrics(adapter_images[order], targets, train_mean)
    adapter_score = selection_score(adapter_validation, adapter_shuffled)
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
            "adapter_score": adapter_score,
            "adapter_v2": args.adapter_v2,
            "oracle_renderer": args.oracle_renderer,
            "latent_bank_manifest": manifest,
            "record": {"validation": initial, "adapter_validation": adapter_validation},
        }
        torch.save(payload, output / "best.pt")
        print(
            f"initial specialist={best:.3f} adapter_v2={adapter_score:.3f} "
            f"rgb={initial['rgb_mae']:.3f}",
            flush=True,
        )
    for epoch in range(start, args.epochs):
        train = train_epoch(
            model, ema, baseline, oracle, loaders["train"], optimizer, scaler, device, args
        )
        validation, adapter_validation, shuffled = evaluate(
            ema, baseline, loaders["val"], device, train_mean, args.seed + epoch + 1
        )
        hard, _ = evaluate(ema, baseline, loaders["val"], device, train_mean, hard=True)
        current = specialist_score(validation, shuffled, adapter_validation)
        improved = current > best + 1e-5
        strict_improvement = strictly_beats_adapter(
            validation, adapter_validation, current, adapter_score
        )
        stale = 0 if improved else stale + 1
        scheduler.step(current)
        record = {
            "epoch": epoch,
            "score": current,
            "beats_adapter_v2": current > adapter_score,
            "train": train,
            "validation": validation,
            "validation_hard": hard,
            "adapter_validation": adapter_validation,
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
            "adapter_score": adapter_score,
            "adapter_v2": args.adapter_v2,
            "oracle_renderer": args.oracle_renderer,
            "oracle_epoch": oracle_payload.get("epoch"),
            "latent_bank_manifest": manifest,
        }
        torch.save(payload, output / "last.pt")
        if improved:
            best = current
            torch.save(payload, output / "best.pt")
        if strict_improvement:
            previous = -math.inf
            strict_path = output / "best_beating_adapter.pt"
            if strict_path.exists():
                previous = torch.load(strict_path, map_location="cpu", weights_only=False).get(
                    "best_score", previous
                )
            if current > previous:
                torch.save(payload, strict_path)
        save_panel(
            ema,
            baseline,
            oracle,
            datasets["val"],
            device,
            output / "panels" / f"epoch_{epoch:03d}.png",
            args.panel_images,
        )
        print(
            f"epoch={epoch:03d} loss={train['loss']:.3f} score={current:.3f} "
            f"adapter={adapter_score:.3f} rgb={validation['rgb_mae']:.3f} "
            f"chroma={validation['chroma_mae']:.3f} palette={validation['palette_pearson']:.3f} "
            f"alpha={train['alpha']:.3f}",
            flush=True,
        )
        if stale >= args.patience:
            break
    selected = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    ema.load_state_dict(selected["ema"])
    ema.to(device)
    test, adapter_test, shuffled = evaluate(
        ema, baseline, loaders["test"], device, train_mean, args.seed + 999
    )
    report = {
        "checkpoint_epoch": selected["epoch"],
        "color_specialist": test,
        "adapter_v2": adapter_test,
        "shuffled": shuffled,
        "validation_adapter_score": adapter_score,
        "validation_selected_score": selected["best_score"],
        "test_images": len(datasets["test"]),
        "untouched_full_test": not args.smoke,
        "inference_inputs": ["preprocessed_eeg"],
    }
    (output / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
