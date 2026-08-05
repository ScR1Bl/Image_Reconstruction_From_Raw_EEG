"""Train an independent raw-EEG specialist for multi-scale image structure."""

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

from .build_pyramid_edge_targets import PYRAMID_KEYS
from .edge_specialist import EEGEdgeSpecialist
from .official_200way_benchmark import cached_array
from .pyramid_edge import (
    PyramidEdgeOutput,
    load_pyramid_edge_decoder,
    pyramid_edge_loss,
    pyramid_edge_sample_errors,
)
from .semantic_edge import EEGDINOGridPredictor
from .semantic_encoder import load_semantic_encoder
from .task_adapter import cvar, improvement_loss, shuffled_ranking_loss
from .train_color_oracle import update_ema
from .train_edge_baseline import EEGPyramidDataset, concatenate, split_records
from .train_edge_decoder import metrics as pyramid_metrics


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent structure-specialized raw EEG encoder"
    )
    parser.add_argument("--baseline", default="runs/eeg_edge_encoder/components/baseline.pt")
    parser.add_argument(
        "--semantic-encoder", default="runs/eeg_edge_encoder/components/semantic_encoder.pt"
    )
    parser.add_argument(
        "--pyramid-decoder", default="runs/eeg_edge_encoder/components/edge_decoder.pt"
    )
    parser.add_argument("--visual-bank", default="data/derived/visual_targets_dinov2s_192.pt")
    parser.add_argument("--target-bank", default="data/derived/pyramid_edge_targets_v1.pt")
    parser.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    parser.add_argument("--archives", default="data/things_eeg2_osf/preprocessed")
    parser.add_argument("--cache", default="data/derived/eeg_float32_cache")
    parser.add_argument("--output", default="runs/eeg_edge_encoder")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.015)
    parser.add_argument("--margin", type=float, default=0.018)
    parser.add_argument("--rank-margin", type=float, default=0.10)
    parser.add_argument("--ema-decay", type=float, default=0.998)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--panel-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


class FrozenEdgeBaseline:
    def __init__(
        self, checkpoint: Path, semantic_path: str, decoder_path: str, device: torch.device
    ) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        encoder = load_semantic_encoder(semantic_path, device)
        self.predictor = EEGDINOGridPredictor(encoder, **payload["model_config"]).to(device)
        self.predictor.load_state_dict(payload["model"], strict=True)
        self.decoder, self.decoder_payload = load_pyramid_edge_decoder(decoder_path, device)
        self.payload = payload
        for model in (self.predictor, self.decoder):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, eeg: torch.Tensor) -> PyramidEdgeOutput:
        grid, _ = self.predictor.forward_ensemble(eeg)
        return self.decoder(grid)


def shuffled_target(target, order):
    return {key: value[order] for key, value in target.items()}


def topology_loss(output: PyramidEdgeOutput, target) -> torch.Tensor:
    prediction = output.crisp_logits64.sigmoid()
    truth = target["crisp64"].float()
    pred_boundary = F.max_pool2d(prediction, 3, 1, 1) - (-F.max_pool2d(-prediction, 3, 1, 1))
    true_boundary = F.max_pool2d(truth, 3, 1, 1) - (-F.max_pool2d(-truth, 3, 1, 1))
    return 1 - (2 * (pred_boundary * true_boundary).sum() + 1) / (
        pred_boundary.sum() + true_boundary.sum() + 1
    )


def diversity_loss(output: PyramidEdgeOutput, target) -> torch.Tensor:
    losses = []
    for prediction, truth in (
        (output.semantic_logits32.sigmoid(), target["semantic32"]),
        (output.crisp_logits64.sigmoid(), target["crisp64"]),
    ):
        pred_std = prediction.flatten(1).std(0, unbiased=False).mean()
        true_std = truth.float().flatten(1).std(0, unbiased=False).mean()
        losses.append(F.relu(0.70 * true_std - pred_std))
    return sum(losses)


def residual_energy(output: PyramidEdgeOutput, baseline: PyramidEdgeOutput) -> torch.Tensor:
    return sum(
        (getattr(output, field) - getattr(baseline, field)).square().mean()
        for field in (
            "shape_logits16",
            "semantic_logits32",
            "perceptual_logits64",
            "crisp_logits64",
        )
    )


def train_epoch(model, ema, baseline, loader, optimizer, scaler, device, args):
    model.train()
    totals = {}
    batches = 0
    for batch in loader:
        eeg = batch["eeg"].to(device, non_blocking=True)
        target = {key: batch[key].to(device, non_blocking=True) for key in PYRAMID_KEYS}
        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            base = baseline(eeg)
            base_error, _ = pyramid_edge_sample_errors(base, target)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            result = model(eeg, base)
            output = result.pyramid
            pyramid, parts = pyramid_edge_loss(output, target)
            errors, sample_parts = pyramid_edge_sample_errors(output, target)
            task, hard = errors.mean(), cvar(errors, 0.25)
            improve = improvement_loss(errors, base_error, args.margin)
            order = torch.roll(torch.arange(len(eeg), device=device), 1)
            wrong_errors, _ = pyramid_edge_sample_errors(output, shuffled_target(target, order))
            ranking = (
                shuffled_ranking_loss(errors, wrong_errors, args.rank_margin)
                if len(eeg) > 1
                else task * 0
            )
            topology = topology_loss(output, target)
            diversity = diversity_loss(output, target)
            residual = residual_energy(output, base)
            weights = result.condition.subject_weights.clamp_min(1e-8)
            effective_subjects = (-weights * weights.log()).sum(1).exp()
            subject_collapse = F.relu(3 - effective_subjects).mean() / 3
            loss = (
                pyramid
                + 0.55 * task
                + 0.70 * hard
                + 1.7 * improve
                + 0.55 * ranking
                + 0.45 * topology
                + 0.60 * diversity
                + 0.002 * residual
                + 0.05 * subject_collapse
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
            "pyramid": pyramid,
            "task": task,
            "hard": hard,
            "improvement": improve,
            "ranking": ranking,
            "topology": topology,
            "diversity": diversity,
            "residual_energy": residual,
            "effective_subjects": effective_subjects.mean(),
            **parts,
            **{f"sample_{key}": value.mean() for key, value in sample_parts.items()},
            **{f"gate_{key}": value for key, value in result.gates.items()},
        }
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
    return {key: value / max(1, batches) for key, value in totals.items()}


@torch.no_grad()
def predict(model, baseline, loader, device):
    model.eval()
    outputs = []
    baselines = []
    targets = {key: [] for key in PYRAMID_KEYS}
    names = []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            base = baseline(eeg)
            output = model(eeg, base).pyramid
        outputs.append(
            PyramidEdgeOutput(
                **{
                    field: getattr(output, field).float().cpu()
                    for field in PyramidEdgeOutput.__dataclass_fields__
                }
            )
        )
        baselines.append(
            PyramidEdgeOutput(
                **{
                    field: getattr(base, field).float().cpu()
                    for field in PyramidEdgeOutput.__dataclass_fields__
                }
            )
        )
        for key in PYRAMID_KEYS:
            targets[key].append(batch[key].float())
        names.extend(batch["filename"])
    return (
        concatenate(outputs),
        concatenate(baselines),
        {key: torch.cat(value) for key, value in targets.items()},
        names,
    )


def shuffle_output(output, seed):
    order = torch.randperm(
        len(output.shape_logits16), generator=torch.Generator().manual_seed(seed)
    )
    return PyramidEdgeOutput(
        **{field: getattr(output, field)[order] for field in PyramidEdgeOutput.__dataclass_fields__}
    )


def evaluate(model, baseline, loader, device, seed=None):
    output, base, target, _ = predict(model, baseline, loader, device)
    values, base_values = pyramid_metrics(output, target), pyramid_metrics(base, target)
    if seed is None:
        return values, base_values
    return (
        values,
        base_values,
        pyramid_metrics(shuffle_output(output, seed), target),
        pyramid_metrics(shuffle_output(base, seed), target),
    )


def score(value, wrong, baseline_value):
    keys = (
        "semantic_tolerant_f1",
        "crisp_tolerant_f1",
        "perceptual_pearson",
        "distance_correlation",
    )
    specificity = sum(value[key] - wrong[key] for key in keys)
    base = (
        value["shape_iou"]
        + value["semantic_tolerant_f1"]
        + 1.2 * value["crisp_tolerant_f1"]
        + value["perceptual_pearson"]
        + value["distance_correlation"]
        + 0.9 * specificity
        + 0.25 * value["orientation_cosine"]
    )
    improvement = (
        1.0 * (value["semantic_tolerant_f1"] - baseline_value["semantic_tolerant_f1"])
        + 1.2 * (value["crisp_tolerant_f1"] - baseline_value["crisp_tolerant_f1"])
        + 0.6 * (value["distance_correlation"] - baseline_value["distance_correlation"])
    )
    return base + improvement


def strictly_better(value, baseline_value, score_value, baseline_score):
    return (
        score_value > baseline_score
        and value["semantic_tolerant_f1"] > baseline_value["semantic_tolerant_f1"]
        and value["crisp_tolerant_f1"] > baseline_value["crisp_tolerant_f1"]
        and value["distance_correlation"] > baseline_value["distance_correlation"]
        and value["shape_iou"] >= baseline_value["shape_iou"] - 0.002
    )


def gray(value, tile):
    array = (value.squeeze().float().clamp(0, 1) * 255).byte().numpy()
    return Image.fromarray(array, "L").resize((tile, tile), Image.Resampling.NEAREST).convert("RGB")


@torch.no_grad()
def save_panel(model, baseline, dataset, device, path, count):
    samples = [dataset[index] for index in range(min(count, len(dataset)))]
    if not samples:
        return
    eeg = torch.stack([sample["eeg"] for sample in samples]).to(device)
    base = baseline(eeg)
    model.eval()
    output = model(eeg, base).pyramid
    headers = (
        "semantic GT",
        "semantic baseline",
        "semantic specialist",
        "semantic change x4",
        "crisp GT",
        "crisp baseline",
        "crisp specialist",
        "crisp change x4",
    )
    tile, label = 112, 22
    canvas = Image.new("RGB", (len(headers) * tile, (len(samples) + 1) * (tile + label)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, header in enumerate(headers):
        draw.text((column * tile + 3, 3), header, fill="black")
    for row, sample in enumerate(samples):
        y = (row + 1) * (tile + label)
        draw.text((3, y - label + 3), sample["filename"], fill="black")
        semantic_base, semantic = (
            base.semantic_logits32.sigmoid()[row].cpu(),
            output.semantic_logits32.sigmoid()[row].cpu(),
        )
        crisp_base, crisp = (
            base.crisp_logits64.sigmoid()[row].cpu(),
            output.crisp_logits64.sigmoid()[row].cpu(),
        )
        values = (
            sample["semantic32"],
            semantic_base,
            semantic,
            (semantic - semantic_base).abs() * 4,
            sample["crisp64"],
            crisp_base,
            crisp,
            (crisp - crisp_base).abs() * 4,
        )
        for column, value in enumerate(values):
            canvas.paste(gray(value, tile), (column * tile, y))
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
    visual = torch.load(args.visual_bank, map_location="cpu", weights_only=False)
    target_bank = torch.load(args.target_bank, map_location="cpu", weights_only=False)
    visual_grid = F.normalize(visual["dino_grid"].float(), dim=-1)
    arrays = [
        cached_array(Path(args.archives), Path(args.cache), subject, "training")
        for subject in range(10)
    ]
    records = {
        split: split_records(args.index, split, visual, target_bank)
        for split in ("train", "val", "test")
    }
    if args.smoke:
        records = {key: value[:16] for key, value in records.items()}
    datasets = {
        key: EEGPyramidDataset(arrays, value, visual_grid, target_bank)
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
    baseline = FrozenEdgeBaseline(
        Path(args.baseline), args.semantic_encoder, args.pyramid_decoder, device
    )
    model = EEGEdgeSpecialist().to(device)
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
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
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
    history = output_dir / "history.jsonl"
    if not args.resume:
        history.unlink(missing_ok=True)
    initial, baseline_validation, shuffled, baseline_shuffled = evaluate(
        ema, baseline, loaders["val"], device, args.seed
    )
    baseline_score = score(baseline_validation, baseline_shuffled, baseline_validation)
    initial_score = score(initial, shuffled, baseline_validation)
    if start == 0:
        best = initial_score
        initial_payload = {
            "epoch": -1,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "model_config": model.config(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": best,
            "stale": 0,
            "baseline_score": baseline_score,
            "baseline": args.baseline,
            "pyramid_decoder": args.pyramid_decoder,
            "record": {"validation": initial, "baseline_validation": baseline_validation},
        }
        torch.save(initial_payload, output_dir / "best.pt")
        print(
            f"initial={initial_score:.3f} baseline={baseline_score:.3f} "
            f"semantic={initial['semantic_tolerant_f1']:.3f} crisp={initial['crisp_tolerant_f1']:.3f}",
            flush=True,
        )
    for epoch in range(start, args.epochs):
        train = train_epoch(model, ema, baseline, loaders["train"], optimizer, scaler, device, args)
        # Fixed permutation makes checkpoint comparisons deterministic.
        validation, baseline_validation, shuffled, _ = evaluate(
            ema, baseline, loaders["val"], device, args.seed
        )
        current = score(validation, shuffled, baseline_validation)
        improved = current > best + 1e-5
        strict = strictly_better(validation, baseline_validation, current, baseline_score)
        stale = 0 if improved else stale + 1
        scheduler.step(current)
        record = {
            "epoch": epoch,
            "score": current,
            "strictly_beats_baseline": strict,
            "train": train,
            "validation": validation,
            "baseline_validation": baseline_validation,
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
            "baseline_score": baseline_score,
            "baseline": args.baseline,
            "pyramid_decoder": args.pyramid_decoder,
        }
        torch.save(payload, output_dir / "last.pt")
        if improved:
            best = current
            torch.save(payload, output_dir / "best.pt")
        if strict:
            path = output_dir / "best_beating_baseline.pt"
            previous = -math.inf
            if path.exists():
                previous = torch.load(path, map_location="cpu", weights_only=False).get(
                    "best_score", previous
                )
            if current > previous:
                torch.save(payload, path)
        save_panel(
            ema,
            baseline,
            datasets["val"],
            device,
            output_dir / "panels" / f"epoch_{epoch:03d}.png",
            args.panel_images,
        )
        print(
            f"epoch={epoch:03d} loss={train['loss']:.3f} score={current:.3f} "
            f"semantic={validation['semantic_tolerant_f1']:.3f} "
            f"crisp={validation['crisp_tolerant_f1']:.3f} distance={validation['distance_correlation']:.3f}",
            flush=True,
        )
        if stale >= args.patience:
            break
    selected = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    ema.load_state_dict(selected["ema"])
    ema.to(device)
    test, baseline_test, shuffled, _ = evaluate(
        ema, baseline, loaders["test"], device, args.seed + 999
    )
    report = {
        "checkpoint_epoch": selected["epoch"],
        "edge_specialist": test,
        "baseline": baseline_test,
        "shuffled": shuffled,
        "validation_baseline_score": baseline_score,
        "validation_selected_score": selected["best_score"],
        "test_images": len(datasets["test"]),
        "untouched_full_test": not args.smoke,
        "inference_inputs": ["preprocessed_eeg"],
    }
    (output_dir / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
