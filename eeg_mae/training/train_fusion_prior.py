"""Train a rectified-flow prior over SDXL latents from frozen EEG conditions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ..data.fusion_conditions import FusionConditionBank
from ..data.hierarchical_targets import HierarchicalTargetBank
from ..evaluation.official_200way_benchmark import retrieval_metrics, shuffled_metrics
from ..models.fusion_prior import EEGFusionPrior, load_fusion_prior
from ..models.hierarchical_semantic import load_hierarchical_semantic_encoder


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train semantic + color + edge EEG fusion prior")
    parser.add_argument("--condition-bank", default="data/derived/eeg_fusion_conditions")
    parser.add_argument("--target-bank", default="data/derived/hierarchical_visual_targets")
    parser.add_argument("--semantic-checkpoint", default="runs/eeg_hierarchical_encoder/best.pt")
    parser.add_argument("--output", default="runs/eeg_fusion_prior")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--sample-steps", type=int, default=16)
    parser.add_argument("--guidance", type=float, default=1.6)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


class FusionPriorDataset(Dataset):
    def __init__(self, conditions: FusionConditionBank, targets: HierarchicalTargetBank, limit=0):
        if len(conditions) != len(targets):
            raise RuntimeError("condition and target bank lengths differ")
        if conditions.manifest.get("source_target_fingerprint") != targets.manifest.get(
            "source_fingerprint"
        ):
            raise RuntimeError("condition bank was built from a different visual target bank")
        self.conditions, self.targets = conditions, targets
        self.length = min(limit, len(targets)) if limit else len(targets)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        condition, target = self.conditions[item], self.targets[item]
        if condition["image_index"] != target["image_index"]:
            raise RuntimeError("condition/target index mismatch")
        return {
            "conditions": condition["conditions"],
            "targets": target["targets"],
            "filename": target["filename"],
        }


def make_condition(model, values, *, drop=False, ablate=None):
    branches = {
        "synclr": values["semantic_synclr"],
        "clip_b32": values["semantic_clip_b32"],
        "vae_sdxl": values["semantic_vae_sdxl"],
    }
    return model.encode_condition(
        values["semantic_fused"],
        branches,
        values["color_grid16"],
        values["edge_grid16"],
        drop_modalities=drop,
        ablate=ablate,
    )


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def symmetric_contrastive(prediction, target, temperature=0.08):
    prediction, target = (
        F.normalize(prediction.float(), dim=-1),
        F.normalize(target.float(), dim=-1),
    )
    logits = prediction @ target.T / temperature
    labels = torch.arange(len(prediction), device=prediction.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def gradient_loss(prediction, target):
    return 0.5 * (
        F.smooth_l1_loss(
            prediction[..., 1:, :] - prediction[..., :-1, :],
            target[..., 1:, :] - target[..., :-1, :],
        )
        + F.smooth_l1_loss(
            prediction[..., :, 1:] - prediction[..., :, :-1],
            target[..., :, 1:] - target[..., :, :-1],
        )
    )


def train_epoch(model, visual_fusion, loader, optimizer, scaler, device, ema, ema_decay):
    model.train()
    totals, batches = {}, 0
    for batch in loader:
        values = {k: v.to(device, non_blocking=True) for k, v in batch["conditions"].items()}
        targets = {k: v.float().to(device, non_blocking=True) for k, v in batch["targets"].items()}
        latent = targets["vae_sdxl"]
        noise = torch.randn_like(latent)
        # A mild logit-normal schedule spends more updates away from trivial endpoints.
        time = torch.sigmoid(torch.randn(len(latent), device=device))
        noisy = (1 - time[:, None, None, None]) * noise + time[:, None, None, None] * latent
        target_velocity = latent - noise
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            condition = make_condition(model, values, drop=True)
            unconditional = bool(torch.rand(()) < model.condition_dropout)
            velocity = model.predict_velocity(noisy, time, condition, unconditional=unconditional)
            estimate = noisy + (1 - time[:, None, None, None]) * velocity
            with torch.no_grad():
                target_fused, _ = visual_fusion(targets)
            flow = F.mse_loss(velocity.float(), target_velocity.float())
            latent_loss = F.smooth_l1_loss(estimate.float(), latent.float())
            gradients = gradient_loss(estimate.float(), latent.float())
            semantic = symmetric_contrastive(condition.semantic_prediction, target_fused)
            semantic_cosine = (
                1
                - F.cosine_similarity(
                    condition.semantic_prediction.float(), target_fused.float(), dim=-1
                ).mean()
            )
            loss = (
                flow
                + 0.35 * latent_loss
                + 0.12 * gradients
                + 0.20 * semantic
                + 0.08 * semantic_cosine
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            for target_parameter, parameter in zip(
                ema.parameters(), model.parameters(), strict=True
            ):
                target_parameter.lerp_(parameter.detach(), 1 - ema_decay)
            for target_buffer, buffer in zip(ema.buffers(), model.buffers(), strict=True):
                target_buffer.copy_(buffer)
        parts = {
            "loss": loss,
            "flow": flow,
            "latent": latent_loss,
            "gradient": gradients,
            "semantic": semantic,
        }
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
        batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(model, visual_fusion, loader, device, steps, guidance, seed, *, ablations=False):
    model.eval()
    predictions, targets_all, semantic_predictions, semantic_targets = [], [], [], []
    ablation_errors = {name: 0.0 for name in ("semantic", "color", "edge", "all")}
    samples = 0
    generator = torch.Generator(device=device).manual_seed(seed)
    for batch in loader:
        values = {k: v.to(device, non_blocking=True) for k, v in batch["conditions"].items()}
        targets = {k: v.float().to(device, non_blocking=True) for k, v in batch["targets"].items()}
        latent = targets["vae_sdxl"]
        condition = make_condition(model, values)
        noise = torch.randn(latent.shape, generator=generator, device=device, dtype=latent.dtype)
        prediction = model.sample(condition, steps=steps, guidance=guidance, noise=noise)
        target_fused, _ = visual_fusion(targets)
        predictions.append(prediction.float().cpu())
        targets_all.append(latent.float().cpu())
        semantic_predictions.append(condition.semantic_prediction.float().cpu())
        semantic_targets.append(target_fused.float().cpu())
        if ablations:
            time = latent.new_full((len(latent),), 0.5)
            noisy = 0.5 * (noise + latent)
            clean_velocity = model.predict_velocity(noisy, time, condition)
            clean_error = (
                F.mse_loss(clean_velocity, latent - noise, reduction="none").flatten(1).mean(1)
            )
            for name in ablation_errors:
                altered = make_condition(model, values, ablate=name)
                error = (
                    F.mse_loss(
                        model.predict_velocity(noisy, time, altered),
                        latent - noise,
                        reduction="none",
                    )
                    .flatten(1)
                    .mean(1)
                )
                ablation_errors[name] += float((error - clean_error).sum())
        samples += len(latent)
    prediction, target = torch.cat(predictions), torch.cat(targets_all)
    semantic_prediction, semantic_target = (
        torch.cat(semantic_predictions),
        torch.cat(semantic_targets),
    )
    latent_metrics, scores = retrieval_metrics(prediction.flatten(1), target.flatten(1))
    semantic_metrics, _ = retrieval_metrics(semantic_prediction, semantic_target)
    report = {
        "latent": latent_metrics,
        "latent_mae": float((prediction - target).abs().mean()),
        "latent_rmse": float((prediction - target).square().mean().sqrt()),
        "semantic": semantic_metrics,
        "selection_score": latent_metrics["mrr"]
        + 0.20 * semantic_metrics["mrr"]
        - 0.03 * float((prediction - target).abs().mean()),
    }
    if ablations:
        report["ablation_flow_mse_increase"] = {
            key: value / samples for key, value in ablation_errors.items()
        }
        report["shuffled"] = shuffled_metrics(scores, permutations=100, seed=seed + 71)
    return report


def checkpoint(
    model, ema, optimizer, scheduler, scaler, epoch, best_score, stale, args, validation
):
    return {
        "schema_version": 1,
        "architecture": "eeg_rectified_flow_fusion_prior_v1",
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "ema": {k: v.detach().cpu() for k, v in ema.state_dict().items()},
        "model_config": model.config(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
        "stale": stale,
        "validation": validation,
        "training_config": vars(args),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = arguments(argv)
    if args.smoke:
        args.epochs, args.batch_size, args.eval_batch_size = 1, 8, 8
        args.sample_steps, args.patience = 2, 1
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    limits = {"train": 64, "val": 32, "test": 32} if args.smoke else {}
    loaders = {}
    condition_manifest = None
    for split in ("train", "val", "test"):
        condition_bank = FusionConditionBank(args.condition_bank, split)
        condition_manifest = condition_bank.manifest
        dataset = FusionPriorDataset(
            condition_bank,
            HierarchicalTargetBank(args.target_bank, split),
            limits.get(split, 0),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=args.batch_size if split == "train" else args.eval_batch_size,
            shuffle=split == "train",
            num_workers=0,
            pin_memory=device.type == "cuda",
            drop_last=split == "train",
        )
    expected_semantic = condition_manifest["checkpoints"]["semantic"]["sha256"]
    if sha256(args.semantic_checkpoint) != expected_semantic:
        raise RuntimeError(
            "semantic checkpoint differs from the one used to build the condition bank"
        )
    _, visual_fusion, _ = load_hierarchical_semantic_encoder(args.semantic_checkpoint, device)
    visual_fusion.eval()
    for parameter in visual_fusion.parameters():
        parameter.requires_grad_(False)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    model = EEGFusionPrior().to(device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max(args.epochs, 1), eta_min=args.lr * 0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch, best_score, stale = 0, -math.inf, 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        ema.load_state_dict(state["ema"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        start_epoch, best_score, stale = (
            int(state["epoch"]) + 1,
            float(state["best_score"]),
            int(state.get("stale", 0)),
        )
    if args.eval_only:
        if not args.resume:
            raise ValueError("--eval-only requires --resume")
        loaded, _ = load_fusion_prior(args.resume, device)
        report = evaluate(
            loaded,
            visual_fusion,
            loaders["test"],
            device,
            args.sample_steps,
            args.guidance,
            args.seed + 9000,
            ablations=True,
        )
        report.update(
            {
                "test_images": len(loaders["test"].dataset),
                "untouched_full_test": len(loaders["test"].dataset) == 1654,
                "inference_inputs": ["preprocessed_eeg", "known_subject_id"],
            }
        )
        (output / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    history = output / "history.jsonl"
    if not args.resume:
        history.unlink(missing_ok=True)
    for epoch in range(start_epoch, args.epochs):
        train = train_epoch(
            model, visual_fusion, loaders["train"], optimizer, scaler, device, ema, args.ema_decay
        )
        validation = evaluate(
            ema,
            visual_fusion,
            loaders["val"],
            device,
            args.sample_steps,
            args.guidance,
            args.seed + 5000,
        )
        score = float(validation["selection_score"])
        improved = score > best_score
        if improved:
            best_score, stale = score, 0
        else:
            stale += 1
        scheduler.step()
        state = checkpoint(
            model, ema, optimizer, scheduler, scaler, epoch, best_score, stale, args, validation
        )
        torch.save(state, output / "last.pt")
        if improved:
            torch.save(state, output / "best.pt")
        record = {"epoch": epoch, "train": train, "validation": validation, "best": improved}
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if stale >= args.patience:
            break
    best, _ = load_fusion_prior(output / "best.pt", device)
    report = evaluate(
        best,
        visual_fusion,
        loaders["test"],
        device,
        args.sample_steps,
        args.guidance,
        args.seed + 9000,
        ablations=True,
    )
    report.update(
        {
            "test_images": len(loaders["test"].dataset),
            "untouched_full_test": len(loaders["test"].dataset) == 1654,
            "inference_inputs": ["preprocessed_eeg", "known_subject_id"],
        }
    )
    (output / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
