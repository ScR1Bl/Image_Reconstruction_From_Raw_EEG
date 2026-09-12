"""Train EEG tokens against the SynCLR + CLIP-B/32 + SDXL-VAE bank."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ..data.hierarchical_targets import HierarchicalTargetBank
from ..evaluation.official_200way_benchmark import cached_array, retrieval_metrics, shuffled_metrics
from ..models.hierarchical_semantic import (
    BRANCH_DIMS,
    HierarchicalEEGOutput,
    HierarchicalSemanticEEGEncoder,
    HierarchicalVisualFusion,
    load_hierarchical_semantic_encoder,
)
from ..models.model_utils import variance_covariance_regularization
from ..models.semantic_encoder import load_semantic_encoder


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train hierarchical SynCLR + CLIP + VAE EEG semantic tokens"
    )
    parser.add_argument("--target-bank", default="data/derived/hierarchical_visual_targets")
    parser.add_argument("--archives", default="data/things_eeg2_osf/preprocessed")
    parser.add_argument("--cache", default="data/derived/eeg_float32_cache")
    parser.add_argument(
        "--dino-anchor",
        default="runs/eeg_color_encoder/components/semantic_encoder.pt",
    )
    parser.add_argument("--output", default="runs/eeg_hierarchical_encoder")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr", type=float, default=1.5e-5)
    parser.add_argument("--fusion-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--backbone-freeze-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--modality-dropout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


class SubjectHierarchicalDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        eeg: np.ndarray,
        bank: HierarchicalTargetBank,
        subject: int,
        limit: int = 0,
    ) -> None:
        self.eeg = eeg
        self.bank = bank
        self.subject = subject
        self.length = min(limit, len(bank)) if limit else len(bank)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item: int) -> dict[str, object]:
        target = self.bank[item]
        condition = int(target["image_index"])
        eeg = np.asarray(self.eeg[condition], dtype=np.float32).mean(0)
        return {
            "eeg": torch.from_numpy(eeg.copy()),
            "subject": torch.tensor(self.subject, dtype=torch.long),
            "targets": target["targets"],
            "filename": target["filename"],
            "image_index": condition,
        }


def make_loaders(
    arrays: list[np.ndarray],
    bank: HierarchicalTargetBank,
    batch_size: int,
    shuffle: bool,
    seed: int,
    limit: int = 0,
) -> list[DataLoader]:
    result = []
    for subject, eeg in enumerate(arrays):
        generator = torch.Generator().manual_seed(seed + subject * 1009)
        result.append(
            DataLoader(
                SubjectHierarchicalDataset(eeg, bank, subject, limit),
                batch_size=batch_size,
                shuffle=shuffle,
                generator=generator if shuffle else None,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
                drop_last=shuffle,
            )
        )
    return result


def symmetric_contrastive(
    prediction: torch.Tensor, target: torch.Tensor, temperature: float
) -> torch.Tensor:
    prediction = F.normalize(prediction.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    logits = prediction @ target.T / temperature
    labels = torch.arange(len(prediction), device=prediction.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def hierarchical_loss(
    prediction: HierarchicalEEGOutput,
    target_fused: torch.Tensor,
    target_branches: dict[str, torch.Tensor],
    anchor_reference: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    fused = symmetric_contrastive(prediction.fused, target_fused, temperature)
    branch_losses = {
        key: symmetric_contrastive(prediction.branches[key], target_branches[key], temperature)
        for key in BRANCH_DIMS
    }
    branch = 0.35 * branch_losses["synclr"]
    branch = branch + 0.40 * branch_losses["clip_b32"]
    branch = branch + 0.25 * branch_losses["vae_sdxl"]
    fused_cosine = (
        1 - F.cosine_similarity(prediction.fused.float(), target_fused.float(), dim=-1).mean()
    )
    anchor = (
        1
        - F.cosine_similarity(
            prediction.dino_anchor.float(), anchor_reference.float(), dim=-1
        ).mean()
    )
    variance, covariance = variance_covariance_regularization(prediction.fused, target_std=0.35)
    loss = (
        fused
        + 0.65 * branch
        + 0.15 * fused_cosine
        + 0.12 * anchor
        + 0.08 * variance
        + 0.01 * covariance
    )
    return loss, {
        "fused_contrastive": fused,
        "branch_contrastive": branch,
        "synclr": branch_losses["synclr"],
        "clip_b32": branch_losses["clip_b32"],
        "vae_sdxl": branch_losses["vae_sdxl"],
        "fused_cosine": fused_cosine,
        "anchor_distillation": anchor,
        "variance": variance,
        "covariance": covariance,
    }


def _to_device_targets(batch: dict[str, torch.Tensor], device: torch.device) -> dict:
    return {key: value.float().to(device, non_blocking=True) for key, value in batch.items()}


def train_epoch(
    encoder,
    visual_fusion,
    anchor_teacher,
    loaders,
    optimizer,
    scaler,
    device,
    temperature,
) -> dict[str, float]:
    encoder.train()
    visual_fusion.train()
    anchor_teacher.eval()
    totals: dict[str, float] = {}
    batches = 0
    for subject in torch.randperm(len(loaders)).tolist():
        for batch in loaders[subject]:
            eeg = batch["eeg"].to(device, non_blocking=True)
            sid = batch["subject"].to(device, non_blocking=True)
            targets = _to_device_targets(batch["targets"], device)
            optimizer.zero_grad(set_to_none=True)
            with (
                torch.no_grad(),
                torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"),
            ):
                anchor_reference = anchor_teacher(eeg, sid)
            with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
                prediction = encoder(eeg, sid)
                target_fused, target_branches = visual_fusion(targets, apply_modality_dropout=True)
                loss, parts = hierarchical_loss(
                    prediction,
                    target_fused,
                    target_branches,
                    anchor_reference,
                    temperature,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(visual_fusion.parameters()), 4.0
            )
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach())
            batches += 1
    return {key: value / max(batches, 1) for key, value in totals.items()}


@torch.no_grad()
def predict_loader(encoder, visual_fusion, loader, device):
    encoder.eval()
    visual_fusion.eval()
    fused_predictions, fused_targets = [], []
    branch_predictions = {key: [] for key in BRANCH_DIMS}
    branch_targets = {key: [] for key in BRANCH_DIMS}
    for batch in loader:
        eeg = batch["eeg"].to(device, non_blocking=True)
        sid = batch["subject"].to(device, non_blocking=True)
        targets = _to_device_targets(batch["targets"], device)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            prediction = encoder(eeg, sid)
            target_fused, target_branch = visual_fusion(targets)
        fused_predictions.append(prediction.fused.float().cpu())
        fused_targets.append(target_fused.float().cpu())
        for key in BRANCH_DIMS:
            branch_predictions[key].append(prediction.branches[key].float().cpu())
            branch_targets[key].append(target_branch[key].float().cpu())
    return {
        "fused_prediction": torch.cat(fused_predictions),
        "fused_target": torch.cat(fused_targets),
        "branch_prediction": {key: torch.cat(value) for key, value in branch_predictions.items()},
        "branch_target": {key: torch.cat(value) for key, value in branch_targets.items()},
    }


@torch.no_grad()
def validation_metrics(encoder, visual_fusion, loaders, device) -> dict:
    predictions = []
    per_subject = {}
    common = None
    for subject, loader in enumerate(loaders):
        value = predict_loader(encoder, visual_fusion, loader, device)
        predictions.append(value)
        metrics = {"fused": retrieval_metrics(value["fused_prediction"], value["fused_target"])[0]}
        metrics["branches"] = {
            key: retrieval_metrics(value["branch_prediction"][key], value["branch_target"][key])[0]
            for key in BRANCH_DIMS
        }
        per_subject[str(subject)] = metrics
        common = value

    fused = torch.stack(
        [F.normalize(value["fused_prediction"], dim=-1) for value in predictions]
    ).mean(0)
    branch_ensemble = {
        key: torch.stack(
            [F.normalize(value["branch_prediction"][key], dim=-1) for value in predictions]
        ).mean(0)
        for key in BRANCH_DIMS
    }
    ensemble = {"fused": retrieval_metrics(fused, common["fused_target"])[0]}
    ensemble["branches"] = {
        key: retrieval_metrics(branch_ensemble[key], common["branch_target"][key])[0]
        for key in BRANCH_DIMS
    }
    ensemble["selection_score"] = ensemble["fused"]["mrr"] + 0.15 * sum(
        ensemble["branches"][key]["mrr"] for key in BRANCH_DIMS
    ) / len(BRANCH_DIMS)
    return {"ensemble": ensemble, "per_subject": per_subject}


@torch.no_grad()
def untouched_test_report(encoder, visual_fusion, loaders, device, seed: int) -> dict:
    clean = validation_metrics(encoder, visual_fusion, loaders, device)
    predictions = [predict_loader(encoder, visual_fusion, loader, device) for loader in loaders]
    fused = torch.stack(
        [F.normalize(value["fused_prediction"], dim=-1) for value in predictions]
    ).mean(0)
    target = predictions[0]["fused_target"]
    _, scores = retrieval_metrics(fused, target)

    zero_predictions = []
    for subject, loader in enumerate(loaders):
        outputs = []
        for batch in loader:
            eeg = torch.zeros_like(batch["eeg"]).to(device)
            sid = torch.full((len(eeg),), subject, dtype=torch.long, device=device)
            with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
                outputs.append(encoder(eeg, sid).fused.float().cpu())
        zero_predictions.append(F.normalize(torch.cat(outputs), dim=-1))
    zero = torch.stack(zero_predictions).mean(0)
    return {
        "clean": clean,
        "shuffled": shuffled_metrics(scores, permutations=100, seed=seed + 77),
        "zero_eeg": retrieval_metrics(zero, target)[0],
        "test_images": len(target),
        "untouched_full_test": len(target) == 1654,
        "inference_inputs": ["preprocessed_eeg", "known_subject_id"],
    }


def checkpoint(encoder, visual_fusion, optimizer, scaler, epoch, best_score, stale, args, val):
    return {
        "schema_version": 1,
        "architecture": "hierarchical_synclr_clip_b32_vae_v1",
        "encoder": {key: value.detach().cpu() for key, value in encoder.state_dict().items()},
        "visual_fusion": {
            key: value.detach().cpu() for key, value in visual_fusion.state_dict().items()
        },
        "encoder_config": encoder.config(),
        "fusion_config": visual_fusion.config(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_score": best_score,
        "stale": stale,
        "validation": val,
        "training_config": vars(args),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = arguments(argv)
    if args.smoke:
        args.epochs = 1
        args.batch_size = 32
        args.eval_batch_size = 64
        args.patience = 1
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    banks = {
        split: HierarchicalTargetBank(args.target_bank, split) for split in ("train", "val", "test")
    }
    arrays = [
        cached_array(Path(args.archives), Path(args.cache), subject, "training")
        for subject in range(10)
    ]
    limits = {"train": 512, "val": 256, "test": 256} if args.smoke else {}
    loaders = {
        split: make_loaders(
            arrays,
            bank,
            args.batch_size if split == "train" else args.eval_batch_size,
            split == "train",
            args.seed,
            limits.get(split, 0),
        )
        for split, bank in banks.items()
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    if args.eval_only:
        if not args.resume:
            raise ValueError("--eval-only requires --resume")
        encoder, visual_fusion, _ = load_hierarchical_semantic_encoder(args.resume, device)
        start_epoch = args.epochs
        optimizer = None
        scaler = None
        best_score = -math.inf
        stale = 0
    else:
        encoder = HierarchicalSemanticEEGEncoder().to(device)
        visual_fusion = HierarchicalVisualFusion(modality_dropout=args.modality_dropout).to(device)
        encoder.load_dino_anchor(args.dino_anchor)
        encoder.to(device)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": encoder.backbone.parameters(),
                    "lr": args.backbone_lr,
                    "name": "backbone",
                },
                {"params": encoder.head_parameters(), "lr": args.head_lr, "name": "heads"},
                {"params": visual_fusion.parameters(), "lr": args.fusion_lr, "name": "fusion"},
            ],
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        start_epoch, best_score, stale = 0, -math.inf, 0
        if args.resume:
            state = torch.load(args.resume, map_location="cpu", weights_only=False)
            encoder.load_state_dict(state["encoder"], strict=True)
            visual_fusion.load_state_dict(state["visual_fusion"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            if "scaler" in state:
                scaler.load_state_dict(state["scaler"])
            start_epoch = int(state["epoch"]) + 1
            best_score = float(state["best_score"])
            stale = int(state.get("stale", 0))

        # A resumed run must keep the original validated DINO checkpoint as
        # its teacher instead of silently redefining the anchor at resume time.
        anchor_teacher = load_semantic_encoder(args.dino_anchor, device).eval()
        for parameter in anchor_teacher.parameters():
            parameter.requires_grad_(False)
        history = output / "history.jsonl"
        if not args.resume:
            history.unlink(missing_ok=True)
        initial = validation_metrics(encoder, visual_fusion, loaders["val"], device)
        if start_epoch == 0:
            best_score = initial["ensemble"]["selection_score"]
            initialized = checkpoint(
                encoder, visual_fusion, optimizer, scaler, -1, best_score, 0, args, initial
            )
            torch.save(initialized, output / "initialized.pt")
            torch.save(initialized, output / "best.pt")
        print(
            f"initial val fused_top1={initial['ensemble']['fused']['top1']:.4f} "
            f"score={initial['ensemble']['selection_score']:.4f}",
            flush=True,
        )

        for epoch in range(start_epoch, args.epochs):
            progress = epoch / max(args.epochs - 1, 1)
            decay = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
            optimizer.param_groups[0]["lr"] = (
                0.0 if epoch < args.backbone_freeze_epochs else args.backbone_lr * decay
            )
            optimizer.param_groups[1]["lr"] = args.head_lr * decay
            optimizer.param_groups[2]["lr"] = args.fusion_lr * decay
            train = train_epoch(
                encoder,
                visual_fusion,
                anchor_teacher,
                loaders["train"],
                optimizer,
                scaler,
                device,
                args.temperature,
            )
            val = validation_metrics(encoder, visual_fusion, loaders["val"], device)
            score = val["ensemble"]["selection_score"]
            improved = score > best_score + 1e-6
            if improved:
                best_score, stale = score, 0
            else:
                stale += 1
            state = checkpoint(
                encoder, visual_fusion, optimizer, scaler, epoch, best_score, stale, args, val
            )
            torch.save(state, output / "last.pt")
            if improved:
                torch.save(state, output / "best.pt")
            record = {"epoch": epoch, "train": train, "validation": val, "improved": improved}
            with history.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"epoch={epoch:03d} loss={train['loss']:.4f} "
                f"fused_top1={val['ensemble']['fused']['top1']:.4f} score={score:.4f}",
                flush=True,
            )
            if stale >= args.patience:
                print(f"early stopping after {stale} stale epochs", flush=True)
                break
        encoder, visual_fusion, _ = load_hierarchical_semantic_encoder(output / "best.pt", device)

    report = untouched_test_report(encoder, visual_fusion, loaders["test"], device, args.seed)
    (output / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
