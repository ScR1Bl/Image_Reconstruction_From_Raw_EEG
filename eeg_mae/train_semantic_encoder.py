from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .official_200way_benchmark import (
    REPEAT_COUNTS,
    cached_array,
    mean_metric_dict,
    retrieval_metrics,
    shuffled_metrics,
    training_target_indices,
)
from .semantic_encoder import SubjectAdaptiveSemanticEncoder, load_semantic_encoder


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a strong subject-adaptive EEG semantic encoder"
    )
    parser.add_argument("--training-bank", default="data/derived/visual_targets_dinov2s_192.pt")
    parser.add_argument("--test-bank", default="data/derived/official_test_targets_dinov2s_192.pt")
    parser.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    parser.add_argument("--archives", default="data/things_eeg2_osf/preprocessed")
    parser.add_argument("--cache", default="data/derived/eeg_float32_cache")
    parser.add_argument("--baseline-dir", default="runs/eeg_semantic_encoder/baselines")
    parser.add_argument(
        "--baseline-report", default="runs/eeg_semantic_encoder/baselines/report.json"
    )
    parser.add_argument("--output", default="runs/eeg_semantic_encoder")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--base-lr", type=float, default=3e-5)
    parser.add_argument("--residual-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--base-freeze-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--repeat-draws", type=int, default=8)
    parser.add_argument("--shuffle-permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


class SubjectConditionDataset(Dataset):
    def __init__(
        self,
        eeg: np.ndarray,
        visual: torch.Tensor,
        target_indices: np.ndarray,
        conditions: np.ndarray,
        subject: int,
    ) -> None:
        self.eeg = eeg
        self.visual = visual
        self.target_indices = target_indices
        self.conditions = conditions
        self.subject = subject

    def __len__(self) -> int:
        return len(self.conditions)

    def __getitem__(self, index: int):
        condition = int(self.conditions[index])
        eeg = np.asarray(self.eeg[condition], dtype=np.float32).mean(0)
        target_index = int(self.target_indices[condition])
        return (
            torch.from_numpy(eeg.copy()),
            torch.tensor(self.subject),
            self.visual[target_index].float(),
            torch.tensor(target_index),
        )


def semantic_loss(
    prediction: torch.Tensor, target: torch.Tensor, temperature: float = 0.07
) -> tuple[torch.Tensor, dict]:
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = prediction @ target.T / temperature
    labels = torch.arange(len(prediction), device=prediction.device)
    retrieval = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    cosine = (1.0 - (prediction * target).sum(-1)).mean()
    loss = retrieval + 0.25 * cosine
    return loss, {"retrieval": float(retrieval.detach()), "cosine": float(cosine.detach())}


def make_loaders(
    arrays: list[np.ndarray],
    visual: torch.Tensor,
    target_indices: np.ndarray,
    conditions: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> list[DataLoader]:
    loaders = []
    for subject, eeg in enumerate(arrays):
        dataset = SubjectConditionDataset(eeg, visual, target_indices, conditions, subject)
        generator = torch.Generator().manual_seed(seed + subject * 1009)
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                generator=generator if shuffle else None,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
                drop_last=shuffle,
            )
        )
    return loaders


@torch.no_grad()
def validation_metrics(model, loaders, device) -> dict:
    model.eval()
    predictions, shared_targets = [], None
    per_subject = {}
    for subject, loader in enumerate(loaders):
        values, targets = [], []
        for eeg, sid, target, _ in loader:
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                values.append(
                    model(eeg.to(device, non_blocking=True), sid.to(device, non_blocking=True))
                    .float()
                    .cpu()
                )
            targets.append(target)
        prediction, target = torch.cat(values), torch.cat(targets)
        per_subject[str(subject)] = retrieval_metrics(prediction, target)[0]
        predictions.append(F.normalize(prediction, dim=-1))
        if shared_targets is None:
            shared_targets = target
    ensemble = torch.stack(predictions).mean(0)
    return {"per_subject": per_subject, "ensemble": retrieval_metrics(ensemble, shared_targets)[0]}


@torch.no_grad()
def predict_official(
    model,
    eeg: np.ndarray,
    subject: int,
    repeat_count: int,
    draw: int,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    rng = np.random.default_rng(seed + subject * 100003 + repeat_count * 1009 + draw * 9176)
    indices = (
        np.broadcast_to(np.arange(80), (200, 80))
        if repeat_count == 80
        else np.stack([rng.choice(80, repeat_count, replace=False) for _ in range(200)])
    )
    outputs = []
    model.eval()
    for start in range(0, 200, batch_size):
        stop = min(start + batch_size, 200)
        batch = np.stack(
            [
                np.asarray(eeg[row, indices[row]], dtype=np.float32).mean(0)
                for row in range(start, stop)
            ]
        )
        tensor = torch.from_numpy(batch).to(device)
        sid = torch.full((len(tensor),), subject, dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            outputs.append(model(tensor, sid).float().cpu())
    return torch.cat(outputs)


def official_report(model, arrays, visual_test, args, device) -> dict:
    saved = {subject: {} for subject in range(10)}
    for subject, eeg in enumerate(arrays):
        print(f"official evaluation subject {subject + 1:02d}/10", flush=True)
        for count in REPEAT_COUNTS:
            draws = 1 if count == 80 else args.repeat_draws
            saved[subject][count] = [
                predict_official(
                    model, eeg, subject, count, draw, args.seed, device, args.eval_batch_size
                )
                for draw in range(draws)
            ]
    aggregate, per_subject = {}, {str(subject): {} for subject in range(10)}
    for count in REPEAT_COUNTS:
        records = []
        draws = 1 if count == 80 else args.repeat_draws
        for draw in range(draws):
            predictions = [saved[subject][count][draw] for subject in range(10)]
            for subject, prediction in enumerate(predictions):
                per_subject[str(subject)].setdefault(str(count), []).append(
                    retrieval_metrics(prediction, visual_test)[0]
                )
            # Keep aggregation identical to the decisive baseline protocol.
            ensemble = torch.stack(predictions).mean(0)
            records.append(retrieval_metrics(ensemble, visual_test)[0])
        aggregate[str(count)] = mean_metric_dict(records)
        if count == 80:
            _, scores = retrieval_metrics(ensemble, visual_test)
            aggregate[str(count)]["shuffled"] = shuffled_metrics(
                scores, args.shuffle_permutations, args.seed + 99
            )
    for subject, subject_records in per_subject.items():
        per_subject[subject] = {
            count: mean_metric_dict(records) for count, records in subject_records.items()
        }
    return {"aggregate": aggregate, "per_subject": per_subject}


def checkpoint(model, epoch: int, validation: dict, args) -> dict:
    return {
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "model_config": model.config(),
        "epoch": epoch,
        "validation": validation,
        "training_config": vars(args),
    }


def main() -> None:
    args = arguments()
    if args.smoke:
        args.epochs = 1
        args.patience = 1
        args.repeat_draws = 1
        args.shuffle_permutations = 10
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    train_bank = torch.load(args.training_bank, map_location="cpu", weights_only=False)
    test_bank = torch.load(args.test_bank, map_location="cpu", weights_only=False)
    visual_train = F.normalize(train_bank["dino_global"].float(), dim=-1)
    visual_test = F.normalize(test_bank["dino_global"].float(), dim=-1)
    target_indices, train_indices, validation_indices = training_target_indices(
        Path(args.index), train_bank
    )
    if args.smoke:
        train_indices = train_indices[:1024]
        validation_indices = validation_indices[:512]
    train_arrays = [
        cached_array(Path(args.archives), Path(args.cache), subject, "training")
        for subject in range(10)
    ]

    if args.eval_only:
        if not args.resume:
            raise ValueError("--eval-only requires --resume CHECKPOINT")
        model = load_semantic_encoder(args.resume, device)
    else:
        model = SubjectAdaptiveSemanticEncoder(output_dim=visual_train.shape[1]).to(device)
        start_epoch = 0
        if args.resume:
            payload = torch.load(args.resume, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model"], strict=True)
            start_epoch = int(payload.get("epoch", -1)) + 1
        else:
            model.load_linear_baselines(args.baseline_dir)
            model.to(device)

        train_loaders = make_loaders(
            train_arrays,
            visual_train,
            target_indices,
            train_indices,
            args.batch_size,
            True,
            args.seed,
        )
        val_loaders = make_loaders(
            train_arrays,
            visual_train,
            target_indices,
            validation_indices,
            args.eval_batch_size,
            False,
            args.seed,
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": model.base_parameters(), "lr": args.base_lr, "name": "base"},
                {"params": model.residual_parameters(), "lr": args.residual_lr, "name": "residual"},
            ],
            weight_decay=args.weight_decay,
        )

        initial = validation_metrics(model, val_loaders, device)
        best_score = initial["ensemble"]["mrr"]
        best = checkpoint(model, -1, initial, args)
        torch.save(best, output / "initialized.pt")
        print(
            f"initial val top1={initial['ensemble']['top1']:.4f} "
            f"mrr={initial['ensemble']['mrr']:.4f}",
            flush=True,
        )
        stale = 0
        history = output / "history.jsonl"
        if start_epoch == 0 and history.exists():
            history.unlink()
        for epoch in range(start_epoch, args.epochs):
            model.train()
            totals = {"loss": 0.0, "retrieval": 0.0, "cosine": 0.0}
            batches = 0
            base_factor = 0.0 if epoch < args.base_freeze_epochs else 1.0
            optimizer.param_groups[0]["lr"] = args.base_lr * base_factor
            progress = epoch / max(args.epochs - 1, 1)
            decay = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
            optimizer.param_groups[1]["lr"] = args.residual_lr * decay
            order = torch.randperm(
                10, generator=torch.Generator().manual_seed(args.seed + epoch)
            ).tolist()
            for subject in order:
                for eeg, sid, target, _ in train_loaders[subject]:
                    eeg, sid, target = (
                        eeg.to(device, non_blocking=True),
                        sid.to(device, non_blocking=True),
                        target.to(device, non_blocking=True),
                    )
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                        loss, parts = semantic_loss(model(eeg, sid), target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    totals["loss"] += float(loss.detach())
                    totals["retrieval"] += parts["retrieval"]
                    totals["cosine"] += parts["cosine"]
                    batches += 1
            validation = validation_metrics(model, val_loaders, device)
            record = {
                "epoch": epoch,
                "train": {key: value / max(batches, 1) for key, value in totals.items()},
                "validation": validation,
                "base_lr": optimizer.param_groups[0]["lr"],
                "residual_lr": optimizer.param_groups[1]["lr"],
            }
            with history.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            metric = validation["ensemble"]
            print(
                f"epoch={epoch:03d} loss={record['train']['loss']:.4f} "
                f"val_top1={metric['top1']:.4f} val_mrr={metric['mrr']:.4f} "
                f"median={metric['median_rank']:.1f}",
                flush=True,
            )
            if metric["mrr"] > best_score + 1e-6:
                best_score, stale = metric["mrr"], 0
                best = checkpoint(model, epoch, validation, args)
                torch.save(best, output / "best.pt")
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"early stopping after {stale} stale epochs", flush=True)
                    break
        if not (output / "best.pt").exists():
            torch.save(best, output / "best.pt")
        model = load_semantic_encoder(output / "best.pt", device)

    test_arrays = [
        cached_array(Path(args.archives), Path(args.cache), subject, "test")
        for subject in range(10)
    ]
    result = {
        "protocol": {
            "conditions": 200,
            "repetitions": 80,
            "subjects": 10,
            "selection": "internal validation only",
            "seed": args.seed,
        },
        "semantic_encoder": official_report(model, test_arrays, visual_test, args, device),
    }
    baseline_path = Path(args.baseline_report)
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        reference = baseline["methods"]["linear"]["aggregate"]["80"]
        achieved = result["semantic_encoder"]["aggregate"]["80"]
        result["reference_linear"] = reference
        result["comparison"] = {
            "top1_delta": achieved["top1"] - reference["top1"],
            "top5_delta": achieved["top5"] - reference["top5"],
            "decision": (
                "improved"
                if achieved["top1"] > reference["top1"] + 1e-6
                else "matched"
                if achieved["top1"] >= reference["top1"] - 0.005
                else "regressed"
            ),
        }
    (output / "official_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "80_repeats": result["semantic_encoder"]["aggregate"]["80"],
                "comparison": result.get("comparison"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
