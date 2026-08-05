from __future__ import annotations

import argparse
import csv
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

REPEAT_COUNTS = (1, 2, 4, 8, 16, 32, 40, 64, 80)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decisive official THINGS-EEG2 200-way EEG decoding benchmark"
    )
    parser.add_argument("--training-bank", default="data/derived/visual_targets_dinov2s_192.pt")
    parser.add_argument("--test-bank", default="data/derived/official_test_targets_dinov2s_192.pt")
    parser.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    parser.add_argument("--archives", default="data/things_eeg2_osf/preprocessed")
    parser.add_argument("--cache", default="data/derived/eeg_float32_cache")
    parser.add_argument("--output", default="runs/official_200way_benchmark_v1")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("linear", "linear_full", "tsconv"),
        default=("linear", "tsconv"),
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=list(range(10)),
        help="zero-based subject IDs; default is all 10",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--repeat-draws", type=int, default=8)
    parser.add_argument("--shuffle-permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume-baselines", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def subject_paths(archives: Path, subject: int) -> tuple[Path, str, str]:
    label = f"sub-{subject + 1:02d}"
    return (
        archives / f"{label}.zip",
        f"{label}/preprocessed_eeg_training.npy",
        f"{label}/preprocessed_eeg_test.npy",
    )


def load_packed_array(archive_path: Path, member: str) -> np.ndarray:
    with zipfile.ZipFile(archive_path) as archive:
        payload = archive.read(member)
    packed = np.load(io.BytesIO(payload), allow_pickle=True).item()
    return packed["preprocessed_eeg_data"]


def cached_array(archives: Path, cache: Path, subject: int, split: str) -> np.ndarray:
    archive_path, train_member, test_member = subject_paths(archives, subject)
    member = train_member if split == "training" else test_member
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"sub-{subject + 1:02d}__preprocessed_eeg_{split}.npy"
    if not path.exists():
        print(f"extracting float32 cache {path.name}", flush=True)
        value = np.asarray(load_packed_array(archive_path, member), dtype=np.float32)
        np.save(path, value, allow_pickle=False)
        del value
    value = np.load(path, mmap_mode="r")
    expected = (16540, 4, 17, 100) if split == "training" else (200, 80, 17, 100)
    if value.shape != expected or value.dtype != np.float32:
        raise RuntimeError(
            f"invalid {path}: expected {expected} float32, got {value.shape} {value.dtype}"
        )
    return value


def training_target_indices(
    index_path: Path, bank: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lookup = {name: index for index, name in enumerate(bank["filenames"])}
    image_to_name: dict[int, str] = {}
    image_to_split: dict[int, str] = {}
    with index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["subject_id"]) != 0:
                continue
            image_index = int(row["image_index"])
            name = row["image_file"]
            split = row["split"]
            if image_index in image_to_name and image_to_name[image_index] != name:
                raise RuntimeError(f"inconsistent image mapping at {image_index}")
            image_to_name[image_index], image_to_split[image_index] = name, split
    if sorted(image_to_name) != list(range(16540)):
        raise RuntimeError("index does not cover official training conditions 0..16539")
    targets = np.asarray([lookup[image_to_name[index]] for index in range(16540)], dtype=np.int64)
    train = np.asarray(
        [index for index in range(16540) if image_to_split[index] == "train"], dtype=np.int64
    )
    validation = np.asarray(
        [index for index in range(16540) if image_to_split[index] == "val"], dtype=np.int64
    )
    if len(train) != 13232 or len(validation) != 1654:
        raise RuntimeError(
            f"unexpected internal split sizes: train={len(train)}, val={len(validation)}"
        )
    return targets, train, validation


class AveragedTrainingDataset(Dataset):
    def __init__(
        self,
        eeg: np.ndarray,
        visual: torch.Tensor,
        target_indices: np.ndarray,
        condition_indices: np.ndarray,
        post_stimulus: bool,
    ) -> None:
        self.eeg = eeg
        self.visual = visual
        self.target_indices = target_indices
        self.conditions = condition_indices
        self.post_stimulus = post_stimulus

    def __len__(self) -> int:
        return len(self.conditions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        condition = int(self.conditions[index])
        value = np.asarray(self.eeg[condition], dtype=np.float32).mean(0)
        if self.post_stimulus:
            value = value[:, 20:]
        return torch.from_numpy(value.copy()), self.visual[
            int(self.target_indices[condition])
        ].float()


class LinearEEG(nn.Module):
    def __init__(self, samples: int, output_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(17 * samples)
        self.project = nn.Linear(17 * samples, output_dim)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.project(self.norm(eeg.flatten(1)))


class TSConvEEG(nn.Module):
    """Compact temporal-spatial baseline in the spirit of NICE/EEGNet."""

    def __init__(self, samples: int, output_dim: int) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 40, (1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(40, 40, (17, 1), groups=40, bias=False),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.25),
        )
        width = samples // 4
        self.project = nn.Sequential(nn.Flatten(), nn.Linear(40 * width, output_dim))

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.project(self.spatial(self.temporal(eeg[:, None])))


def contrastive_loss(
    prediction: torch.Tensor, target: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    prediction, target = F.normalize(prediction, dim=-1), F.normalize(target, dim=-1)
    logits = prediction @ target.T / temperature
    labels = torch.arange(len(prediction), device=prediction.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def retrieval_metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[dict, torch.Tensor]:
    prediction, target = (
        F.normalize(prediction.float(), dim=-1),
        F.normalize(target.float(), dim=-1),
    )
    scores = prediction @ target.T
    diagonal = scores.diagonal()
    ranks = 1 + (scores > diagonal[:, None]).sum(1)
    result = {
        "top1": float((ranks <= 1).float().mean()),
        "top5": float((ranks <= 5).float().mean()),
        "top10": float((ranks <= 10).float().mean()),
        "mrr": float((1.0 / ranks.float()).mean()),
        "median_rank": float(ranks.float().median()),
        "mean_rank": float(ranks.float().mean()),
        "matched_cosine": float(diagonal.mean()),
        "chance_top1": 1.0 / len(target),
    }
    return result, scores


def shuffled_metrics(scores: torch.Tensor, permutations: int, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    results = []
    for _ in range(permutations):
        order = torch.randperm(len(scores), generator=generator)
        correct = scores[torch.arange(len(scores)), order]
        ranks = 1 + (scores > correct[:, None]).sum(1)
        results.append(
            (
                (ranks <= 1).float().mean().item(),
                (ranks <= 5).float().mean().item(),
                ranks.float().median().item(),
            )
        )
    values = np.asarray(results)
    return {
        "permutations": permutations,
        "top1_mean": float(values[:, 0].mean()),
        "top1_std": float(values[:, 0].std()),
        "top5_mean": float(values[:, 1].mean()),
        "top5_std": float(values[:, 1].std()),
        "median_rank_mean": float(values[:, 2].mean()),
    }


@torch.no_grad()
def predict_baseline(
    model: nn.Module,
    eeg: np.ndarray,
    repeat_count: int,
    draw: int,
    seed: int,
    device: torch.device,
    batch_size: int,
    post_stimulus: bool = True,
) -> torch.Tensor:
    rng = np.random.default_rng(seed + repeat_count * 1009 + draw * 9176)
    if repeat_count == eeg.shape[1]:
        indices = np.broadcast_to(np.arange(eeg.shape[1]), (eeg.shape[0], eeg.shape[1]))
    else:
        indices = np.stack(
            [rng.choice(eeg.shape[1], repeat_count, replace=False) for _ in range(eeg.shape[0])]
        )
    outputs = []
    model.eval()
    for start in range(0, eeg.shape[0], batch_size):
        stop = min(start + batch_size, eeg.shape[0])
        batch = np.stack(
            [
                np.asarray(eeg[row, indices[row]], dtype=np.float32).mean(0)
                for row in range(start, stop)
            ]
        )
        if post_stimulus:
            batch = batch[:, :, 20:]
        tensor = torch.from_numpy(batch).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            outputs.append(model(tensor).float().cpu())
    return torch.cat(outputs)


def mean_metric_dict(records: list[dict]) -> dict:
    return {key: float(np.mean([record[key] for record in records])) for key in records[0]}


def train_baseline(
    kind: str,
    subject: int,
    eeg: np.ndarray,
    visual: torch.Tensor,
    target_indices: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    output: Path,
) -> nn.Module:
    checkpoint = output / f"{kind}_subject_{subject + 1:02d}.pt"
    post_stimulus = kind != "linear_full"
    samples = 80 if post_stimulus else 100
    constructor = LinearEEG if kind.startswith("linear") else TSConvEEG
    model = constructor(samples, visual.shape[1]).to(device)
    if args.resume_baselines and checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        print(f"loaded {checkpoint}", flush=True)
        return model
    train_set = AveragedTrainingDataset(eeg, visual, target_indices, train_indices, post_stimulus)
    val_set = AveragedTrainingDataset(
        eeg, visual, target_indices, validation_indices, post_stimulus
    )
    generator = torch.Generator().manual_seed(args.seed + subject)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state, best_rank, stale = None, math.inf, 0
    history_path = output / f"{kind}_subject_{subject + 1:02d}_history.jsonl"
    for epoch in range(args.epochs):
        model.train()
        total, batches = 0.0, 0
        for batch_eeg, target in train_loader:
            batch_eeg, target = (
                batch_eeg.to(device, non_blocking=True),
                target.to(device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = contrastive_loss(model(batch_eeg), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item()
            batches += 1
        predictions, targets = [], []
        model.eval()
        with torch.no_grad():
            for batch_eeg, target in val_loader:
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    predictions.append(model(batch_eeg.to(device)).float().cpu())
                targets.append(target)
        metrics, _ = retrieval_metrics(torch.cat(predictions), torch.cat(targets))
        record = {"epoch": epoch, "train_loss": total / max(batches, 1), "validation": metrics}
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"{kind} S{subject + 1:02d} epoch={epoch:03d} loss={record['train_loss']:.4f} "
            f"val_top1={metrics['top1']:.4f} val_median={metrics['median_rank']:.1f}",
            flush=True,
        )
        if metrics["median_rank"] < best_rank:
            best_rank, stale = metrics["median_rank"], 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("baseline training produced no checkpoint")
    model.load_state_dict(best_state)
    torch.save(
        {
            "model": best_state,
            "kind": kind,
            "subject": subject,
            "validation_median_rank": best_rank,
        },
        checkpoint,
    )
    return model


def reliability(eeg: np.ndarray) -> dict:
    first = np.asarray(eeg[:, :40], dtype=np.float32).mean(1)[:, :, 20:]
    second = np.asarray(eeg[:, 40:], dtype=np.float32).mean(1)[:, :, 20:]
    a, b = first.reshape(200, -1), second.reshape(200, -1)
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    raw = np.sum(a * b, 1) / np.maximum(
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12
    )
    result = {
        "raw_poststim_split_half_pearson": float(np.mean(raw)),
        "raw_poststim_spearman_brown": float(2 * np.mean(raw) / max(1 + np.mean(raw), 1e-12)),
    }
    return result


def verdict(report: dict) -> dict:
    methods = report["methods"]
    best = None
    for name, payload in methods.items():
        metric = payload["aggregate"]["80"]["top1"]
        if best is None or metric > best[1]:
            best = (name, metric)
    decision = "use_as_semantic_initialization"
    explanation = (
        "The best linear EEG-to-DINO baseline initializes the subject-adaptive semantic encoder."
    )
    return {
        "decision": decision,
        "explanation": explanation,
        "best_method": best[0] if best else None,
        "best_top1": best[1] if best else None,
    }


def main() -> None:
    args = arguments()
    if sorted(set(args.subjects)) != sorted(args.subjects) or any(
        s < 0 or s > 9 for s in args.subjects
    ):
        raise ValueError("subjects must be unique zero-based IDs in 0..9")
    if args.repeat_draws < 1 or args.shuffle_permutations < 1:
        raise ValueError("repeat-draws and shuffle-permutations must be positive")
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_bank = torch.load(args.training_bank, map_location="cpu", weights_only=False)
    test_bank = torch.load(args.test_bank, map_location="cpu", weights_only=False)
    visual_train = F.normalize(train_bank["dino_global"].float(), dim=-1)
    visual_test = F.normalize(test_bank["dino_global"].float(), dim=-1)
    if visual_test.shape[0] != 200 or visual_train.shape[1] != visual_test.shape[1]:
        raise RuntimeError("incompatible training and official test target banks")
    target_indices, train_indices, validation_indices = training_target_indices(
        Path(args.index), train_bank
    )
    archives, cache = Path(args.archives), Path(args.cache)
    report = {
        "protocol": {
            "official_test_conditions": 200,
            "official_test_repetitions": 80,
            "training_conditions_available": 16540,
            "training_repetitions": 4,
            "internal_train_conditions": len(train_indices),
            "internal_validation_conditions": len(validation_indices),
            "subjects": args.subjects,
            "repeat_counts": list(REPEAT_COUNTS),
            "repeat_draws": args.repeat_draws,
            "shuffle_permutations": args.shuffle_permutations,
            "target": train_bank.get("metadata", {}).get("teacher", "DINO"),
            "seed": args.seed,
        },
        "reliability": {},
        "methods": {},
    }
    method_subject_predictions: dict[str, dict[int, dict[int, list[torch.Tensor]]]] = {
        name: {} for name in args.models
    }
    for subject in args.subjects:
        print(f"=== subject {subject + 1:02d}/10 ===", flush=True)
        train_eeg = cached_array(archives, cache, subject, "training")
        test_eeg = cached_array(archives, cache, subject, "test")
        report["reliability"][str(subject)] = reliability(test_eeg)
        for name in args.models:
            model = train_baseline(
                name,
                subject,
                train_eeg,
                visual_train,
                target_indices,
                train_indices,
                validation_indices,
                args,
                device,
                output,
            )
            method_subject_predictions[name][subject] = {}
            for count in REPEAT_COUNTS:
                draws = 1 if count == 80 else args.repeat_draws
                predictions = []
                for draw in range(draws):
                    value = predict_baseline(
                        model,
                        test_eeg,
                        count,
                        draw,
                        args.seed + subject * 100003,
                        device,
                        args.eval_batch_size,
                        name != "linear_full",
                    )
                    predictions.append(value)
                method_subject_predictions[name][subject][count] = predictions
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del train_eeg, test_eeg

    for name in args.models:
        per_subject: dict[str, dict] = {}
        aggregate: dict[str, dict] = {}
        for count in REPEAT_COUNTS:
            draw_metrics = []
            ensemble_predictions = []
            for draw in range(1 if count == 80 else args.repeat_draws):
                subject_predictions = [
                    method_subject_predictions[name][subject][count][draw]
                    for subject in args.subjects
                ]
                for subject, prediction in zip(args.subjects, subject_predictions):
                    metrics, _ = retrieval_metrics(prediction, visual_test)
                    per_subject.setdefault(str(subject), {}).setdefault(str(count), []).append(
                        metrics
                    )
                ensemble = torch.stack(subject_predictions).mean(0)
                ensemble_predictions.append(ensemble)
                metrics, _ = retrieval_metrics(ensemble, visual_test)
                draw_metrics.append(metrics)
            aggregate[str(count)] = mean_metric_dict(draw_metrics)
            if count == 80:
                _, scores = retrieval_metrics(ensemble_predictions[0], visual_test)
                aggregate[str(count)]["shuffled"] = shuffled_metrics(
                    scores, args.shuffle_permutations, args.seed + 99
                )
        for subject, subject_records in per_subject.items():
            per_subject[subject] = {
                count: mean_metric_dict(records) for count, records in subject_records.items()
            }
        report["methods"][name] = {"per_subject": per_subject, "aggregate": aggregate}
    report["verdict"] = verdict(report)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2), flush=True)
    print(f"saved decisive benchmark to {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
