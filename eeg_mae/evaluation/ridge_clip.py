"""Ridge baselines EEG -> CLIP-image embedding (Step 2, Faza 0).

Per-subject ridge regression (closed form, GPU) from either the frozen trunk's
embeddings or raw repetition-averaged EEG onto RAW CLIP image embeddings.
Lambda is selected on the LEADING metric: mean per-subject retrieval MRR in the
full 1654-candidate validation pool (Step 1b-bis compass). The official 200-way
protocol is reported alongside (N=1 mean per-subject + N=10 mean-fused).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .official_200way_benchmark import (
    cached_array,
    retrieval_metrics,
    shuffled_metrics,
    training_target_indices,
)

POST_STIMULUS = 20


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ridge EEG->CLIP baselines")
    parser.add_argument("--clip-bank", default="data/derived/clip_targets_vitl14_laion2b.pt")
    parser.add_argument(
        "--official-clip-bank", default="data/derived/official_clip_targets_vitl14_laion2b.pt"
    )
    parser.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    parser.add_argument("--archives", default="data/things_eeg2_osf/preprocessed")
    parser.add_argument("--cache", default="data/derived/eeg_float32_cache")
    parser.add_argument(
        "--trunk-checkpoint",
        default="artifacts/eeg_encoder/ablacja-per-subject-sel/checkpoints/best.ckpt",
        help="EnsembleTrunkLightning .ckpt (bieg A) dla cech trunku",
    )
    parser.add_argument("--features", choices=("trunk", "raw"), required=True)
    parser.add_argument(
        "--target-key",
        default="clip_global",
        help="klucz embeddingow w bankach (np. dino_global dla kontroli DINOv2)",
    )
    parser.add_argument("--lambdas", nargs="+", type=float, default=(1e1, 1e2, 1e3, 1e4, 1e5, 1e6))
    parser.add_argument("--shuffle-permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def averaged_eeg(array: np.ndarray, conditions: np.ndarray) -> torch.Tensor:
    """Repetition-averaged EEG [len(conditions), 17, 100] as float32 tensor."""
    return torch.from_numpy(np.asarray(array[conditions], dtype=np.float32).mean(1))


def raw_features(eeg: torch.Tensor) -> torch.Tensor:
    return eeg[:, :, POST_STIMULUS:].flatten(1)


@torch.no_grad()
def trunk_features(trunk, eeg: torch.Tensor, subject: int, device) -> torch.Tensor:
    outputs = []
    for start in range(0, len(eeg), 512):
        batch = eeg[start : start + 512].to(device)
        sid = torch.full((len(batch),), subject, dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            outputs.append(trunk(batch, sid).float().cpu())
    return torch.cat(outputs)


def fit_ridge(x: torch.Tensor, y: torch.Tensor, lam: float, device) -> tuple:
    """Closed-form ridge with feature standardization and target centering."""
    x_mean, x_std = x.mean(0), x.std(0).clamp_min(1e-6)
    y_mean = y.mean(0)
    xs = ((x - x_mean) / x_std).to(device)
    ys = (y - y_mean).to(device)
    gram = xs.T @ xs + lam * torch.eye(xs.shape[1], device=device)
    weight = torch.linalg.solve(gram, xs.T @ ys)
    return weight.cpu(), x_mean, x_std, y_mean


def predict(weight, x_mean, x_std, y_mean, x: torch.Tensor) -> torch.Tensor:
    return ((x - x_mean) / x_std) @ weight + y_mean


def main() -> None:
    args = arguments()
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    torch.set_float32_matmul_precision("high")
    bank = torch.load(args.clip_bank, map_location="cpu", weights_only=False)
    clip = bank[args.target_key].float()
    target_indices, train_idx, val_idx = training_target_indices(Path(args.index), bank)
    y_train_rows = torch.from_numpy(target_indices[train_idx])
    y_val_rows = torch.from_numpy(target_indices[val_idx])
    y_train, y_val = clip[y_train_rows], clip[y_val_rows]

    trunk = None
    if args.features == "trunk":
        from ..lightning.ensemble_trunk import EnsembleTrunkLightning

        module = EnsembleTrunkLightning.load_from_checkpoint(
            args.trunk_checkpoint, map_location="cpu", strict=True
        )
        trunk = module.trunk.to(device).eval()

    arrays = [cached_array(Path(args.archives), Path(args.cache), s, "training") for s in range(10)]
    test_arrays = [
        cached_array(Path(args.archives), Path(args.cache), s, "test") for s in range(10)
    ]
    official = torch.load(args.official_clip_bank, map_location="cpu", weights_only=False)
    clip_test = official[args.target_key].float()

    def features_for(array, conditions, subject) -> torch.Tensor:
        eeg = averaged_eeg(array, conditions)
        if args.features == "raw":
            return raw_features(eeg)
        return trunk_features(trunk, eeg, subject, device)

    def official_features(subject) -> torch.Tensor:
        eeg = torch.from_numpy(np.asarray(test_arrays[subject][:, :80], dtype=np.float32).mean(1))
        if args.features == "raw":
            return raw_features(eeg)
        return trunk_features(trunk, eeg, subject, device)

    x_train = [features_for(arrays[s], train_idx, s) for s in range(10)]
    x_val = [features_for(arrays[s], val_idx, s) for s in range(10)]
    x_test = [official_features(s) for s in range(10)]
    print(f"cechy: {args.features}, wymiar {x_train[0].shape[1]}", flush=True)

    results = {}
    best_lam, best_score = None, -1.0
    for lam in args.lambdas:
        models = [fit_ridge(x_train[s], y_train, lam, device) for s in range(10)]
        val_preds = [predict(*models[s], x_val[s]) for s in range(10)]
        val_metrics = [retrieval_metrics(val_preds[s], y_val)[0] for s in range(10)]
        mrr_n1 = sum(m["mrr"] for m in val_metrics) / 10
        results[str(lam)] = {"val_mrr_n1": mrr_n1}
        print(f"lambda={lam:g}: val mrr_n1={mrr_n1:.4f}", flush=True)
        if mrr_n1 > best_score:
            best_lam, best_score = lam, mrr_n1
            best = (models, val_preds, val_metrics)

    models, val_preds, val_metrics = best
    val_top1 = sum(m["top1"] for m in val_metrics) / 10
    val_fused = retrieval_metrics(torch.stack(val_preds).mean(0), y_val)[0]
    test_preds = [predict(*models[s], x_test[s]) for s in range(10)]
    test_metrics = [retrieval_metrics(p, clip_test)[0] for p in test_preds]
    test_top1_n1 = sum(m["top1"] for m in test_metrics) / 10
    test_mrr_n1 = sum(m["mrr"] for m in test_metrics) / 10
    fused_metrics, fused_scores = retrieval_metrics(torch.stack(test_preds).mean(0), clip_test)
    null = shuffled_metrics(fused_scores, args.shuffle_permutations, args.seed + 99)

    norms = torch.cat(test_preds).norm(dim=-1)
    true_norms = clip_test.norm(dim=-1)
    summary = {
        "features": args.features,
        "lambda": best_lam,
        "val_1654": {
            "mrr_n1": best_score,
            "top1_n1": val_top1,
            "n10": {k: val_fused[k] for k in ("top1", "top5", "mrr", "median_rank")},
        },
        "official_200": {
            "top1_n1": test_top1_n1,
            "mrr_n1": test_mrr_n1,
            "per_subject_top1": [round(m["top1"], 4) for m in test_metrics],
            "n10": {k: fused_metrics[k] for k in ("top1", "top5", "mrr", "median_rank")},
            "shuffled": null,
        },
        "prediction_norms": {
            "pred_mean": float(norms.mean()),
            "pred_std": float(norms.std()),
            "true_mean": float(true_norms.mean()),
            "true_std": float(true_norms.std()),
        },
        "lambda_grid": results,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
