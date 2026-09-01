"""Standalone 200-way gate evaluator (Step 1a).

Evaluates a trained semantic encoder checkpoint on the official THINGS-EEG2
200-way retrieval protocol, independently of any trainer. The aggregation is a
verbatim replica of the protocol previously embedded in
``train_semantic_encoder.official_report`` (which itself mirrors the decisive
baseline protocol of ``official_200way_benchmark``): identical repeat-draw RNG
streams, fp16 autocast, ensemble-of-subject-predictions averaging, and the
permutation null at 80 repetitions.

Gate reference (weights ``artifacts/eeg_encoder/baseline/checkpoints/
semantic_encoder.pt``, commit b9360cc): top-1 0.540 / top-5 0.825 /
median rank 1.0.

Seed note: the default seed is 20260714 — the value the 0.540 baseline run used
(trainer default; recorded in eval_config_semantic_encoder.json). The reference
benchmark's own default (20260713) seeds its *baseline training*, not this
protocol. The 80-repetition aggregate uses all repetitions with a single draw,
so the gate numbers themselves are seed-independent; the seed affects only the
lower repeat counts and the permutation null.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .official_200way_benchmark import (
    REPEAT_COUNTS,
    cached_array,
    mean_metric_dict,
    retrieval_metrics,
    shuffled_metrics,
)

MODEL_KINDS = ("semantic-legacy", "semantic-lightning", "ensemble-lightning")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone official 200-way evaluation of a semantic encoder checkpoint"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=MODEL_KINDS, default="semantic-legacy")
    parser.add_argument("--test-bank", default="data/derived/official_test_targets_dinov2s_192.pt")
    parser.add_argument("--archives", default="data/things_eeg2_osf/preprocessed")
    parser.add_argument("--cache", default="data/derived/eeg_float32_cache")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--repeat-draws", type=int, default=8)
    parser.add_argument("--shuffle-permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--device", default="cuda")
    # Step 1b: krzywa N. Podanie --ensemble-n wlacza tryb ensemble (fuzja po
    # podmiotach na 80 repetycjach); brak flagi = niezmieniona sciezka legacy.
    parser.add_argument("--ensemble-n", nargs="+", type=int, default=None)
    parser.add_argument("--subset-draws", type=int, default=10)
    parser.add_argument("--subset-seed", type=int, default=20260714)
    parser.add_argument("--output", default=None, help="optional path for the JSON report")
    parser.add_argument(
        "--reference-report",
        default=None,
        help="optional prior report (this script's or official_report.json) to diff against",
    )
    return parser.parse_args()


def load_model(kind: str, checkpoint: str | Path, device: torch.device):
    """Return a callable ``model(eeg_3d, subject_ids) -> [batch, dim]`` in eval mode."""

    if kind == "semantic-legacy":
        from ..models.semantic_encoder import load_semantic_encoder

        return load_semantic_encoder(checkpoint, device)
    if kind == "semantic-lightning":
        from ..lightning.eeg_encoder import SemanticEncoderLightning

        module = SemanticEncoderLightning.load_from_checkpoint(
            checkpoint, map_location="cpu", strict=True
        )
        return module.to(device).eval()
    if kind == "ensemble-lightning":
        from ..lightning.ensemble_trunk import EnsembleTrunkLightning

        module = EnsembleTrunkLightning.load_from_checkpoint(
            checkpoint, map_location="cpu", strict=True
        )
        return module.to(device).eval()
    raise SystemExit(f"unknown model kind: {kind}")


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
    # Verbatim protocol replica — do not "improve": the gate depends on it.
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


def ensemble_report(model, kind: str, arrays, visual_test, args, device) -> dict:
    """N-curve at 80 repetitions: per-subject embeddings once, then fusion over
    subject subsets. For the Lightning ensemble model the learned fusion is
    used (identity-indexed); legacy kinds get the raw-mean gate fusion."""

    embed_model = model.trunk if hasattr(model, "trunk") else model
    if kind == "ensemble-lightning":

        def fuse(embeddings: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
            fused, _ = model.fusion(embeddings.to(device), ids.to(device))
            return fused.cpu()

    else:

        def fuse(embeddings: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
            return embeddings.mean(1)

    per_subject = [
        predict_official(embed_model, eeg, subject, 80, 0, args.seed, device, args.eval_batch_size)
        for subject, eeg in enumerate(arrays)
    ]
    stacked = torch.stack(per_subject, dim=1)  # [200, 10, dim]
    generator = torch.Generator().manual_seed(args.subset_seed)
    report = {}
    for count in sorted(set(args.ensemble_n)):
        if count == 1:
            subsets = [[subject] for subject in range(10)]
        elif count >= 10:
            subsets = [list(range(10))]
        else:
            subsets = [
                sorted(torch.randperm(10, generator=generator)[:count].tolist())
                for _ in range(args.subset_draws)
            ]
        records, first_scores = [], None
        for subset in subsets:
            ids = torch.tensor(subset)
            with torch.no_grad():
                fused = fuse(stacked[:, ids], ids)
            metrics, scores = retrieval_metrics(fused, visual_test)
            records.append(metrics)
            if first_scores is None:
                first_scores = scores
        entry = {
            "subsets": subsets,
            "aggregate": mean_metric_dict(records),
            "shuffled": shuffled_metrics(first_scores, args.shuffle_permutations, args.seed + 99),
        }
        if count == 1:
            entry["per_subject"] = {
                str(subset[0]): record for subset, record in zip(subsets, records)
            }
        report[str(count)] = entry
        print(
            f"N={count}: top1={entry['aggregate']['top1']:.4f} "
            f"top5={entry['aggregate']['top5']:.4f} "
            f"median={entry['aggregate']['median_rank']:.1f} "
            f"({len(subsets)} podzbior(ow))",
            flush=True,
        )
    return report


def main() -> None:
    args = arguments()
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    model = load_model(args.model, args.checkpoint, device)
    test_bank = torch.load(args.test_bank, map_location="cpu", weights_only=False)
    visual_test = F.normalize(test_bank["dino_global"].float(), dim=-1)
    test_arrays = [
        cached_array(Path(args.archives), Path(args.cache), subject, "test")
        for subject in range(10)
    ]

    if args.ensemble_n:
        result = {
            "protocol": {
                "conditions": 200,
                "repetitions": 80,
                "subjects": 10,
                "seed": args.seed,
                "subset_seed": args.subset_seed,
                "subset_draws": args.subset_draws,
                "model": args.model,
                "checkpoint": str(args.checkpoint),
            },
            "ensemble": ensemble_report(model, args.model, test_arrays, visual_test, args, device),
        }
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        summary = {
            n: {
                key: round(entry["aggregate"][key], 4)
                for key in ("top1", "top5", "median_rank", "mrr")
            }
            for n, entry in result["ensemble"].items()
        }
        print(json.dumps(summary, indent=2), flush=True)
        return

    result = {
        "protocol": {
            "conditions": 200,
            "repetitions": 80,
            "subjects": 10,
            "seed": args.seed,
            "model": args.model,
            "checkpoint": str(args.checkpoint),
        },
        "semantic_encoder": official_report(model, test_arrays, visual_test, args, device),
    }

    if args.reference_report:
        reference = json.loads(Path(args.reference_report).read_text(encoding="utf-8"))
        ref80 = reference["semantic_encoder"]["aggregate"]["80"]
        got80 = result["semantic_encoder"]["aggregate"]["80"]
        result["reference_delta"] = {
            key: got80[key] - ref80[key]
            for key in ("top1", "top5", "top10", "mrr", "median_rank")
            if key in ref80
        }

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary = {
        "80_repeats": result["semantic_encoder"]["aggregate"]["80"],
        "reference_delta": result.get("reference_delta"),
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
