"""Score already generated PNGs against explicitly paired held-out CLIP targets."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F

from ..data.build_clip_targets import TEACHER, load_embedder
from .official_200way_benchmark import retrieval_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", required=True, help="reconstruct_eeg output directory")
    parser.add_argument("--targets", required=True, help="complete CLIP bank")
    parser.add_argument(
        "--rows",
        required=True,
        type=int,
        nargs="+",
        help="target bank rows in generated sample order",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    directory = Path(args.generated)
    report = json.loads((directory / "report.json").read_text())
    samples = report["samples"]
    if len(samples) != len(args.rows) or not samples:
        raise ValueError("exactly one target row required per generated sample")
    bank = torch.load(args.targets, map_location="cpu", weights_only=False)
    if bank.get("metadata", {}).get("complete") is False:
        raise ValueError("scoring requires a complete target bank")
    if len(set(args.rows)) != len(args.rows) or any(
        row < 0 or row >= len(bank["filenames"]) for row in args.rows
    ):
        raise ValueError("target rows must be distinct valid nonnegative bank indices")
    if bank.get("metadata", {}).get("teacher") != TEACHER or bank["clip_global"].shape[1] != 768:
        raise ValueError("targets must use the baseline CLIP-L/14 teacher")
    targets = bank["clip_global"][args.rows].float()
    embed, _ = load_embedder("clip", TEACHER, torch.device(args.device))
    features = []
    with torch.no_grad():
        for sample in samples:
            with Image.open(directory / sample["image"]) as image:
                features.append(embed([image.convert("RGB")]).cpu())
    features = torch.cat(features)
    metrics = retrieval_metrics(features, targets)[0]
    scores = {
        "generated_vs_true_clip_cosine": float(F.cosine_similarity(features, targets).mean()),
        "paired_candidate_retrieval": metrics,
        "target_rows": args.rows,
        "target_filenames": [bank["filenames"][row] for row in args.rows],
        "candidate_count": len(targets),
        "split_verification": "user supplied row mapping; caller must ensure held-out samples",
    }
    (directory / "evaluation.json").write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
