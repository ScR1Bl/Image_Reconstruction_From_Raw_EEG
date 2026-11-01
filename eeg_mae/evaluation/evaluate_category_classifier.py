"""Evaluate a frozen classifier on the historical image-disjoint test split.

This split was previously evaluated for the hierarchical encoder. It is not the
untouched single-subject test and is not claimed to be a pristine research test.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ..data.build_category_prior import category
from ..data.dit_ten_subjects import ARTIFACTS, OUTPUT
from ..lightning.common import sha256_of_file
from .reconstruct_eeg import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=OUTPUT / "category_classifier/historical_test"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("test output exists; do not tune against repeated test evaluation")
    torch.set_num_threads(2)
    with Path("data/derived/hierarchical_visual_targets/index.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == "test"]
    if len(rows) != 1654:
        raise ValueError("unexpected historical test size")
    bank = torch.load(OUTPUT / "category_prior/training_bank.pt", weights_only=False)
    if {r["image_index"] for r in rows} & {r["image_index"] for r in bank["rows"]}:
        raise ValueError("test overlaps training references")
    ids = [int(r["image_index"]) for r in rows]
    arrays = [
        np.load(
            f"data/derived/eeg_float32_cache/sub-{s + 1:02d}__preprocessed_eeg_training.npy",
            mmap_mode="r",
        )
        for s in range(10)
    ]
    eeg = np.stack([a[ids] for a in arrays], 1)
    checkpoint = OUTPUT / "category_classifier/checkpoints/best.ckpt"
    # All inference is finished before evaluation labels are used.
    condition, records = predict(
        eeg, bank, checkpoint, ARTIFACTS / "eeg_encoder/clip2b-warm-lr2/checkpoints/best.ckpt"
    )
    truth = [category(r) for r in rows]
    correct = [r["predicted_category"] == y for r, y in zip(records, truth)]
    top5 = [y in r["top5"] for r, y in zip(records, truth)]
    permutation = torch.randperm(
        len(records), generator=torch.Generator().manual_seed(20260913)
    ).tolist()
    shuffled = [records[j]["predicted_category"] == truth[i] for i, j in enumerate(permutation)]
    report = {
        "complete": True,
        "split": "historical image-disjoint test, known categories",
        "previously_evaluated_for_hierarchical_encoder": True,
        "new_single_subject_test_opened": False,
        "classifier_selection": "frozen before this test, using legacy validation only",
        "checkpoint_sha256": sha256_of_file(checkpoint),
        "images": len(rows),
        "classes": 1654,
        "top1": sum(correct) / len(rows),
        "top5": sum(top5) / len(rows),
        "shuffled_top1": sum(shuffled) / len(rows),
        "sequence_disjoint_verified": False,
    }
    args.output.mkdir(parents=True)
    torch.save(
        {"rows": rows, "variants": {"classified_train_prior": condition}, "metadata": report},
        args.output / "conditions.pt",
    )
    (args.output / "predictions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
