"""Prepare paired text-only and text+image-prior conditions from EEG predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .dit_ten_subjects import OUTPUT


def text_for_predictions(records, bank):
    lookup = {name: i for i, name in enumerate(bank["category_names"])}
    ids = [lookup[r["predicted_category"]] for r in records]
    return {"embeds": bank["embeds"][ids], "mask": bank["mask"][ids]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", type=Path, default=OUTPUT / "category_classifier")
    parser.add_argument("--text-bank", type=Path, default=OUTPUT / "category_text.pt")
    parser.add_argument("--output", type=Path, default=OUTPUT / "text_comparison_conditions.pt")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("conditions exist; use a fresh output")
    base = torch.load(args.classifier / "conditions.pt", weights_only=False)
    text = torch.load(args.text_bank, weights_only=False, mmap=True)
    records = json.loads((args.classifier / "predictions.json").read_text())
    generator = torch.Generator().manual_seed(20260913)
    positions = (
        list(range(8)) + (torch.randperm(len(records) - 8, generator=generator)[:16] + 8).tolist()
    )
    selected = [records[i] for i in positions]
    prompts = text_for_predictions(selected, text)
    prior = base["variants"]["classified_train_prior"][positions]
    result = {
        "rows": [base["rows"][i] for i in positions],
        "variants": {
            "category_text_only": torch.zeros_like(prior),
            "category_text_and_prior": prior,
        },
        "text_conditions": {
            "category_text_only": {**prompts, "lambda_override": 0.0},
            "category_text_and_prior": prompts,
        },
        "metadata": {
            **base["metadata"],
            "text": text["metadata"],
            "sample_policy": "first 8 validation categories + 16 seeded random remaining validation images; no cherry picking",
            "validation_positions": positions,
            "predictions": selected,
            "generation_claim": "predicted category depiction; no proof of instance geometry",
        },
    }
    torch.save(result, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
