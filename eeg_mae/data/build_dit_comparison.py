"""Assemble paired DiT conditions after all four EEG bridges finish."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..evaluation.geometry_metrics import retrieval_metrics
from ..lightning.dit_condition_bridge import BridgeDataset, DiTConditionBridge
from .dit_ten_subjects import OUTPUT, normalize_clip


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.root
    baseline = torch.load(root / "baseline_conditions.pt", map_location="cpu", weights_only=False)
    data = BridgeDataset(root / "conditions", "val")
    target = torch.from_numpy(np.array(data.data["target"], dtype=np.float32))
    original = torch.from_numpy(np.array(data.data["S"], dtype=np.float32))
    norm = baseline["metadata"]["clip_norm_from_training"]
    predictions = {"eeg10_raw": original, "eeg10_norm": normalize_clip(original, norm)}
    for variant in ("S", "S_E", "S_C", "S_E_C"):
        checkpoint = torch.load(
            root / "bridges" / variant / "predictions.pt", map_location="cpu", weights_only=False
        )
        if checkpoint["rows"][:8] != baseline["rows"]:
            raise ValueError("bridge image alignment changed")
        predictions[f"bridge_{variant}"] = checkpoint["prediction"]
    # Same S, incorrect E/C: intervention on inputs, no retraining or image input.
    model = DiTConditionBridge.load_from_checkpoint(
        root / "bridges/S_E_C/checkpoints/best.ckpt", map_location="cpu", weights_only=False
    ).eval()
    order = torch.randperm(len(data), generator=torch.Generator().manual_seed(20260913))
    parts = {
        key: torch.stack(
            [data[i][key] if key == "S" else data[int(order[i])][key] for i in range(8)]
        )
        for key in ("S", "E", "C")
    }
    swapped = model(parts)
    variants = {
        "image_oracle": target[:8],
        **{k: v[:8] for k, v in predictions.items()},
        "bridge_SEC_shuffled": swapped,
    }
    bundle = {
        "rows": baseline["rows"],
        "variants": variants,
        "metadata": {
            **baseline["metadata"],
            "structural_inputs": "independent ten-person edge/color latent grids pooled to 8x8",
            "bridge": "global residual into CLIP768; frozen original PixArt adapter; no native spatial injection",
            "selection": "native CLIP validation MSE; 40 epochs max, patience 7; no generated-image selection",
            "training_limitation": "bridge train uses in-sample frozen encoder outputs; validation used in previous encoder selection",
            "shuffled_structure_source_images": [int(order[i]) for i in range(8)],
        },
    }
    output = root / "comparison_conditions.pt"
    if output.exists():
        raise RuntimeError("comparison conditions already exist")
    torch.save(bundle, output)
    metrics = {}
    for name, pred in predictions.items():
        metrics[name] = {
            **retrieval_metrics(pred, target),
            "native_clip_mse": float(F.mse_loss(pred, target)),
            "matched_cosine": float(F.cosine_similarity(pred, target).mean()),
            "mean_norm": float(pred.norm(dim=-1).mean()),
        }
    (root / "bridge_comparison.json").write_text(
        json.dumps({"metrics": metrics, "metadata": bundle["metadata"]}, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
