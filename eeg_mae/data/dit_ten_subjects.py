"""Legacy-split, ten-subject EEG conditions for the colleague's frozen DiT.

This experiment explicitly fuses ten participants and is separate from the
single-subject protocol. Its validation was used to select existing checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..lightning.common import sha256_of_file
from ..models.color_encoder import load_color_encoder
from ..models.edge_encoder import load_edge_encoder
from ..models.semantic_encoder import SubjectAdaptiveSemanticEncoder

ARTIFACTS = Path("runs/colleague_artifacts_20260913/artifacts")
OUTPUT = Path("runs/dit_ten_subjects")


def legacy_rows():
    with Path("data/derived/hierarchical_visual_targets/index.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = sorted(csv.DictReader(handle), key=lambda r: int(r["image_index"]))
    if [int(r["image_index"]) for r in rows] != list(range(16540)):
        raise ValueError("image index is incomplete")
    return {split: [r for r in rows if r["split"] == split] for split in ("train", "val")}


def load_archive_features(root=ARTIFACTS):
    return torch.load(
        root / "clip_head/features_clip2b_warm_lr2.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )


def normalize_clip(value, norm):
    return F.normalize(value.float(), dim=-1) * norm


def make_preview(root=ARTIFACTS, output=OUTPUT, count=8):
    rows = legacy_rows()["val"]
    blob = load_archive_features(root)
    if len(rows) != len(blob["val"]):
        raise ValueError("archive feature count does not match legacy validation")
    norm = float(blob["y_train"].float().norm(dim=-1).mean())
    fused = blob["val"][:count].float().mean(1)
    output.mkdir(parents=True, exist_ok=True)
    bundle = {
        "rows": rows[:count],
        "variants": {
            "image_oracle": blob["y_val"][:count].float(),
            "eeg10_raw": fused,
            "eeg10_norm": normalize_clip(fused, norm),
        },
        "metadata": {
            "subjects": list(range(10)),
            "repeats_per_subject": 4,
            "end_ms": 800,
            "fusion": "arithmetic mean of raw per-person embeddings",
            "split": "legacy validation; not untouched test",
            "test_evaluated": False,
            "sequence_disjoint_verified": False,
            "clip_norm_from_training": norm,
            "oracle_warning": "image_oracle uses the true image embedding; other rows use cached EEG predictions",
            "feature_cache": str(root / "clip_head/features_clip2b_warm_lr2.pt"),
        },
    }
    torch.save(bundle, output / "baseline_conditions.pt")
    return bundle


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(2)
    make_preview(args.artifacts, args.output)
    if args.preview_only:
        return
    output = args.output / "conditions"
    if (output / "manifest.json").exists():
        raise RuntimeError("completed conditions exist")
    output.mkdir(parents=True, exist_ok=True)
    rows = legacy_rows()
    archive = load_archive_features(args.artifacts)
    paths = {
        "semantic": args.artifacts / "eeg_encoder/clip2b-warm-lr2/checkpoints/best.ckpt",
        "color": Path("runs/eeg_color_encoder_direct/best.pt"),
        "edge": Path("runs/eeg_edge_encoder_direct/best.pt"),
    }
    state = torch.load(paths["semantic"], map_location="cpu", weights_only=False, mmap=True)
    semantic = SubjectAdaptiveSemanticEncoder(output_dim=768).eval().requires_grad_(False)
    semantic.load_state_dict(
        {
            k.removeprefix("trunk."): v
            for k, v in state["state_dict"].items()
            if k.startswith("trunk.")
        },
        strict=True,
    )
    semantic = semantic.cuda()
    color, _ = load_color_encoder(paths["color"].parent, "cuda")
    edge, _ = load_edge_encoder(paths["edge"].parent, "cuda")
    arrays = [
        np.load(
            f"data/derived/eeg_float32_cache/sub-{sid + 1:02d}__preprocessed_eeg_training.npy",
            mmap_mode="r",
        )
        for sid in range(10)
    ]
    alignments = {}
    for split, split_rows in rows.items():
        n = len(split_rows)
        if n != len(archive[split]):
            raise ValueError("cached features do not match split")
        files = {
            name: np.lib.format.open_memmap(
                output / f"{split}_{name}.npy", mode="w+", dtype=np.float16, shape=(n, *shape)
            )
            for name, shape in {"S": (768,), "E": (8, 8, 192), "C": (8, 8, 192)}.items()
        }
        similarities = []
        previews = {"rgb": [], "shape": []}
        for start in range(0, n, args.batch_size):
            selected = split_rows[start : start + args.batch_size]
            ids = [int(r["image_index"]) for r in selected]
            eeg = torch.from_numpy(
                np.stack([np.asarray(a[ids], dtype=np.float32).mean(1) for a in arrays], 1)
            ).cuda()
            with torch.autocast("cuda", torch.float16):
                per_subject = semantic(
                    eeg.flatten(0, 1), torch.arange(10, device="cuda").repeat(len(eeg))
                ).reshape(len(eeg), 10, 768)
                c, e = color(eeg), edge(eeg)
            reference = archive[split][start : start + len(eeg)].float()
            similarities.append(
                F.cosine_similarity(per_subject.float().cpu(), reference, dim=-1).flatten()
            )
            files["S"][start : start + len(eeg)] = (
                per_subject.float().mean(1).cpu().numpy().astype(np.float16)
            )
            for label, grid in (("E", e.condition.grid16), ("C", c.condition.grid16)):
                pooled = F.adaptive_avg_pool2d(grid.float().permute(0, 3, 1, 2), 8).permute(
                    0, 2, 3, 1
                )
                files[label][start : start + len(eeg)] = pooled.cpu().numpy().astype(np.float16)
            if split == "val" and start < 8:
                previews["rgb"].append(c.image.rgb.float().cpu())
                previews["shape"].append(e.pyramid.shape_logits16.sigmoid().float().cpu())
            if start % (args.batch_size * 50) == 0:
                print(
                    json.dumps({"split": split, "done": start + len(eeg), "total": n}), flush=True
                )
        for value in files.values():
            value.flush()
        alignment = torch.cat(similarities)
        alignments[split] = {
            "mean_cosine_to_colleague_cache": float(alignment.mean()),
            "min_cosine": float(alignment.min()),
        }
        if float(alignment.min()) < 0.999:
            raise ValueError(
                "recomputed EEG predictions do not match cached image/subject alignment"
            )
        np.save(output / f"{split}_target.npy", archive[f"y_{split}"].numpy())
        if previews["rgb"]:
            torch.save(
                {k: torch.cat(v)[:8] for k, v in previews.items()}, output / "encoder_preview.pt"
            )
    metadata = {
        "complete": True,
        "subjects": list(range(10)),
        "repeats_per_subject": 4,
        "end_ms": 800,
        "split": "legacy 8/1/1; validation only; no new single-subject test used",
        "test_evaluated": False,
        "sequence_disjoint_verified": False,
        "rows": rows,
        "checkpoint_sha256": {k: sha256_of_file(p) for k, p in paths.items()},
        "alignment": alignments,
        "structural_features": "independent 10-person encoders; latent grid16 pooled to grid8, 192 channels",
        "stacking_limitation": "bridge train uses in-sample frozen encoder predictions; not cross-fitted",
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(alignments), flush=True)


if __name__ == "__main__":
    main()
