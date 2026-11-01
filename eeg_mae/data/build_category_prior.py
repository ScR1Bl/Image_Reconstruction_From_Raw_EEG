"""Train-only semantic retrieval for category-assisted EEG-to-PixArt inference.

The retrieved photograph supplies a visual prior, NOT recovered EEG geometry.
Validation images and labels are used only for evaluation, never for selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..lightning.common import sha256_of_file
from ..models.hierarchical_semantic import BRANCH_DIMS, load_hierarchical_semantic_encoder
from .dit_ten_subjects import OUTPUT, legacy_rows, load_archive_features


def category(row):
    return Path(row["zip_member"]).parent.name


def training_labels(rows):
    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("the reference bank must contain only training images")
    names = sorted({category(row) for row in rows})
    mapping = {name: i for i, name in enumerate(names)}
    return names, torch.tensor([mapping[category(row)] for row in rows])


def retrieve(query, references, labels, classes):
    """Predict without access to query labels or query image features."""
    q, r = F.normalize(query.float(), dim=-1), F.normalize(references.float(), dim=-1)
    prototypes = torch.zeros(classes, r.shape[1], device=r.device)
    prototypes.index_add_(0, labels, r)
    prototypes = F.normalize(prototypes, dim=-1)
    prototype_scores = q @ prototypes.T
    similarity = q @ r.T
    nearest = similarity.argmax(1)
    predicted = prototype_scores.argmax(1)
    within_class = similarity.masked_fill(labels[None] != predicted[:, None], -torch.inf)
    selected = within_class.argmax(1)
    return prototype_scores, selected, nearest


def accuracy(scores, truth):
    indices = scores.topk(min(5, scores.shape[1]), dim=1).indices
    return {
        "top1": float((indices[:, 0] == truth).float().mean()),
        "top5": float((indices == truth[:, None]).any(1).float().mean()),
        "n": len(truth),
        "classes": scores.shape[1],
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("runs/eeg_hierarchical_encoder/best.pt")
    )
    parser.add_argument("--output", type=Path, default=OUTPUT / "category_prior")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20260913)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise ValueError("completed category experiment exists; use a fresh output")
    rows = legacy_rows()
    names, labels = training_labels(rows["train"])
    train_ids = [int(r["image_index"]) for r in rows["train"]]
    val_ids = [int(r["image_index"]) for r in rows["val"]]
    if set(train_ids) & set(val_ids):
        raise ValueError("training references overlap evaluation images")
    encoder, fusion, _ = load_hierarchical_semantic_encoder(args.checkpoint, "cuda")
    encoder.eval().requires_grad_(False)
    fusion.eval().requires_grad_(False)
    teachers = {
        k: np.load(f"data/derived/hierarchical_visual_targets/{k}.float16.npy", mmap_mode="r")
        for k in BRANCH_DIMS
    }
    references = []
    for start in range(0, len(train_ids), 256):
        ids = train_ids[start : start + 256]
        targets = {
            k: torch.from_numpy(np.array(v[ids], dtype=np.float32)).cuda()
            for k, v in teachers.items()
        }
        fused, _ = fusion(targets)
        references.append(F.normalize(fused, dim=-1).cpu())
    references = torch.cat(references)
    del fusion, teachers
    arrays = [
        np.load(
            f"data/derived/eeg_float32_cache/sub-{sid + 1:02d}__preprocessed_eeg_training.npy",
            mmap_mode="r",
        )
        for sid in range(10)
    ]
    predictions = []
    for start in range(0, len(val_ids), args.batch_size):
        ids = val_ids[start : start + args.batch_size]
        eeg = torch.from_numpy(
            np.stack([np.asarray(a[ids], dtype=np.float32).mean(1) for a in arrays], 1)
        ).cuda()
        with torch.autocast("cuda", torch.float16):
            value = encoder(
                eeg.flatten(0, 1), torch.arange(10, device="cuda").repeat(len(eeg))
            ).fused
        value = F.normalize(value.float().reshape(len(eeg), 10, -1), dim=-1).mean(1)
        predictions.append(F.normalize(value, dim=-1).cpu())
        if start % (args.batch_size * 20) == 0:
            print(f"hierarchical EEG {start + len(ids)}/{len(val_ids)}", flush=True)
    predictions = torch.cat(predictions)
    scores, selected, nearest = retrieve(predictions, references, labels, len(names))
    archive = load_archive_features()
    if len(archive["y_train"]) != len(train_ids):
        raise ValueError("CLIP training bank count mismatch")
    # Ground truth starts here, after every prediction and reference selection.
    mapping = {name: i for i, name in enumerate(names)}
    truth = torch.tensor([mapping[category(r)] for r in rows["val"]])
    baseline_scores, _, _ = retrieve(
        archive["val"].float().mean(1), archive["y_train"], labels, len(names)
    )
    random_order = torch.randperm(len(truth), generator=torch.Generator().manual_seed(20260913))
    report = {
        "complete": True,
        "protocol": "1654 known training categories; held-out images; legacy validation",
        "fusion": "ten subjects, four repetitions each, 0-800ms",
        "hierarchical_prototypes": accuracy(scores, truth),
        "hierarchical_nearest_exemplar_top1": float((labels[nearest] == truth).float().mean()),
        "colleague_clip_prototypes": accuracy(baseline_scores, truth),
        "shuffled_eeg_prototypes": accuracy(scores[random_order], truth),
        "references": "13232 TRAINING photographs only; held-out images excluded",
        "inference_inputs": ["EEG", "known subject ID", "frozen training reference bank"],
        "geometry_warning": "Retrieved reference geometry comes from a training photograph, not reconstructed EEG geometry",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_of_file(args.checkpoint),
        "test_evaluated": False,
        "sequence_disjoint_verified": False,
        "checkpoint_selection": "existing encoder selected using legacy validation",
    }
    top = scores.topk(5, dim=1)
    records = []
    for i, row in enumerate(rows["val"]):
        records.append(
            {
                "filename": row["filename"],
                "true_category": category(row),
                "predicted_category": names[int(top.indices[i, 0])],
                "top5": [names[int(k)] for k in top.indices[i]],
                "cosine_margin": float(top.values[i, 0] - top.values[i, 1]),
                "reference": rows["train"][int(selected[i])],
                "correct": bool(top.indices[i, 0] == truth[i]),
            }
        )
    torch.save(
        {
            "references": references,
            "labels": labels,
            "category_names": names,
            "clip": archive["y_train"].clone(),
            "rows": rows["train"],
            "metadata": report,
        },
        output / "training_bank.pt",
    )
    torch.save(
        {
            "predictions": predictions,
            "scores": scores,
            "selected_train_indices": selected,
            "nearest_train_indices": nearest,
            "rows": rows["val"],
        },
        output / "predictions.pt",
    )
    # Both routes are fixed before looking at validation examples or generations.
    bundle = {
        "rows": rows["val"],
        "variants": {
            "category_train_prior": archive["y_train"][selected].float(),
            "nearest_train_prior": archive["y_train"][nearest].float(),
        },
        "metadata": report,
    }
    torch.save(bundle, output / "conditions.pt")
    (output / "predictions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
