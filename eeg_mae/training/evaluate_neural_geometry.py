"""Conditional geometry from out-of-fold neural predictions, validation only.

The final map is a ridge probe, not a diffusion generator. Edge inputs are
predicted masks; color inputs are predicted low-resolution RGB layouts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..data.single_subject import SingleSubjectBank, training_folds
from ..evaluation.geometry_metrics import geometry_metrics, same_category_2afc
from .evaluate_geometry_complementarity import paired_interval
from .train_geometry_probes import fit_from_basis, predict, ridge_basis


def load_predictions(root, task, bank, oof=False):
    paths = (
        [root / f"{task}_400"] if not oof else [root / "oof" / task / f"fold_{i}" for i in range(3)]
    )
    expected_folds = training_folds(bank.rows)
    values = {}
    for fold, path in enumerate(paths):
        report = json.loads((path / "report.json").read_text(encoding="utf-8"))
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if (
            report["protocol_sha256"] != bank.protocol["sha256"]
            or report["subject_id"] != bank.subject
        ):
            raise ValueError(f"prediction provenance mismatch: {path}")
        if config["end_ms"] != 400 or config["task"] != task:
            raise ValueError(f"wrong target or EEG window: {path}")
        if oof and (report["oof_fold"] != fold or config.get("init_ridge") is not None):
            raise ValueError(f"unsafe OOF initialization or wrong fold: {path}")
        saved = np.load(path / ("predictions_oof_r4.npz" if oof else "predictions_val_r4.npz"))
        prediction = torch.from_numpy(saved["prediction"])
        if task == "color":
            prediction = F.adaptive_avg_pool2d(prediction, 8)
        prediction = prediction.flatten(1)
        if task in {"dino", "clip_b32", "synclr"}:
            prediction = F.normalize(prediction, dim=-1)
        for row, output in zip(saved["rows"], prediction):
            row = int(row)
            if row in values or bank.rows[row]["split"] != ("train" if oof else "val"):
                raise ValueError("duplicated prediction or wrong split")
            if oof and expected_folds[row] != fold:
                raise ValueError("prediction was not held out from its EEG model")
            values[row] = output
    return values


def fit_map(train, target, val, truth):
    basis = ridge_basis(train)
    best = None
    for alpha in (0.1, 1, 10, 100, 1000, 10000):
        state = fit_from_basis(basis, target, alpha)
        output = predict(val, state).clamp(0, 1)
        error = float(F.mse_loss(output, truth))
        if best is None or error < best[0]:
            best = error, alpha, state, output
    return best[3], best[1], best[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/single_subject_geometry/sub-01"))
    parser.add_argument("--protocol", default="data/derived/single_subject_geometry/protocol.json")
    parser.add_argument("--subject", type=int, default=0)
    parser.add_argument("--semantic", choices=("dino", "clip_b32", "synclr"), default="synclr")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output_dir = args.output or args.root / "neural_complementarity"
    if (output_dir / "report.json").exists():
        raise RuntimeError("completed report exists; use a new output")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    bank = SingleSubjectBank(args.protocol, args.subject, edge_keys=("shape16",))
    rows = {split: bank.split(split, valid_edges=True) for split in ("train", "val")}
    features = {}
    for label, task in (("S", args.semantic), ("E", "edge"), ("C", "color")):
        features[label] = {}
        for split in ("train", "val"):
            predictions = load_predictions(args.root, task, bank, oof=split == "train")
            features[label][split] = torch.stack([predictions[int(row)] for row in rows[split]])
    target = {s: torch.from_numpy(bank.targets["shape16"][r]).flatten(1) for s, r in rows.items()}
    concepts = [bank.rows[int(row)]["concept"] for row in rows["val"]]
    generator = torch.Generator().manual_seed(20260912)
    # Fixed extra-feature control with exactly the same number of coefficients as S+E+C.
    for label in ("E_noise", "C_noise"):
        source = features[label[0]]
        features[label] = {
            split: torch.randn(value.shape, generator=generator) for split, value in source.items()
        }
    variants = {
        "S": ("S",),
        "S_E": ("S", "E"),
        "S_C": ("S", "C"),
        "S_E_C": ("S", "E", "C"),
        "S_noise": ("S", "E_noise", "C_noise"),
    }
    metrics, scores, alphas = {}, {}, {}

    def record(name, pred):
        pred = pred.reshape(-1, 1, 16, 16)
        truth = target["val"].reshape_as(pred)
        metrics[name] = geometry_metrics(pred, truth, concepts)
        scores[name] = same_category_2afc(pred, truth, concepts)["scores"]
        np.savez_compressed(
            output_dir / f"{name}_val.npz", rows=rows["val"], prediction=pred.numpy()
        )

    for name, labels in variants.items():
        data = {s: torch.cat([features[label][s] for label in labels], 1) for s in rows}
        pred, alpha, state = fit_map(data["train"], target["train"], data["val"], target["val"])
        alphas[name] = alpha
        record(name, pred)
        torch.save(
            {
                "weight": state[0],
                "bias": state[1],
                "inputs": labels,
                "protocol_sha256": bank.protocol["sha256"],
            },
            output_dir / f"{name}.pt",
        )
        if name == "S_E_C":
            # Swap E/C between the two real images of each category. Semantics stays fixed.
            groups = {}
            for i, concept in enumerate(concepts):
                groups.setdefault(concept, []).append(i)
            order = torch.arange(len(concepts))
            for indices in groups.values():
                if len(indices) == 2:
                    order[indices] = torch.tensor(indices[::-1])
            swapped = torch.cat(
                [features["S"]["val"], features["E"]["val"][order], features["C"]["val"][order]], 1
            )
            record("S_E_C_swapped_within_category", predict(swapped, state).clamp(0, 1))
        print(f"{name}: {metrics[name]}", flush=True)
    record("E_direct", features["E"]["val"].clamp(0, 1))
    gains = {
        name: paired_interval(scores[name], scores["S"])
        for name in ("S_E", "S_C", "S_E_C", "S_noise")
    }
    swap_gain = paired_interval(scores["S_E_C"], scores["S_E_C_swapped_within_category"])
    report = {
        "subject_id": args.subject,
        "repeats": 4,
        "end_ms": 400,
        "protocol_sha256": bank.protocol["sha256"],
        "sequence_disjoint_verified": False,
        "test_evaluated": False,
        "semantic": args.semantic,
        "training_inputs": "three-fold held-out neural EEG predictions; no true image input",
        "structural_inputs": "E: predicted mask16, C: predicted RGB pooled to 8x8; not full latent tokens",
        "selection": "validation mask MSE; all confidence intervals exploratory",
        "metrics": metrics,
        "alpha": alphas,
        "incremental_geometry": gains,
        "benefit_over_within_category_swapped_structure": swap_gain,
        "five_point_geometry_gate_met": gains["S_E_C"]["gain"] >= 0.05
        and gains["S_E_C"]["ci95"][0] > 0,
        "diffusion_training_started": False,
        "next_step": "review geometry, timing metadata and artifacts before any diffusion training",
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
