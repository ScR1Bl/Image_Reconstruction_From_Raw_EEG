"""Cross-fitted semantic-only vs EEG-geometry controls, on validation only."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..data.single_subject import SingleSubjectBank
from ..evaluation.geometry_metrics import geometry_metrics, same_category_2afc
from .train_geometry_probes import fit_from_basis, predict, prepare_eeg, ridge_basis


def select_mapping(xtrain, ytrain, xval, yval):
    basis = ridge_basis(xtrain)
    best = None
    for alpha in (0.1, 1, 10, 100, 1000, 10000):
        state = fit_from_basis(basis, ytrain, alpha)
        output = predict(xval, state)
        score = float(F.mse_loss(output.clamp(0, 1), yval))
        if best is None or score < best[0]:
            best = (score, alpha, output)
    return best[2], best[1]


def paired_interval(a, b, draws=2000):
    # Each concept contributes two dependent image decisions.
    differences = (np.asarray(a).reshape(-1, 2) - np.asarray(b).reshape(-1, 2)).mean(1)
    rng = np.random.default_rng(20260912)
    samples = differences[rng.integers(len(differences), size=(draws, len(differences)))].mean(1)
    return {
        "gain": float(differences.mean()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "unit": "category",
        "split": "validation; exploratory, not final confirmation",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="data/derived/single_subject_geometry/protocol.json")
    parser.add_argument("--subject", type=int, default=0)
    parser.add_argument("--end-ms", type=int, default=400)
    parser.add_argument(
        "--probes", type=Path, default=Path("runs/single_subject_geometry/sub-01/probes")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/single_subject_geometry/sub-01/complementarity")
    )
    args = parser.parse_args()
    torch.set_num_threads(3)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "report.json").exists():
        raise RuntimeError("completed report exists; choose a fresh output")
    bank = SingleSubjectBank(args.protocol, args.subject, edge_keys=("shape16",))
    train = bank.split("train")
    valid_train = bank.split("train", valid_edges=True)
    val = bank.split("val", valid_edges=True)
    shape = torch.from_numpy(bank.targets["shape16"]).flatten(1)
    x = prepare_eeg(bank.mean_eeg(4, args.end_ms))
    names = ("dino", "clip_b32", "synclr")
    targets = {name: F.normalize(torch.from_numpy(bank.targets[name]), dim=-1) for name in names}
    targets["shape"] = shape
    settings = json.loads((args.probes / "report.json").read_text(encoding="utf-8"))
    if (
        settings["protocol_sha256"] != bank.protocol["sha256"]
        or settings["subject_id"] != args.subject
    ):
        raise ValueError("probes belong to another protocol or subject")
    alphas = {name: settings["windows"][str(args.end_ms)][name]["alpha"] for name in targets}
    by_concept = defaultdict(list)
    for row in train:
        by_concept[bank.rows[int(row)]["concept"]].append(int(row))
    folds = np.full(len(bank.rows), -1)
    for group in by_concept.values():
        for index, row in enumerate(group):
            folds[row] = index % 3
    oof = {name: torch.zeros_like(target) for name, target in targets.items()}
    for fold in range(3):
        fit_rows = train[folds[train] != fold]
        held_rows = train[folds[train] == fold]
        if set(fit_rows) & set(held_rows):
            raise AssertionError("OOF leakage")
        basis = ridge_basis(x[fit_rows])
        for name in names:
            state = fit_from_basis(basis, targets[name][fit_rows], alphas[name])
            oof[name][held_rows] = predict(x[held_rows], state)
        fit_shape = fit_rows[bank.valid[fit_rows]]
        state = fit_from_basis(ridge_basis(x[fit_shape]), shape[fit_shape], alphas["shape"])
        oof["shape"][held_rows] = predict(x[held_rows], state)
        print(f"cross-fit fold {fold + 1}/3 complete", flush=True)
    val_predictions = {}
    for name in targets:
        saved = np.load(args.probes / f"{name}_{args.end_ms}ms_val_r4.npz")
        positions = {int(row): i for i, row in enumerate(saved["rows"])}
        val_predictions[name] = torch.from_numpy(
            saved["prediction"][[positions[int(row)] for row in val]]
        )
    concepts = [bank.rows[int(row)]["concept"] for row in val]
    variants = {"E_ridge": val_predictions["shape"]}
    selections = {}
    for name in names:
        sem_train = F.normalize(oof[name][valid_train], dim=-1)
        sem_val = F.normalize(val_predictions[name], dim=-1)
        variants[f"{name}_S"], selections[f"{name}_S"] = select_mapping(
            sem_train, shape[valid_train], sem_val, shape[val]
        )
        combined_train = torch.cat((sem_train, oof["shape"][valid_train]), 1)
        combined_val = torch.cat((sem_val, val_predictions["shape"]), 1)
        variants[f"{name}_S_plus_E"], selections[f"{name}_S_plus_E"] = select_mapping(
            combined_train, shape[valid_train], combined_val, shape[val]
        )
        # Separately labelled visual-feature reference: NOT an EEG prediction.
        variants[f"{name}_image_oracle"], selections[f"{name}_image_oracle"] = select_mapping(
            targets[name][valid_train], shape[valid_train], targets[name][val], shape[val]
        )
    scores, metrics = {}, {}
    for name, output in variants.items():
        output = output.clamp(0, 1).reshape(-1, 1, 16, 16)
        truth = shape[val].reshape(-1, 1, 16, 16)
        metrics[name] = geometry_metrics(output, truth, concepts)
        scores[name] = same_category_2afc(output, truth, concepts)["scores"]
        np.savez_compressed(args.output / f"{name}_val.npz", rows=val, prediction=output.numpy())
        print(f"{name}: {metrics[name]}", flush=True)
    differences = {
        name: paired_interval(scores[f"{name}_S_plus_E"], scores[f"{name}_S"]) for name in names
    }
    report = {
        "subject_id": args.subject,
        "repeats": 4,
        "end_ms": args.end_ms,
        "protocol_sha256": bank.protocol["sha256"],
        "sequence_disjoint_verified": False,
        "test_evaluated": False,
        "semantic_mapping_inputs": "three-fold out-of-fold EEG predictions",
        "E_definition": "ridge from EEG to foreground mask; not the neural edge encoder",
        "hyperparameters": selections,
        "metrics": metrics,
        "incremental_geometry": differences,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(differences), flush=True)


if __name__ == "__main__":
    main()
