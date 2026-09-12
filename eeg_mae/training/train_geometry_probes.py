"""CPU ridge pilot: geometry and semantics from ONE subject, selected on val.

Test evaluation is opt-in. Outputs include the exact linear initialization for
the existing SubjectAdaptiveSemanticEncoder (per-trial LayerNorm input).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..data.single_subject import SingleSubjectBank
from ..evaluation.geometry_metrics import geometry_metrics, retrieval_metrics


def prepare_eeg(eeg):
    value = torch.as_tensor(eeg).float().flatten(1)
    return F.layer_norm(value, (value.shape[1],))


def ridge_basis(x):
    mean = x.mean(0)
    std = x.std(0, unbiased=False).clamp_min(1e-5)
    z = (x - mean) / std
    eigenvalues, eigenvectors = torch.linalg.eigh(z.T @ z)
    return mean, std, z, eigenvalues.clamp_min(0), eigenvectors


def fit_from_basis(basis, y, alpha):
    mean, std, z, eigenvalues, eigenvectors = basis
    ymean = y.mean(0)
    weights = eigenvectors @ (
        (eigenvectors.T @ (z.T @ (y - ymean))) / (eigenvalues[:, None] + alpha)
    )
    # Absorb feature standardization into the linear layer. Input still needs
    # exactly prepare_eeg / SubjectAdaptiveSemanticEncoder's LayerNorm.
    weight = (weights / std[:, None]).T.contiguous()
    bias = ymean - (mean / std) @ weights
    return weight, bias


def predict(x, state):
    return F.linear(x, state[0], state[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="data/derived/single_subject_geometry/protocol.json")
    parser.add_argument("--subject", type=int, default=0)
    parser.add_argument("--windows", nargs="+", type=int, default=[200, 400, 800])
    parser.add_argument("--alphas", nargs="+", type=float, default=[1, 10, 100, 1000, 10000])
    parser.add_argument(
        "--output", type=Path, default=Path("runs/single_subject_geometry/sub-01/probes")
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "report.json").exists():
        raise RuntimeError("choose a fresh output directory; completed reports are immutable")
    bank = SingleSubjectBank(args.protocol, args.subject, edge_keys=("shape16",))
    train, val = bank.split("train"), bank.split("val")
    shape = torch.from_numpy(bank.targets["shape16"]).flatten(1)
    splits = ["val"] + (["test"] if args.evaluate_test else [])
    report = {
        "subject_id": args.subject,
        "subject_count": 1,
        "train_repeats": 4,
        "protocol_sha256": bank.protocol["sha256"],
        "sequence_disjoint_verified": False,
        "selection_split": "val",
        "test_evaluated": args.evaluate_test,
        "clip_teacher": "LAION ViT-B/32 512d; NOT colleague's ViT-L/14 768d",
        "windows": {},
    }
    targets = {
        name: F.normalize(torch.from_numpy(bank.targets[name]), dim=-1)
        for name in ("dino", "clip_b32", "synclr")
    }
    targets["shape"] = shape
    for end_ms in args.windows:
        started = time.monotonic()
        print(f"window={end_ms} preparing ridge bases", flush=True)
        x = prepare_eeg(bank.mean_eeg(4, end_ms))
        basis = ridge_basis(x[train])
        shape_train = bank.split("train", valid_edges=True)
        shape_val = bank.split("val", valid_edges=True)
        shape_basis = ridge_basis(x[shape_train])
        window = {}
        for name, target in targets.items():
            ti, vi = (shape_train, shape_val) if name == "shape" else (train, val)
            best, history = None, []
            for alpha in args.alphas:
                state = fit_from_basis(shape_basis if name == "shape" else basis, target[ti], alpha)
                output = predict(x[vi], state)
                score = (
                    -float(F.mse_loss(output, target[vi]))
                    if name == "shape"
                    else float(F.cosine_similarity(output, target[vi]).mean())
                )
                history.append({"alpha": alpha, "validation_selection_score": score})
                if best is None or score > best[0]:
                    best = (score, alpha, state)
            _, alpha, state = best
            torch.save(
                {
                    "weight": state[0],
                    "bias": state[1],
                    "alpha": alpha,
                    "target": name,
                    "end_ms": end_ms,
                    "subject_id": args.subject,
                    "input_normalization": "per-example layer_norm eps=1e-5",
                    "protocol_sha256": bank.protocol["sha256"],
                },
                args.output / f"{name}_{end_ms}ms.pt",
            )
            entry = {"alpha": alpha, "selection": history, "evaluation": {}}
            for split in splits:
                rows = bank.split(split, valid_edges=name == "shape")
                concepts = [bank.rows[int(i)]["concept"] for i in rows]
                for repeats in (1, 2, 4):
                    xe = x if repeats == 4 else prepare_eeg(bank.mean_eeg(repeats, end_ms))
                    output = predict(xe[rows], state)
                    if name == "shape":
                        metrics = geometry_metrics(
                            output.reshape(-1, 1, 16, 16),
                            target[rows].reshape(-1, 1, 16, 16),
                            concepts,
                        )
                    else:
                        metrics = retrieval_metrics(output, target[rows])
                    entry["evaluation"][f"{split}_r{repeats}"] = metrics
                    np.savez_compressed(
                        args.output / f"{name}_{end_ms}ms_{split}_r{repeats}.npz",
                        rows=rows,
                        prediction=output.numpy(),
                    )
            if name == "shape":
                for split in splits:
                    rows = bank.split(split, valid_edges=True)
                    concepts = [bank.rows[int(i)]["concept"] for i in rows]
                    mean = target[ti].mean(0).expand(len(rows), -1)
                    entry[f"{split}_train_mean"] = geometry_metrics(
                        mean.reshape(-1, 1, 16, 16), target[rows].reshape(-1, 1, 16, 16), concepts
                    )
            window[name] = entry
            print(
                f"window={end_ms} target={name} alpha={alpha} val={entry['evaluation']['val_r4']}",
                flush=True,
            )
        window["elapsed_seconds"] = time.monotonic() - started
        report["windows"][str(end_ms)] = window
        (args.output / "progress.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"completed {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
