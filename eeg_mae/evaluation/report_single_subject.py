"""Reproducible validation summaries and a fixed, uncurated diagnostic panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..data.single_subject import SingleSubjectBank
from .geometry_metrics import geometry_metrics, same_category_2afc, semantic_category_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/single_subject_geometry/sub-01"))
    parser.add_argument("--protocol", default="data/derived/single_subject_geometry/protocol.json")
    parser.add_argument("--subject", type=int, default=0)
    parser.add_argument("--panel", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    bank = SingleSubjectBank(
        args.protocol, args.subject, edge_keys=("shape16",), load_rgb=args.panel
    )
    result = {
        "subject_id": args.subject,
        "test_evaluated": False,
        "protocol_sha256": bank.protocol["sha256"],
        "sequence_disjoint_verified": False,
        "classification": {},
        "geometry": {},
        "color": {},
    }
    stored = {}
    for task in ("dino", "clip_b32", "synclr", "edge", "color"):
        path = args.root / f"{task}_400"
        if not (path / "report.json").exists():
            continue
        report = json.loads((path / "report.json").read_text(encoding="utf-8"))
        if (
            report["protocol_sha256"] != bank.protocol["sha256"]
            or report["subject_id"] != args.subject
        ):
            raise ValueError("mixed protocols or subjects")
        for repeats in (1, 2, 4):
            saved = np.load(path / f"predictions_val_r{repeats}.npz")
            rows, prediction = saved["rows"], torch.from_numpy(saved["prediction"])
            if any(bank.rows[int(row)]["split"] != "val" for row in rows):
                raise ValueError("expected validation predictions")
            concepts = [bank.rows[int(row)]["concept"] for row in rows]
            if task in {"dino", "clip_b32", "synclr"}:
                target = torch.from_numpy(bank.targets[task][rows])
                result["classification"][f"{task}_r{repeats}"] = {
                    **report["evaluation"][f"val_r{repeats}"],
                    **semantic_category_metrics(prediction, target, concepts),
                }
            elif task == "edge":
                truth = torch.from_numpy(bank.targets["shape16"][rows])
                rng = torch.Generator().manual_seed(20260912)
                shuffled = prediction[torch.randperm(len(prediction), generator=rng)]
                result["geometry"][f"edge_r{repeats}"] = {
                    "aligned": geometry_metrics(prediction, truth, concepts),
                    "shuffled_eeg": geometry_metrics(shuffled, truth, concepts),
                }
            elif args.panel:
                target = (
                    torch.from_numpy(
                        np.array(bank.rgb[bank.rgb_positions[rows]], copy=True)
                    ).float()
                    / 255
                )
                from ..models.color import rgb_to_oklab

                pred_ab = rgb_to_oklab(F.adaptive_avg_pool2d(prediction, 8))[:, 1:]
                truth_ab = rgb_to_oklab(F.adaptive_avg_pool2d(target, 8))[:, 1:]
                result["color"][f"color_r{repeats}"] = {
                    **report["evaluation"][f"val_r{repeats}"],
                    "within_category_chroma_2afc": same_category_2afc(pred_ab, truth_ab, concepts)[
                        "accuracy"
                    ],
                }
            if repeats == 4 and task in {"edge", "color"}:
                stored[task] = {int(row): value for row, value in zip(rows, prediction)}
    output = args.root / "diagnostics"
    output.mkdir(exist_ok=True)
    (output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.panel:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        groups = {}
        for row in bank.split("val", valid_edges=True):
            groups.setdefault(bank.rows[int(row)]["concept"], []).append(int(row))
        candidates = sorted(name for name, rows in groups.items() if len(rows) == 2)
        chosen = np.random.default_rng(20260912).choice(candidates, 4, replace=False)
        rows = [row for concept in chosen for row in groups[concept]]
        layers = ["Stimulus", "Teacher mask", "Ridge mask"]
        if "edge" in stored:
            layers.append("EEG edge mask")
        if "color" in stored:
            layers.append("EEG color")
        ridge = np.load(args.root / "probes" / "shape_400ms_val_r4.npz")
        lookup = {
            int(row): value.reshape(16, 16)
            for row, value in zip(ridge["rows"], ridge["prediction"])
        }
        fig, axes = plt.subplots(
            len(layers), len(rows), figsize=(16, 2.15 * len(layers)), squeeze=False
        )
        for column, row in enumerate(rows):
            rgb = np.asarray(bank.rgb[bank.rgb_positions[row]]).transpose(1, 2, 0)
            values = [rgb, bank.targets["shape16"][row].squeeze(), lookup[row].clip(0, 1)]
            if "edge" in stored:
                values.append(stored["edge"][row].squeeze().numpy())
            if "color" in stored:
                values.append(stored["color"][row].permute(1, 2, 0).numpy())
            for line, value in enumerate(values):
                axes[line, column].imshow(
                    value, cmap="gray", vmin=0, vmax=1, interpolation="nearest"
                )
                axes[line, column].set_xticks([])
                axes[line, column].set_yticks([])
                if column == 0:
                    axes[line, column].set_ylabel(layers[line])
            axes[0, column].set_title(bank.rows[row]["filename"], fontsize=7)
        fig.suptitle(
            "Sub-01 | 0-400 ms | 4 repetitions | validation | fixed random pairs, seed 20260912\nMasks/RGB diagnostics; no diffusion generation; sequence-disjoint audit pending",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(output / "validation_pairs.png", dpi=140)
        plt.close(fig)
        (output / "panel_rows.json").write_text(
            json.dumps(
                {
                    "seed": 20260912,
                    "rows": rows,
                    "filenames": [bank.rows[row]["filename"] for row in rows],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps(result["classification"], indent=2), flush=True)


if __name__ == "__main__":
    main()
