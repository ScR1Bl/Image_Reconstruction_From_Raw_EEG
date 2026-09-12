"""Train one EEG subject; full Lightning checkpoints, EMA and explicit test opt-in."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from ..data.single_subject import SingleSubjectBank, SingleSubjectDataset, training_folds
from ..evaluation.geometry_metrics import geometry_metrics, retrieval_metrics
from .single_subject import SingleSubjectLightning


@torch.no_grad()
def evaluate(module, dataset, batch_size, device):
    module.to(device).eval()
    predictions, targets, rows = [], [], []
    for batch in DataLoader(dataset, batch_size=batch_size, num_workers=0):
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            output = module(batch["eeg"].to(device), use_ema=True)
        if module.task == "edge":
            pred, truth = output.pyramid.shape_logits16.sigmoid(), batch["shape16"]
        elif module.task == "color":
            pred, truth = output.image.rgb, batch["rgb"]
        else:
            pred, truth = output, batch["target"]
        predictions.append(pred.float().cpu())
        targets.append(truth.float())
        rows.append(batch["row"])
    pred, target, rows = torch.cat(predictions), torch.cat(targets), torch.cat(rows)
    if module.task == "edge":
        concepts = [dataset.bank.rows[int(i)]["concept"] for i in rows]
        metrics = geometry_metrics(pred, target, concepts)
    elif module.task == "color":
        from ..models.color import rgb_to_oklab

        metrics = {
            "rgb_mae": float((pred - target).abs().mean()),
            "chroma_mae": float(
                (rgb_to_oklab(pred)[:, 1:] - rgb_to_oklab(target)[:, 1:]).abs().mean()
            ),
        }
    else:
        metrics = retrieval_metrics(pred, target)
    return metrics, rows.numpy(), pred.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", choices=("dino", "clip_b32", "synclr", "edge", "color"), required=True
    )
    parser.add_argument("--protocol", default="data/derived/single_subject_geometry/protocol.json")
    parser.add_argument("--subject", type=int, default=0)
    parser.add_argument("--end-ms", type=int, choices=(200, 400, 800), default=800)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--repeat-augmentation", action="store_true")
    parser.add_argument("--init-ridge", type=Path)
    parser.add_argument("--oof-fold", type=int, choices=(-1, 0, 1, 2), default=-1)
    parser.add_argument("--ridge-alpha", type=float)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    if args.oof_fold >= 0 and (args.init_ridge or args.evaluate_test or args.limit_images):
        raise ValueError(
            "OOF runs must not use full-train initialization, test or image truncation"
        )
    if args.ridge_alpha is not None and args.task not in {"dino", "clip_b32", "synclr"}:
        raise ValueError("ridge initialization is for semantic tasks")
    if (args.output / "checkpoints").exists() and not args.resume:
        raise RuntimeError("existing run: use --resume or choose a fresh directory")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    L.seed_everything(args.seed, workers=True)
    bank = SingleSubjectBank(
        args.protocol, args.subject, load_edges=args.task == "edge", load_rgb=args.task == "color"
    )
    config = {
        **vars(args),
        "protocol_sha256": bank.protocol["sha256"],
        "sequence_disjoint_verified": False,
        "training_subjects": [args.subject],
        "inference_subjects": [args.subject],
        "selection_split": "val",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "lightning_version": L.__version__,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "source_sha256": hashlib.sha256(
            b"".join(
                str(path.relative_to(Path(__file__).parents[1])).encode() + path.read_bytes()
                for path in sorted(Path(__file__).parents[1].rglob("*.py"))
            )
        ).hexdigest(),
    }
    if args.resume:
        previous = json.loads((args.output / "config.json").read_text(encoding="utf-8"))
        for key in (
            "task",
            "subject",
            "end_ms",
            "protocol_sha256",
            "compact",
            "repeat_augmentation",
            "oof_fold",
            "batch_size",
            "seed",
            "lr",
        ):
            if previous.get(key, -1 if key == "oof_fold" else None) != config[key]:
                raise ValueError(f"resume configuration changed {key}")
    else:
        (args.output / "config.json").write_text(
            json.dumps(config, indent=2, default=str), encoding="utf-8"
        )
    datasets = {
        split: SingleSubjectDataset(
            bank,
            split,
            args.task,
            end_ms=args.end_ms,
            repeat_augmentation=args.repeat_augmentation and split == "train",
            limit=args.limit_images,
        )
        for split in ("train", "val")
    }
    folds = training_folds(bank.rows)
    if args.oof_fold >= 0:
        train_rows = datasets["train"].rows
        datasets["train"].rows = train_rows[folds[train_rows] != args.oof_fold]
    loaders = {
        s: DataLoader(
            d,
            batch_size=args.batch_size,
            shuffle=s == "train",
            num_workers=0,
            pin_memory=args.device == "cuda",
            drop_last=s == "train",
        )
        for s, d in datasets.items()
    }
    module = SingleSubjectLightning(
        task=args.task,
        subject_id=args.subject,
        end_ms=args.end_ms,
        lr=args.lr,
        compact=args.compact,
        protocol_sha256=bank.protocol["sha256"],
    )
    if args.init_ridge and not args.resume:
        module.initialize_ridge(args.init_ridge)
    if args.ridge_alpha is not None and not args.resume:
        from ..training.train_geometry_probes import prepare_eeg, ridge_basis, fit_from_basis
        from torch.nn import functional as F

        rows = datasets["train"].rows
        features = prepare_eeg(bank.mean_eeg(4, args.end_ms))[rows]
        targets = F.normalize(torch.from_numpy(bank.targets[args.task][rows]), dim=-1)
        weight, bias = fit_from_basis(ridge_basis(features), targets, args.ridge_alpha)
        with torch.no_grad():
            module.model.linear_weight[0].copy_(weight)
            module.model.linear_bias[0].copy_(bias)
            module.ema.load_state_dict(module.model.state_dict())
    module.concepts = [row["concept"] for row in bank.rows]
    checkpoint = ModelCheckpoint(
        dirpath=args.output / "checkpoints",
        filename="best",
        monitor="val/selection",
        mode="max",
        save_last=True,
        save_top_k=1,
        enable_version_counter=False,
    )
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if use_cuda else "cpu",
        devices=1,
        precision="16-mixed" if use_cuda else "32-true",
        gradient_clip_val=4,
        callbacks=[
            checkpoint,
            EarlyStopping(monitor="val/selection", mode="max", patience=args.patience),
        ],
        logger=CSVLogger(str(args.output), name="logs"),
        default_root_dir=str(args.output),
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
    )
    started = time.monotonic()
    trainer.fit(
        module,
        loaders["train"],
        loaders["val"],
        ckpt_path=str(args.resume) if args.resume else None,
        weights_only=False,  # Our full checkpoint includes optimizer and NumPy RNG state.
    )
    selected = SingleSubjectLightning.load_from_checkpoint(
        checkpoint.best_model_path, map_location="cpu", weights_only=False
    )
    report = {
        "task": args.task,
        "subject_id": args.subject,
        "subjects": 1,
        "protocol_sha256": bank.protocol["sha256"],
        "sequence_disjoint_verified": False,
        "test_evaluated": args.evaluate_test,
        "oof_fold": args.oof_fold,
        "best_checkpoint": checkpoint.best_model_path,
        "fit_seconds": time.monotonic() - started,
        "evaluation": {},
    }
    if args.oof_fold >= 0:
        dataset = SingleSubjectDataset(bank, "train", args.task, end_ms=args.end_ms)
        dataset.rows = dataset.rows[folds[dataset.rows] == args.oof_fold]
        _, rows, pred = evaluate(
            selected, dataset, args.batch_size, torch.device("cuda" if use_cuda else "cpu")
        )
        np.savez_compressed(args.output / "predictions_oof_r4.npz", rows=rows, prediction=pred)
        report["oof_images"] = len(rows)
    for split in ["val"] + (["test"] if args.evaluate_test else []):
        for repeats in (1, 2, 4):
            dataset = SingleSubjectDataset(
                bank, split, args.task, end_ms=args.end_ms, repeats=repeats, limit=args.limit_images
            )
            metrics, rows, pred = evaluate(
                selected, dataset, args.batch_size, torch.device("cuda" if use_cuda else "cpu")
            )
            report["evaluation"][f"{split}_r{repeats}"] = metrics
            np.savez_compressed(
                args.output / f"predictions_{split}_r{repeats}.npz", rows=rows, prediction=pred
            )
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
