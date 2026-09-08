"""Shared CLI plumbing for the Lightning train scripts (CLAUDE.md conventions):
``--run-name`` is REQUIRED with no default; a run directory that already holds
checkpoints refuses to start without an explicit ``--overwrite``."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-name",
        required=True,
        help="nazwa przebiegu (czlowiek, bez daty); artifacts/<model>/<run-name>/",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")


def prepare_run_dir(model_name: str, args: argparse.Namespace) -> Path:
    run_dir = Path("artifacts") / model_name / args.run_name
    checkpoints = run_dir / "checkpoints"
    existing = sorted(checkpoints.glob("*.ckpt")) if checkpoints.exists() else []
    if existing and not args.overwrite:
        raise SystemExit(
            f"ODMOWA STARTU: {checkpoints} zawiera checkpointy ({existing[0].name}, ...). "
            "Nadpisanie wylacznie przez jawne --overwrite."
        )
    checkpoints.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return run_dir


def build_trainer(
    run_dir: Path,
    args: argparse.Namespace,
    max_epochs: int,
    gradient_clip_val: float,
    monitor: str,
    patience: int,
    mode: str = "min",
    check_val_every: int = 1,
    accumulate_grad_batches: int = 1,
    save_weights_only: bool = False,
) -> L.Trainer:
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    return L.Trainer(
        max_epochs=1 if args.smoke else max_epochs,
        accelerator="gpu" if use_cuda else "cpu",
        devices=1,
        # fp16 autocast jak w starych trenerach (torch.autocast('cuda', float16)).
        precision="16-mixed" if use_cuda else "32-true",
        gradient_clip_val=gradient_clip_val,
        check_val_every_n_epoch=check_val_every,
        accumulate_grad_batches=accumulate_grad_batches,
        default_root_dir=str(run_dir),
        logger=CSVLogger(save_dir=str(run_dir), name="logs"),
        callbacks=[
            ModelCheckpoint(
                dirpath=str(run_dir / "checkpoints"),
                filename="best",
                monitor=monitor,
                mode=mode,
                save_top_k=1,
                save_last=True,
                save_weights_only=save_weights_only,
            ),
            EarlyStopping(monitor=monitor, mode=mode, patience=patience),
        ],
        deterministic=False,
        enable_progress_bar=True,
    )
