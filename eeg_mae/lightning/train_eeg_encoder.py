"""Lightning entry point for the shared EEG semantic trunk.

Structural port of ``train_semantic_encoder`` (Step 1a). Selection metric and
per-epoch subject-order parity are Step 1b items (see data.py docstring);
no training run in Step 1a itself.
"""

from __future__ import annotations

import argparse

import lightning as L

from .cli import add_run_arguments, build_trainer, prepare_run_dir
from .data import SemanticDataModule
from .eeg_encoder import SemanticEncoderLightning

MODEL_NAME = "eeg_encoder"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the EEG semantic trunk (Lightning)")
    add_run_arguments(parser)
    parser.add_argument("--training-bank", default="data/derived/visual_targets_dinov2s_192.pt")
    parser.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    parser.add_argument("--archives", default="data/things_eeg2_osf/preprocessed")
    parser.add_argument("--cache", default="data/derived/eeg_float32_cache")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Lightning .ckpt startowy (np. baseline-lightning trunk)",
    )
    parser.add_argument(
        "--baseline-dir", default=None, help="katalog linear_subject_XX.pt do inicjalizacji od zera"
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--base-lr", type=float, default=3e-5)
    parser.add_argument("--residual-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--base-freeze-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    run_dir = prepare_run_dir(MODEL_NAME, args)
    L.seed_everything(args.seed)
    if args.init_checkpoint:
        module = SemanticEncoderLightning.load_from_checkpoint(
            args.init_checkpoint, map_location="cpu"
        )
    else:
        module = SemanticEncoderLightning(
            base_lr=args.base_lr,
            residual_lr=args.residual_lr,
            weight_decay=args.weight_decay,
            base_freeze_epochs=args.base_freeze_epochs,
            max_epochs=args.epochs,
        )
        if args.baseline_dir:
            module.model.load_linear_baselines(args.baseline_dir)
    datamodule = SemanticDataModule(
        training_bank=args.training_bank,
        index=args.index,
        archives=args.archives,
        cache=args.cache,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        smoke=args.smoke,
    )
    trainer = build_trainer(
        run_dir,
        args,
        max_epochs=args.epochs,
        gradient_clip_val=5.0,
        monitor="val/loss",
        patience=args.patience,
    )
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
