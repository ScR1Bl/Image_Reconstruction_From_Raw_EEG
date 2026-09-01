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
    # Step 1b: fuzja podmiotow. "legacy" = trening per-subject jak w 1a;
    # "mean"/"attention" = ensemble z losowym N (EnsembleTrunkLightning).
    parser.add_argument("--fusion", choices=("legacy", "mean", "attention"), default="legacy")
    parser.add_argument("--n1-prob", type=float, default=0.25)
    parser.add_argument(
        "--trunk-checkpoint",
        default="artifacts/eeg_encoder/baseline-lightning/checkpoints/semantic_encoder.ckpt",
        help="trunk startowy dla trybow ensemble (Lightning .ckpt)",
    )
    parser.add_argument("--fusion-lr", type=float, default=3e-4)
    parser.add_argument("--ensemble-base-lr", type=float, default=1e-5)
    parser.add_argument("--ensemble-residual-lr", type=float, default=1e-4)
    parser.add_argument(
        "--freeze-trunk",
        action="store_true",
        help="zamroz trunk (N=1 bitowo = init); uczy sie tylko fuzja; monitor=val/mrr_n10",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    run_dir = prepare_run_dir(MODEL_NAME, args)
    L.seed_everything(args.seed)
    if args.fusion != "legacy":
        from .common import sha256_of_file
        from .data import EnsembleDataModule
        from .ensemble_trunk import EnsembleTrunkLightning

        module = EnsembleTrunkLightning(
            trunk_checkpoint=args.trunk_checkpoint,
            trunk_sha256=sha256_of_file(args.trunk_checkpoint),
            fusion=args.fusion,
            n1_prob=args.n1_prob,
            freeze_trunk=args.freeze_trunk,
            base_lr=args.ensemble_base_lr,
            residual_lr=args.ensemble_residual_lr,
            fusion_lr=args.fusion_lr,
            weight_decay=args.weight_decay,
        )
        datamodule = EnsembleDataModule(
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
            # przy zamrozonym trunku val/mrr_n1 jest stala - selekcja po n10
            monitor="val/mrr_n10" if args.freeze_trunk else "val/mrr_n1",
            patience=args.patience,
            mode="max",
        )
        trainer.fit(module, datamodule=datamodule)
        return
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
