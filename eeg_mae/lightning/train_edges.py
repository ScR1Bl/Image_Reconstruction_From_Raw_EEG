"""Lightning entry point for the edges branch (Step 1a structural port).

Starts from a converted Lightning checkpoint (baseline-lightning) — this step
trains nothing from scratch. Selection-metric parity is a Step 1b item.
"""

from __future__ import annotations

import argparse

import lightning as L

from .cli import add_run_arguments, build_trainer, prepare_run_dir
from .data import EdgesDataModule
from .edges import EdgesLightning

MODEL_NAME = "edges"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the edges branch (Lightning)")
    add_run_arguments(parser)
    parser.add_argument(
        "--init-checkpoint", default="artifacts/edges/baseline-lightning/checkpoints/edges.ckpt"
    )
    parser.add_argument("--visual-bank", default="data/derived/visual_targets_dinov2s_192.pt")
    parser.add_argument("--target-bank", default="data/derived/pyramid_edge_targets_v1.pt")
    parser.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    parser.add_argument("--archives", default="data/things_eeg2_osf/preprocessed")
    parser.add_argument("--cache", default="data/derived/eeg_float32_cache")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    run_dir = prepare_run_dir(MODEL_NAME, args)
    L.seed_everything(args.seed)
    module = EdgesLightning.load_converted(args.init_checkpoint)
    datamodule = EdgesDataModule(
        visual_bank=args.visual_bank,
        target_bank=args.target_bank,
        index=args.index,
        archives=args.archives,
        cache=args.cache,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        smoke=args.smoke,
    )
    trainer = build_trainer(
        run_dir,
        args,
        max_epochs=args.epochs,
        gradient_clip_val=4.0,
        monitor="val/loss",
        patience=args.patience,
    )
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
