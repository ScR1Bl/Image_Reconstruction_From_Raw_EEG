"""Lightning entry point for the color branch (Step 1a structural port).

Starts from a converted Lightning checkpoint (baseline-lightning) — this step
trains nothing from scratch. Selection-metric parity (predict + selection_score
vs shuffled) is a Step 1b item; the module logs loss components.
"""

from __future__ import annotations

import argparse

import lightning as L

from .cli import add_run_arguments, build_trainer, prepare_run_dir
from .color import ColorLightning
from .data import ColorDataModule

MODEL_NAME = "color"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the color branch (Lightning)")
    add_run_arguments(parser)
    parser.add_argument(
        "--init-checkpoint", default="artifacts/color/baseline-lightning/checkpoints/color.ckpt"
    )
    parser.add_argument("--latent-bank", default="data/derived/full_color_latents_antiring")
    parser.add_argument("--rich-bank", default="data/derived/rich_dino_color_v2")
    parser.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    parser.add_argument("--archives", default="data/things_eeg2_osf/preprocessed")
    parser.add_argument("--cache", default="data/derived/eeg_float32_cache")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    run_dir = prepare_run_dir(MODEL_NAME, args)
    L.seed_everything(args.seed)
    module = ColorLightning.load_converted(args.init_checkpoint)
    datamodule = ColorDataModule(
        latent_bank=args.latent_bank,
        rich_bank=args.rich_bank,
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
