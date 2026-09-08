"""Train the thin CLIP head on precomputed trunk features (Step 2, Faza 1 (d)).

Features file: artifacts/clip_head/features_clip2b_warm_lr2.pt
(train/val/test [M, 10, 768] fp16 + y_* raw CLIP targets), produced by the
Faza-1 ladder script from the clip2b-warm-lr2 trunk.
"""

from __future__ import annotations

import argparse

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset

from .cli import add_run_arguments, build_trainer, prepare_run_dir
from .clip_head import CLIPHeadLightning

MODEL_NAME = "clip_head"


class FeatureDataModule(L.LightningDataModule):
    def __init__(self, features_file: str, batch_size: int = 512, seed: int = 20260714) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        blob = torch.load(self.hparams.features_file, map_location="cpu", weights_only=False)
        train = blob["train"].float()  # [M, 10, dim]
        y_train = blob["y_train"].float()
        flat = train.flatten(0, 1)
        flat_y = y_train[:, None, :].expand_as(train).flatten(0, 1)
        self._train = TensorDataset(flat, flat_y)
        self._val = TensorDataset(blob["val"].float(), blob["y_val"].float())

    def train_dataloader(self):
        generator = torch.Generator().manual_seed(self.hparams.seed)
        return DataLoader(
            self._train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            generator=generator,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(self._val, batch_size=256, shuffle=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CLIP head (Lightning)")
    add_run_arguments(parser)
    parser.add_argument(
        "--features-file", default="artifacts/clip_head/features_clip2b_warm_lr2.pt"
    )
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument("--target-norm-mean", type=float, default=20.42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    run_dir = prepare_run_dir(MODEL_NAME, args)
    L.seed_everything(args.seed)
    module = CLIPHeadLightning(
        hidden=args.hidden,
        mse_weight=args.mse_weight,
        target_norm_mean=args.target_norm_mean,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    datamodule = FeatureDataModule(args.features_file, seed=args.seed)
    trainer = build_trainer(
        run_dir,
        args,
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        monitor="val/mrr_n1",
        patience=args.patience,
        mode="max",
    )
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
