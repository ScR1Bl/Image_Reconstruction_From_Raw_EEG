"""Train the PixArt-alpha DiT adapter on images only (Step 3, Faza 1).

No EEG: pairs are (VAE latent moments, REAL CLIP-image embedding) of the same
image, TRAIN split only (13 232 images from the holdout index). Val split
(1 654 images) is held out for Faza 2. Periodic generation samples are written
to the run's samples/ dir so drift is visible during training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader, TensorDataset

from ..evaluation.official_200way_benchmark import training_target_indices
from .cli import add_run_arguments, build_trainer, prepare_run_dir
from .dit_adapter import DiTAdapterLightning

MODEL_NAME = "dit_adapter"


class PairDataModule(L.LightningDataModule):
    def __init__(
        self,
        clip_bank: str,
        vae_bank: str,
        index: str,
        batch_size: int = 8,
        seed: int = 20260714,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        clip = torch.load(self.hparams.clip_bank, map_location="cpu", weights_only=False)
        vae = torch.load(self.hparams.vae_bank, map_location="cpu", weights_only=False)
        if clip["filenames"] != vae["filenames"]:
            raise RuntimeError("clip i vae bank maja rozny porzadek filenames")
        target_indices, train_idx, val_idx = training_target_indices(Path(self.hparams.index), clip)
        train_rows = torch.from_numpy(target_indices[train_idx]).long()
        val_rows = torch.from_numpy(target_indices[val_idx]).long()
        clip_emb = clip["clip_global"].float()
        moments = vae["latent_moments"]
        self._train = TensorDataset(moments[train_rows], clip_emb[train_rows])
        self._val = TensorDataset(moments[val_rows], clip_emb[val_rows])

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
        return DataLoader(self._val, batch_size=self.hparams.batch_size, shuffle=False)


class SampleImages(Callback):
    """Co N krokow: generuj kilka obrazow z prawdziwych CLIP-emb held-out."""

    def __init__(self, clip_bank: str, vae_bank: str, index: str, run_dir: Path, every: int = 500):
        self.clip_bank, self.vae_bank, self.index = clip_bank, vae_bank, index
        self.samples_dir = run_dir / "samples"
        self.samples_dir.mkdir(exist_ok=True)
        self.every = every
        self._val_clip = None

    def _load_val(self, module):
        clip = torch.load(self.clip_bank, map_location="cpu", weights_only=False)
        _, _, val_idx = training_target_indices(Path(self.index), clip)
        import numpy as np

        ti, _, _ = training_target_indices(Path(self.index), clip)
        rows = torch.from_numpy(ti[val_idx][:6]).long()
        self._val_clip = clip["clip_global"][rows].float()

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == 0 or step % self.every != 0:
            return
        if self._val_clip is None:
            self._load_val(module)
        module.eval()
        with torch.no_grad():
            latents = module.sample_images(self._val_clip.to(module.device), steps=20)
            grid = module.decode_and_grid(latents)
        grid.save(self.samples_dir / f"step_{step:06d}.png")
        module.train()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PixArt DiT adapter (Lightning)")
    add_run_arguments(parser)
    parser.add_argument("--clip-bank", default="data/derived/clip_targets_vitl14_laion2b.pt")
    parser.add_argument("--vae-bank", default="data/derived/vae_latents_pixart512.pt")
    parser.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    parser.add_argument("--image-tokens", type=int, default=8)
    parser.add_argument("--cond-dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--resume", default=None, help="wznow z .ckpt (Lightning ckpt_path)")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    run_dir = prepare_run_dir(MODEL_NAME, args)
    L.seed_everything(args.seed)
    module = DiTAdapterLightning(
        clip_dim=768,
        image_tokens=args.image_tokens,
        cond_dropout=args.cond_dropout,
        lr=args.lr,
    )
    datamodule = PairDataModule(
        args.clip_bank, args.vae_bank, args.index, batch_size=args.batch_size, seed=args.seed
    )
    trainer = build_trainer(
        run_dir,
        args,
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
        monitor="val/loss",
        patience=args.patience,
        accumulate_grad_batches=args.accumulate,
        # Checkpoint bez stanu optymalizatora: 3x mniejszy plik, wezsze okno
        # na konflikt z synchronizacja OneDrive (crash zapisu, 2026-09-02).
        save_weights_only=True,
    )
    trainer.callbacks.append(
        SampleImages(args.clip_bank, args.vae_bank, args.index, run_dir, args.sample_every)
    )
    if args.resume:
        module = DiTAdapterLightning.load_from_checkpoint(
            args.resume, map_location="cpu", strict=False
        )
        print(f"wznowiono wagi adaptera z {args.resume}", flush=True)
    trainer.fit(module, datamodule=datamodule)


if __name__ == "__main__":
    main()
