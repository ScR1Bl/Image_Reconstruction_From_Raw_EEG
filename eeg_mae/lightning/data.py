"""LightningDataModules wrapping the ORIGINAL datasets of the old trainers.

Dataset classes and split logic are imported from ``eeg_mae.training`` — never
copied. Known Step-1b items (documented, deliberate):
- Semantic: the old loop permutes the subject order every epoch
  (train_semantic_encoder.py:332-334); ``CombinedLoader('sequential')`` keeps a
  fixed order. Training-dynamics parity is validated in Step 1b before any
  result is trusted.
- Branch validation/selection metrics (predict + selection_score vs shuffled)
  are not ported yet; modules log loss components only.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.utilities import CombinedLoader
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..evaluation.official_200way_benchmark import cached_array, training_target_indices


def _training_arrays(archives: str, cache: str) -> list:
    return [cached_array(Path(archives), Path(cache), subject, "training") for subject in range(10)]


class SemanticDataModule(L.LightningDataModule):
    """Data path of train_semantic_encoder.py:254-301, wrapped."""

    def __init__(
        self,
        training_bank: str = "data/derived/visual_targets_dinov2s_192.pt",
        index: str = "data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv",
        archives: str = "data/things_eeg2_osf/preprocessed",
        cache: str = "data/derived/eeg_float32_cache",
        batch_size: int = 512,
        eval_batch_size: int = 256,
        seed: int = 20260714,
        smoke: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        from ..training.train_semantic_encoder import make_loaders

        hp = self.hparams
        train_bank = torch.load(hp.training_bank, map_location="cpu", weights_only=False)
        visual_train = F.normalize(train_bank["dino_global"].float(), dim=-1)
        target_indices, train_indices, validation_indices = training_target_indices(
            Path(hp.index), train_bank
        )
        if hp.smoke:
            train_indices = train_indices[:1024]
            validation_indices = validation_indices[:512]
        arrays = _training_arrays(hp.archives, hp.cache)
        self._train_loaders = make_loaders(
            arrays, visual_train, target_indices, train_indices, hp.batch_size, True, hp.seed
        )
        self._val_loaders = make_loaders(
            arrays,
            visual_train,
            target_indices,
            validation_indices,
            hp.eval_batch_size,
            False,
            hp.seed,
        )

    def train_dataloader(self):
        return CombinedLoader(self._train_loaders, mode="sequential")

    def val_dataloader(self):
        return CombinedLoader(self._val_loaders, mode="sequential")


class EnsembleEEGDataset(torch.utils.data.Dataset):
    """RAM-preloaded ensemble samples: [10, 17, 100] repetition-averaged EEG per
    stimulus + DINO target. Preload kills the measured 90% data-wait bottleneck
    (Step 1b Faza 0: 44.1 s/epoch, 39.5 s waiting on memmap reads)."""

    def __init__(self, arrays, conditions, targets: torch.Tensor) -> None:
        stacks = []
        for array in arrays:
            subject = np.asarray(array[conditions], dtype=np.float32).mean(1)
            stacks.append(torch.from_numpy(subject))
        self.eeg = torch.stack(stacks, dim=1)  # [M, 10, 17, 100]
        self.targets = targets

    def __len__(self) -> int:
        return len(self.eeg)

    def __getitem__(self, index: int):
        return self.eeg[index], self.targets[index]


class EnsembleDataModule(L.LightningDataModule):
    """Ensemble data for Step 1b trunk training (random-N fusion)."""

    def __init__(
        self,
        training_bank: str = "data/derived/visual_targets_dinov2s_192.pt",
        target_key: str = "dino_global",
        index: str = "data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv",
        archives: str = "data/things_eeg2_osf/preprocessed",
        cache: str = "data/derived/eeg_float32_cache",
        batch_size: int = 512,
        eval_batch_size: int = 256,
        seed: int = 20260714,
        smoke: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        hp = self.hparams
        train_bank = torch.load(hp.training_bank, map_location="cpu", weights_only=False)
        # Targety normalizowane jak dotad; loss i retrieval sa kierunkowe
        # (normalizacja wewnatrz), wiec bank surowy vs znormalizowany bez roznicy.
        visual_train = F.normalize(train_bank[hp.target_key].float(), dim=-1)
        target_indices, train_indices, validation_indices = training_target_indices(
            Path(hp.index), train_bank
        )
        if hp.smoke:
            train_indices = train_indices[:1024]
            validation_indices = validation_indices[:512]
        arrays = _training_arrays(hp.archives, hp.cache)
        train_targets = visual_train[target_indices[train_indices]]
        val_targets = visual_train[target_indices[validation_indices]]
        self._train = EnsembleEEGDataset(arrays, train_indices, train_targets)
        self._val = EnsembleEEGDataset(arrays, validation_indices, val_targets)

    def train_dataloader(self):
        generator = torch.Generator().manual_seed(self.hparams.seed)
        return DataLoader(
            self._train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            generator=generator,
            drop_last=True,
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val, batch_size=self.hparams.eval_batch_size, shuffle=False, num_workers=0
        )


class ColorDataModule(L.LightningDataModule):
    """Data path of train_color_encoder.py:375-402, wrapped."""

    def __init__(
        self,
        latent_bank: str = "data/derived/full_color_latents_antiring",
        rich_bank: str = "data/derived/rich_dino_color_v2",
        index: str = "data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv",
        archives: str = "data/things_eeg2_osf/preprocessed",
        cache: str = "data/derived/eeg_float32_cache",
        batch_size: int = 8,
        eval_batch_size: int = 16,
        smoke: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        from ..training.color_training import EEGColorDataset, split_records

        hp = self.hparams
        latent_root = Path(hp.latent_bank)
        manifest = json.loads((latent_root / "manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("complete"):
            raise RuntimeError("full color latent bank is incomplete")
        arrays = _training_arrays(hp.archives, hp.cache)
        records = {
            split: split_records(Path(hp.index), split, manifest["filenames"])
            for split in ("train", "val", "test")
        }
        if hp.smoke:
            records = {key: value[:16] for key, value in records.items()}
        self._datasets = {
            key: EEGColorDataset(arrays, value, latent_root, Path(hp.rich_bank))
            for key, value in records.items()
        }

    def _loader(self, split: str) -> DataLoader:
        hp = self.hparams
        return DataLoader(
            self._datasets[split],
            batch_size=hp.batch_size if split == "train" else hp.eval_batch_size,
            shuffle=split == "train",
            num_workers=0,
            drop_last=split == "train",
        )

    def train_dataloader(self):
        return self._loader("train")

    def val_dataloader(self):
        return self._loader("val")

    def test_dataloader(self):
        return self._loader("test")


class EdgesDataModule(L.LightningDataModule):
    """Data path of train_edge_encoder.py:365-391, wrapped."""

    def __init__(
        self,
        visual_bank: str = "data/derived/visual_targets_dinov2s_192.pt",
        target_bank: str = "data/derived/pyramid_edge_targets_v1.pt",
        index: str = "data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv",
        archives: str = "data/things_eeg2_osf/preprocessed",
        cache: str = "data/derived/eeg_float32_cache",
        batch_size: int = 8,
        eval_batch_size: int = 16,
        smoke: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        from ..training.train_edge_baseline import EEGPyramidDataset, split_records

        hp = self.hparams
        visual = torch.load(hp.visual_bank, map_location="cpu", weights_only=False)
        target_bank = torch.load(hp.target_bank, map_location="cpu", weights_only=False)
        visual_grid = F.normalize(visual["dino_grid"].float(), dim=-1)
        arrays = _training_arrays(hp.archives, hp.cache)
        records = {
            split: split_records(hp.index, split, visual, target_bank)
            for split in ("train", "val", "test")
        }
        if hp.smoke:
            records = {key: value[:16] for key, value in records.items()}
        self._datasets = {
            key: EEGPyramidDataset(arrays, value, visual_grid, target_bank)
            for key, value in records.items()
        }

    def _loader(self, split: str) -> DataLoader:
        hp = self.hparams
        return DataLoader(
            self._datasets[split],
            batch_size=hp.batch_size if split == "train" else hp.eval_batch_size,
            shuffle=split == "train",
            num_workers=0,
            drop_last=split == "train",
        )

    def train_dataloader(self):
        return self._loader("train")

    def val_dataloader(self):
        return self._loader("val")

    def test_dataloader(self):
        return self._loader("test")
