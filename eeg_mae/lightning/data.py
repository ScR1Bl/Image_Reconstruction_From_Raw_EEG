"""LightningDataModules wrapping the ORIGINAL datasets of the old trainers.

Dataset classes and split logic are imported from ``eeg_mae.training``.
Legacy semantic training concatenates subjects and shuffles examples because
Lightning does not support sequential CombinedLoader in its training loop.
Color/edge training uses the original independent trainers.
"""

from __future__ import annotations

from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.utilities import CombinedLoader
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader

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
        return DataLoader(
            ConcatDataset([loader.dataset for loader in self._train_loaders]),
            batch_size=self.hparams.batch_size,
            shuffle=True,
            drop_last=True,
            generator=torch.Generator().manual_seed(self.hparams.seed),
        )

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
        if train_bank.get("metadata", {}).get("complete") is False:
            raise ValueError("training requires a complete target bank")
        visual_train = train_bank[hp.target_key].float()
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
