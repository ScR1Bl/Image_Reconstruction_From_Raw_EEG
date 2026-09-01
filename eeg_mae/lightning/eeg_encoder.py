"""Lightning shell for the shared EEG semantic trunk.

Wraps the ORIGINAL ``SubjectAdaptiveSemanticEncoder`` by composition — forward
delegates 1:1, so a converted checkpoint reproduces the pre-refactor numbers
exactly (gate: top-1 0.540 on the official 200-way protocol).

Training logic is a faithful port of ``train_semantic_encoder`` (loss imported
from it, LR schedule transcribed from its epoch loop). Training parity beyond a
single step is validated in Step 1b, not here.
"""

from __future__ import annotations

import math

import lightning as L
import torch

from ..models.semantic_encoder import SubjectAdaptiveSemanticEncoder
from ..training.train_semantic_encoder import semantic_loss


class SemanticEncoderLightning(L.LightningModule):
    def __init__(
        self,
        model_config: dict | None = None,
        base_lr: float = 3e-5,
        residual_lr: float = 3e-4,
        weight_decay: float = 1e-3,
        base_freeze_epochs: int = 3,
        max_epochs: int = 60,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = SubjectAdaptiveSemanticEncoder(**(model_config or {}))

    def forward(self, eeg: torch.Tensor, subject: torch.Tensor) -> torch.Tensor:
        return self.model(eeg, subject)

    def compute_training_loss(self, batch) -> tuple[torch.Tensor, dict]:
        eeg, subject, target, _ = batch
        return semantic_loss(self(eeg, subject), target)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, parts = self.compute_training_loss(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.log_dict({f"train/{key}": value for key, value in parts.items()})
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        loss, parts = self.compute_training_loss(batch)
        self.log("val/loss", loss, prog_bar=True)
        self.log_dict({f"val/{key}": value for key, value in parts.items()})

    def configure_optimizers(self):
        return torch.optim.AdamW(
            [
                {
                    "params": self.model.base_parameters(),
                    "lr": self.hparams.base_lr,
                    "name": "base",
                },
                {
                    "params": self.model.residual_parameters(),
                    "lr": self.hparams.residual_lr,
                    "name": "residual",
                },
            ],
            weight_decay=self.hparams.weight_decay,
        )

    def on_train_epoch_start(self) -> None:
        # Transcribed from train_semantic_encoder.py:327-331 (base freeze + cosine decay).
        optimizer = self.optimizers().optimizer
        epoch = self.current_epoch
        base_factor = 0.0 if epoch < self.hparams.base_freeze_epochs else 1.0
        optimizer.param_groups[0]["lr"] = self.hparams.base_lr * base_factor
        progress = epoch / max(self.hparams.max_epochs - 1, 1)
        decay = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        optimizer.param_groups[1]["lr"] = self.hparams.residual_lr * decay
