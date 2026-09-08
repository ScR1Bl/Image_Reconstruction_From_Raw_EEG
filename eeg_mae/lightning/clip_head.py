"""Thin CLIP head on trunk features (Step 2, Faza 1 wariant (d)).

Residual bottleneck MLP with zero-init output — starts as the identity, i.e.
exactly variant (a) raw-trunk performance (house identity-at-init style).
Loss: contrastive (semantic_loss) + mse_weight * scale-normalized MSE on RAW
embeddings. Selection metric: val mrr_n1 in the full 1654 pool.
"""

from __future__ import annotations

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F

from ..evaluation.official_200way_benchmark import retrieval_metrics
from ..training.train_semantic_encoder import semantic_loss


class CLIPHeadLightning(L.LightningModule):
    def __init__(
        self,
        dim: int = 768,
        hidden: int = 128,
        mse_weight: float = 0.25,
        target_norm_mean: float = 20.42,
        lr: float = 3e-4,
        weight_decay: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.head(features)

    def compute_loss(self, batch) -> tuple[torch.Tensor, dict]:
        features, target = batch
        prediction = self(features)
        contrastive, parts = semantic_loss(prediction, target)
        scale = self.hparams.target_norm_mean**2
        mse = F.mse_loss(prediction, target) / scale
        loss = contrastive + self.hparams.mse_weight * mse
        return loss, {
            **parts,
            "mse": float(mse.detach()),
            "contrastive": float(contrastive.detach()),
        }

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, parts = self.compute_loss(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.log_dict({f"train/{key}": value for key, value in parts.items()})
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        features, target = batch  # [B, subjects, dim], [B, dim]
        self._val_features.append(features.float().cpu())
        self._val_targets.append(target.float().cpu())

    def on_validation_epoch_start(self) -> None:
        self._val_features, self._val_targets = [], []

    def on_validation_epoch_end(self) -> None:
        features = torch.cat(self._val_features)
        targets = torch.cat(self._val_targets)
        subjects = features.shape[1]
        with torch.no_grad():
            preds = [self(features[:, s].to(self.device)).cpu() for s in range(subjects)]
        metrics = [retrieval_metrics(p, targets)[0] for p in preds]
        norms = torch.cat(preds).norm(dim=-1)
        self.log_dict(
            {
                "val/mrr_n1": sum(m["mrr"] for m in metrics) / subjects,
                "val/top1_n1": sum(m["top1"] for m in metrics) / subjects,
                "val/pred_norm_mean": float(norms.mean()),
                "val/pred_norm_std": float(norms.std()),
            },
            prog_bar=True,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.head.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
