"""Ensemble trunk (Step 1b): existing SubjectAdaptiveSemanticEncoder run
per-subject + permutation-invariant fusion, trained with a random subject-count
N per batch (P(N=1)=n1_prob, remainder uniform over 2..n_max).

Validation follows the GATE definition exactly: fusion on RAW embeddings,
normalization inside the metric (amendment #1, 2026-09-01). The monitored
metric is val/mrr_n1 — N=1 is the primary result of this step.
"""

from __future__ import annotations

import lightning as L
import torch

from ..evaluation.official_200way_benchmark import retrieval_metrics
from ..models.semantic_encoder import SubjectAdaptiveSemanticEncoder
from ..training.train_semantic_encoder import semantic_loss
from .common import load_trunk_state
from .subject_fusion import build_fusion


class EnsembleTrunkLightning(L.LightningModule):
    def __init__(
        self,
        trunk_config: dict | None = None,
        trunk_checkpoint: str | None = None,
        trunk_sha256: str | None = None,
        fusion: str = "mean",
        n1_prob: float = 0.25,
        n_max: int = 10,
        freeze_trunk: bool = False,
        base_lr: float = 1e-5,
        residual_lr: float = 1e-4,
        fusion_lr: float = 3e-4,
        weight_decay: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.trunk = SubjectAdaptiveSemanticEncoder(**(trunk_config or {}))
        if trunk_checkpoint:
            self.trunk.load_state_dict(
                load_trunk_state(trunk_checkpoint, trunk_sha256), strict=True
            )
        self.fusion = build_fusion(fusion, dim=self.trunk.output_dim, roster=self.trunk.subjects)
        if freeze_trunk:
            # Zamrozony trunk = N=1 bitowo rowne baseline'owi; uczy sie tylko
            # fuzja. Trunk musi tez zostac w eval() (BatchNorm/dropout) - patrz
            # on_train_epoch_start.
            for parameter in self.trunk.parameters():
                parameter.requires_grad_(False)
            self.trunk.eval()

    def on_train_epoch_start(self) -> None:
        if self.hparams.freeze_trunk:
            self.trunk.eval()

    def embed_subjects(self, eeg: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        """Per-subject RAW embeddings for a stacked ensemble [batch, n, ch, time]."""
        batch, count = eeg.shape[:2]
        ids = torch.as_tensor(subject_ids, device=eeg.device)
        flat_ids = ids.repeat(batch)
        embeddings = self.trunk(eeg.flatten(0, 1), flat_ids)
        return embeddings.reshape(batch, count, -1)

    def forward(self, eeg: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embed_subjects(eeg, subject_ids)
        fused, _ = self.fusion(embeddings, torch.as_tensor(subject_ids, device=eeg.device))
        return fused

    def sample_subset(self) -> torch.Tensor:
        """N per batch: P(N=1)=n1_prob, remainder uniform over [2, n_max]."""
        if torch.rand(()) < self.hparams.n1_prob:
            count = 1
        else:
            count = int(torch.randint(2, self.hparams.n_max + 1, ()).item())
        return torch.randperm(self.hparams.n_max)[:count]

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        eeg_full, target = batch
        subset = self.sample_subset().to(eeg_full.device)
        fused = self(eeg_full[:, subset], subset)
        loss, parts = semantic_loss(fused, target)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/n", float(len(subset)))
        self.log_dict({f"train/{key}": value for key, value in parts.items()})
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        eeg_full, target = batch
        roster = torch.arange(self.hparams.n_max, device=eeg_full.device)
        embeddings = self.embed_subjects(eeg_full, roster)
        self._val_embeddings.append(embeddings.float().cpu())
        self._val_targets.append(target.float().cpu())

    def on_validation_epoch_start(self) -> None:
        self._val_embeddings, self._val_targets = [], []

    def on_validation_epoch_end(self) -> None:
        embeddings = torch.cat(self._val_embeddings)  # [V, roster, dim]
        targets = torch.cat(self._val_targets)
        roster = torch.arange(embeddings.shape[1])
        # N=1: srednia metryk po pojedynczych podmiotach (wynik glowny kroku).
        singles = [retrieval_metrics(embeddings[:, s], targets)[0] for s in range(len(roster))]
        mrr_n1 = sum(m["mrr"] for m in singles) / len(singles)
        top1_n1 = sum(m["top1"] for m in singles) / len(singles)
        # N=10: fuzja na SUROWYCH embeddingach, normalizacja w metryce (bramka).
        with torch.no_grad():
            fused, _ = self.fusion.cpu()(embeddings, roster)
        self.fusion.to(self.device)
        full = retrieval_metrics(fused, targets)[0]
        self.log_dict(
            {
                "val/mrr_n1": mrr_n1,
                "val/top1_n1": top1_n1,
                "val/mrr_n10": full["mrr"],
                "val/top1_n10": full["top1"],
            },
            prog_bar=True,
        )

    def configure_optimizers(self):
        groups = []
        if not self.hparams.freeze_trunk:
            groups.append({"params": self.trunk.base_parameters(), "lr": self.hparams.base_lr})
            groups.append(
                {"params": self.trunk.residual_parameters(), "lr": self.hparams.residual_lr}
            )
        fusion_params = list(self.fusion.parameters())
        if fusion_params:
            groups.append({"params": fusion_params, "lr": self.hparams.fusion_lr})
        if not groups:
            raise RuntimeError("freeze_trunk z fuzja bez parametrow: nie ma czego trenowac")
        return torch.optim.AdamW(groups, weight_decay=self.hparams.weight_decay)
