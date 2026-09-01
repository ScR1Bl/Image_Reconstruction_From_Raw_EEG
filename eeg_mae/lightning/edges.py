"""Lightning shell for the edges branch.

Inference: holds the ORIGINAL ``EEGEdgeEncoderPipeline`` (specialist + frozen
predictor/decoder) — forward delegates 1:1. Trunk-by-path like the color branch.

Training: ``compute_training_loss`` transcribes the loss block of
``train_edge_encoder.train_epoch`` (train_edge_encoder.py:135-172); helpers are
imported from the original modules. Validated by the one-step loss-parity test.
"""

from __future__ import annotations

import copy

import lightning as L
import torch
from torch.nn import functional as F

from ..data.build_pyramid_edge_targets import PYRAMID_KEYS
from ..models.edge_specialist import EEGEdgeEncoderPipeline, EEGEdgeSpecialist
from ..models.pyramid_edge import (
    PyramidEdgeDecoder,
    pyramid_edge_loss,
    pyramid_edge_sample_errors,
)
from ..models.semantic_edge import EEGDINOGridPredictor
from ..models.semantic_encoder import SubjectAdaptiveSemanticEncoder
from ..models.task_adapter import cvar, improvement_loss, shuffled_ranking_loss
from ..training.train_color_oracle import update_ema
from ..training.train_edge_encoder import (
    diversity_loss,
    residual_energy,
    shuffled_target,
    topology_loss,
)
from .common import freeze, load_converted_branch, load_trunk_state


class EdgesLightning(L.LightningModule):
    def __init__(
        self,
        specialist_config: dict | None = None,
        predictor_config: dict | None = None,
        decoder_config: dict | None = None,
        trunk_config: dict | None = None,
        trunk_checkpoint: str | None = None,
        trunk_sha256: str | None = None,
        lr: float = 2e-4,
        weight_decay: float = 0.015,
        ema_decay: float = 0.998,
        margin: float = 0.018,
        rank_margin: float = 0.10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        trunk = SubjectAdaptiveSemanticEncoder(**(trunk_config or {}))
        if trunk_checkpoint:
            trunk.load_state_dict(load_trunk_state(trunk_checkpoint, trunk_sha256), strict=True)
        specialist = EEGEdgeSpecialist(**(specialist_config or {}))
        predictor = EEGDINOGridPredictor(trunk, **(predictor_config or {}))
        decoder = PyramidEdgeDecoder(**(decoder_config or {}))
        self.pipeline = EEGEdgeEncoderPipeline(specialist, predictor, decoder)
        for frozen in (predictor, decoder):
            freeze(frozen)
        self.ema_specialist = freeze(copy.deepcopy(specialist))

    @classmethod
    def load_converted(cls, path, map_location: str = "cpu") -> "EdgesLightning":
        return load_converted_branch(cls, path, map_location)

    @property
    def specialist(self) -> EEGEdgeSpecialist:
        return self.pipeline.specialist

    def forward(self, eeg: torch.Tensor):
        return self.pipeline(eeg)

    @torch.no_grad()
    def _frozen_baseline(self, eeg: torch.Tensor):
        # FrozenEdgeBaseline.__call__ (train_edge_encoder.py:87-90), verbatim.
        grid, _ = self.pipeline.predictor.forward_ensemble(eeg)
        return self.pipeline.decoder(grid)

    def compute_training_loss(self, batch) -> tuple[torch.Tensor, dict]:
        # Transcription of train_edge_encoder.py:135-172 (train_epoch loss block).
        eeg = batch["eeg"]
        target = {key: batch[key] for key in PYRAMID_KEYS}
        with torch.no_grad():
            base = self._frozen_baseline(eeg)
            base_error, _ = pyramid_edge_sample_errors(base, target)
        result = self.specialist(eeg, base)
        output = result.pyramid
        pyramid, parts = pyramid_edge_loss(output, target)
        errors, sample_parts = pyramid_edge_sample_errors(output, target)
        task, hard = errors.mean(), cvar(errors, 0.25)
        improve = improvement_loss(errors, base_error, self.hparams.margin)
        order = torch.roll(torch.arange(len(eeg), device=eeg.device), 1)
        wrong_errors, _ = pyramid_edge_sample_errors(output, shuffled_target(target, order))
        ranking = (
            shuffled_ranking_loss(errors, wrong_errors, self.hparams.rank_margin)
            if len(eeg) > 1
            else task * 0
        )
        topology = topology_loss(output, target)
        diversity = diversity_loss(output, target)
        residual = residual_energy(output, base)
        weights = result.condition.subject_weights.clamp_min(1e-8)
        effective_subjects = (-weights * weights.log()).sum(1).exp()
        subject_collapse = F.relu(3 - effective_subjects).mean() / 3
        loss = (
            pyramid
            + 0.55 * task
            + 0.70 * hard
            + 1.7 * improve
            + 0.55 * ranking
            + 0.45 * topology
            + 0.60 * diversity
            + 0.002 * residual
            + 0.05 * subject_collapse
        )
        components = {
            "loss": loss,
            "pyramid": pyramid,
            "task": task,
            "hard": hard,
            "improvement": improve,
            "ranking": ranking,
            "topology": topology,
            "diversity": diversity,
            "residual_energy": residual,
            "effective_subjects": effective_subjects.mean(),
        }
        return loss, components

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, components = self.compute_training_loss(batch)
        self.log_dict({f"train/{key}": value for key, value in components.items()}, prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        update_ema(self.ema_specialist, self.specialist, self.hparams.ema_decay)

    def on_fit_start(self) -> None:
        # A converted checkpoint arrives fully frozen (inference parity with the
        # legacy loaders); training explicitly re-enables the trainable part.
        for parameter in self.specialist.parameters():
            parameter.requires_grad_(True)

    def on_train_epoch_start(self) -> None:
        for frozen in (self.pipeline.predictor, self.pipeline.decoder):
            frozen.eval()

    def validation_step(self, batch, batch_idx: int) -> None:
        loss, components = self.compute_training_loss(batch)
        self.log_dict({f"val/{key}": value for key, value in components.items()}, prog_bar=True)

    def configure_optimizers(self):
        # AdamW z betas=(0.9, 0.95) jak w train_edge_encoder.py:399-401.
        return torch.optim.AdamW(
            self.specialist.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95),
        )
