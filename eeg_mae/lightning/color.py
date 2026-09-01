"""Lightning shell for the color branch.

Inference: holds the ORIGINAL ``EEGColorEncoderPipeline`` (specialist +
frozen predictor/adapter/decoder) — forward delegates 1:1.

Trunk-by-path: the shared semantic trunk is NOT stored in this branch's
checkpoint; it is loaded from the single trunk artifact named in the
hyperparameters (``trunk_checkpoint`` + ``trunk_sha256``).

Training: ``compute_training_loss`` is a transcription of the loss block of
``train_color_encoder.train_epoch`` (train_color_encoder.py:138-203), with every
helper imported from the original modules. Validated by the one-step loss-parity
test; full training-dynamics parity is a Step 1b concern.
"""

from __future__ import annotations

import copy

import lightning as L
import torch
from torch.nn import functional as F

from ..models.advanced_color import FullColorLatent, advanced_color_loss
from ..models.color import ColorDecoder, color_descriptor, rendered_descriptor_contrastive
from ..models.color_specialist import EEGColorEncoderPipeline, EEGColorSpecialist
from ..models.semantic_edge import EEGDINOGridPredictor
from ..models.semantic_encoder import SubjectAdaptiveSemanticEncoder
from ..models.task_adapter import TaskResidualAdapter, cvar, improvement_loss, shuffled_ranking_loss
from ..training.color_training import load_color_oracle, semantic_ensemble
from ..training.train_color_encoder import (
    color_sample_error,
    rare_chroma_loss,
    residual_smoothness,
)
from ..training.train_color_oracle import update_ema
from .common import freeze, load_converted_branch, load_trunk_state


class ColorLightning(L.LightningModule):
    def __init__(
        self,
        specialist_config: dict | None = None,
        predictor_config: dict | None = None,
        adapter_config: dict | None = None,
        decoder_config: dict | None = None,
        trunk_config: dict | None = None,
        trunk_checkpoint: str | None = None,
        trunk_sha256: str | None = None,
        oracle_checkpoint: str | None = None,
        lr: float = 2e-4,
        weight_decay: float = 0.015,
        ema_decay: float = 0.998,
        margin: float = 0.008,
        rank_margin: float = 0.035,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        trunk = SubjectAdaptiveSemanticEncoder(**(trunk_config or {}))
        if trunk_checkpoint:
            trunk.load_state_dict(load_trunk_state(trunk_checkpoint, trunk_sha256), strict=True)
        specialist = EEGColorSpecialist(**(specialist_config or {}))
        predictor = EEGDINOGridPredictor(trunk, **(predictor_config or {}))
        adapter = TaskResidualAdapter(**(adapter_config or {}))
        decoder = ColorDecoder(**(decoder_config or {}))
        self.pipeline = EEGColorEncoderPipeline(specialist, predictor, adapter, decoder)
        for frozen in (predictor, adapter, decoder):
            freeze(frozen)
        self.ema_specialist = freeze(copy.deepcopy(specialist))
        # The oracle is a training-only teacher (87 MB) — kept OUT of the module
        # graph so it never enters checkpoints; loaded lazily from its path.
        self._oracle_container: list = []

    @classmethod
    def load_converted(cls, path, map_location: str = "cpu") -> "ColorLightning":
        return load_converted_branch(cls, path, map_location)

    @property
    def specialist(self) -> EEGColorSpecialist:
        return self.pipeline.specialist

    @property
    def oracle(self):
        if not self._oracle_container:
            if not self.hparams.oracle_checkpoint:
                raise RuntimeError("training requires oracle_checkpoint in hyperparameters")
            oracle, _ = load_color_oracle(self.hparams.oracle_checkpoint, self.device)
            self._oracle_container.append(oracle)
        return self._oracle_container[0]

    def forward(self, eeg: torch.Tensor, hard_chroma: bool = False):
        return self.pipeline(eeg, hard_chroma)

    @torch.no_grad()
    def _frozen_baseline32(self, eeg: torch.Tensor) -> torch.Tensor:
        # FrozenColorBaseline.__call__ (train_color_encoder.py:97-102), verbatim.
        grid, contexts = self.pipeline.predictor.forward_ensemble(eeg)
        global_feature = semantic_ensemble(self.pipeline.predictor, eeg)
        adapted = self.pipeline.adapter(grid, contexts, global_feature)
        return self.pipeline.decoder(adapted.grid, adapted.global_feature).rgb32.float()

    def compute_training_loss(self, batch) -> tuple[torch.Tensor, dict]:
        # Transcription of train_color_encoder.py:138-203 (train_epoch loss block).
        eeg, target = batch["eeg"], batch["rgb"]
        true_latent = FullColorLatent(batch["grid"].float(), batch["global"].float())
        with torch.no_grad():
            base32 = self._frozen_baseline32(eeg)
            base = F.interpolate(base32, target.shape[-2:], mode="bilinear", align_corners=False)
            teacher = self.oracle.decode_full_latent(true_latent)
            base_error = color_sample_error(base, target)
        output = self.specialist(eeg, base32)
        reconstruction, parts = advanced_color_loss(output.image, target)
        oracle_chroma = F.smooth_l1_loss(
            output.image.oklab[:, 1:], teacher.oklab[:, 1:], beta=0.012
        )
        oracle_palette = F.smooth_l1_loss(
            F.adaptive_avg_pool2d(output.image.rgb, 4),
            F.adaptive_avg_pool2d(teacher.rgb, 4),
            beta=0.025,
        )
        descriptor_target = color_descriptor(target)
        descriptor = F.smooth_l1_loss(output.condition.descriptor, descriptor_target, beta=0.03)
        contrastive = (
            rendered_descriptor_contrastive(output.image.rgb, descriptor_target)
            if len(eeg) > 1
            else reconstruction * 0
        )
        error = color_sample_error(output.image.rgb, target)
        task = error.mean()
        hard = cvar(error)
        improve = improvement_loss(error, base_error, self.hparams.margin)
        order = torch.roll(torch.arange(len(target), device=target.device), 1)
        wrong_error = color_sample_error(output.image.rgb, target[order])
        ranking = (
            shuffled_ranking_loss(error, wrong_error, self.hparams.rank_margin)
            if len(eeg) > 1
            else task * 0
        )
        rare = rare_chroma_loss(output.image.chroma_logits16, target)
        target_std = target.flatten(1).std(0, unbiased=False).mean()
        output_std = output.image.rgb.flatten(1).std(0, unbiased=False).mean()
        diversity = F.relu(0.72 * target_std - output_std)
        smooth = residual_smoothness(output.chroma_delta16)
        delta = output.chroma_delta16.square().mean()
        alpha_open = F.relu(0.06 - output.alpha16.mean())
        loss = (
            reconstruction
            + 0.55 * oracle_chroma
            + 0.30 * oracle_palette
            + 0.30 * descriptor
            + 0.15 * contrastive
            + 0.45 * task
            + 0.65 * hard
            + 1.6 * improve
            + 0.40 * ranking
            + 0.22 * rare
            + 0.40 * diversity
            + 0.04 * smooth
            + 0.004 * delta
            + 0.08 * alpha_open
        )
        components = {
            "loss": loss,
            "task": task,
            "hard": hard,
            "improvement": improve,
            "ranking": ranking,
            "oracle_chroma": oracle_chroma,
            "oracle_palette": oracle_palette,
            "descriptor_direct": descriptor,
        }
        return loss, components

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        # NOTE: the specialist trains while the rest of the pipeline stays frozen;
        # dropout etc. follow module train/eval flags exactly like the old loop.
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
        # Frozen components must never flip into train mode.
        for frozen in (self.pipeline.predictor, self.pipeline.adapter, self.pipeline.decoder):
            frozen.eval()

    def validation_step(self, batch, batch_idx: int) -> None:
        loss, components = self.compute_training_loss(batch)
        self.log_dict({f"val/{key}": value for key, value in components.items()}, prog_bar=True)

    def configure_optimizers(self):
        # AdamW z betas=(0.9, 0.95) jak w train_color_encoder.py:414-416.
        return torch.optim.AdamW(
            self.specialist.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95),
        )
