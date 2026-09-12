"""Single-person Lightning training of the project's independent EEG models."""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch.nn import functional as F

from ..evaluation.geometry_metrics import geometry_metrics, retrieval_metrics
from ..models.color_specialist import EEGColorSpecialist
from ..models.edge_specialist import EEGEdgeSpecialist
from ..models.semantic_encoder import SubjectAdaptiveSemanticEncoder


class SingleSubjectLightning(L.LightningModule):
    def __init__(
        self,
        task="dino",
        subject_id=0,
        end_ms=800,
        lr=0.0002,
        weight_decay=0.015,
        ema_decay=0.99,
        compact=False,
        protocol_sha256="",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        config = {"subjects": 1, "input_samples": 20 + end_ms // 10, "post_stimulus_start": 20}
        if task in {"dino", "clip_b32", "synclr"}:
            self.model = SubjectAdaptiveSemanticEncoder(
                **config,
                samples=end_ms // 10,
                output_dim={"dino": 384, "clip_b32": 512, "synclr": 768}[task],
            )
        else:
            if compact:
                config.update(dim=96, heads=4, branch_width=12)
                if task == "edge":
                    config.update(temporal_depth=1, spatial_depth=1)
                else:
                    config.update(depth=1)
            decoder = (
                {"input_dim" if task == "edge" else "style_dim": 96, "width": 64} if compact else {}
            )
            cls = EEGEdgeSpecialist if task == "edge" else EEGColorSpecialist
            if task not in {"edge", "color"}:
                raise ValueError(task)
            self.model = cls(encoder_config=config, refiner_config=decoder)
        self.ema = copy.deepcopy(self.model).requires_grad_(False).eval()
        self.validation_outputs = []
        self.concepts = []
        self._resume_rng = None
        self._epoch_started = 0.0

    def initialize_ridge(self, path):
        if self.task not in {"dino", "clip_b32", "synclr"}:
            raise ValueError("ridge initialization is for semantics")
        state = torch.load(path, map_location="cpu", weights_only=False)
        for key, expected in (
            ("target", self.task),
            ("subject_id", self.hparams.subject_id),
            ("end_ms", self.hparams.end_ms),
            ("protocol_sha256", self.hparams.protocol_sha256),
        ):
            if state[key] != expected:
                raise ValueError(f"ridge checkpoint mismatch: {key}")
        with torch.no_grad():
            self.model.linear_weight[0].copy_(state["weight"])
            self.model.linear_bias[0].copy_(state["bias"])
            self.ema.load_state_dict(self.model.state_dict())

    def forward(self, eeg, use_ema=False):
        model = self.ema if use_ema else self.model
        if self.task in {"dino", "clip_b32", "synclr"}:
            return model(eeg, torch.zeros(len(eeg), device=eeg.device, dtype=torch.long))
        return model(eeg[:, None])

    def training_step(self, batch, batch_idx):
        output = self(batch["eeg"])
        if self.task in {"dino", "clip_b32", "synclr"}:
            pred, target = (
                F.normalize(output.float(), dim=-1),
                F.normalize(batch["target"].float(), dim=-1),
            )
            logits = pred @ target.T / 0.07
            labels = torch.arange(len(pred), device=pred.device)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
            loss = loss + 0.5 * (1 - (pred * target).sum(-1).mean())
        elif self.task == "edge":
            from ..models.pyramid_edge import pyramid_edge_loss, pyramid_edge_sample_errors
            from ..models.task_adapter import cvar, shuffled_ranking_loss
            from ..training.train_edge_encoder import diversity_loss, topology_loss

            pyramid, _ = pyramid_edge_loss(output.pyramid, batch)
            errors, _ = pyramid_edge_sample_errors(output.pyramid, batch)
            order = torch.roll(torch.arange(len(errors), device=errors.device), 1)
            wrong = {k: v[order] for k, v in batch.items() if torch.is_tensor(v)}
            wrong_errors, _ = pyramid_edge_sample_errors(output.pyramid, wrong)
            ranking = (
                shuffled_ranking_loss(errors, wrong_errors, 0.10)
                if len(errors) > 1
                else errors.mean() * 0
            )
            loss = (
                pyramid
                + 0.55 * errors.mean()
                + 0.70 * cvar(errors, 0.25)
                + 0.55 * ranking
                + 0.45 * topology_loss(output.pyramid, batch)
                + 0.60 * diversity_loss(output.pyramid, batch)
            )
            # No multi-person-collapse penalty in a genuinely single-person run.
        else:
            from ..models.advanced_color import advanced_color_loss
            from ..models.color import color_descriptor
            from ..training.train_color_encoder import color_sample_error, rare_chroma_loss

            target = batch["rgb"].float()
            reconstruction, _ = advanced_color_loss(output.image, target)
            descriptor = F.smooth_l1_loss(
                output.condition.descriptor, color_descriptor(target), beta=0.03
            )
            loss = (
                reconstruction
                + 0.30 * descriptor
                + 0.45 * color_sample_error(output.image.rgb, target).mean()
            )
            loss = loss + 0.22 * rare_chroma_loss(output.image.chroma_logits16, target)
            # The old local color oracle may have seen the new test images.
            # Direct RGB supervision deliberately avoids this checkpoint.
        self.log("train/loss", loss, on_step=False, on_epoch=True, batch_size=len(batch["eeg"]))
        return loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        with torch.no_grad():
            for target, source in zip(self.ema.parameters(), self.model.parameters()):
                target.lerp_(source, 1 - self.hparams.ema_decay)
            for target, source in zip(self.ema.buffers(), self.model.buffers()):
                target.copy_(source)
        self.ema.eval()

    def on_train_epoch_start(self):
        self.ema.eval()
        self._epoch_started = time.monotonic()

    def on_validation_epoch_start(self):
        self.validation_outputs = []

    def validation_step(self, batch, batch_idx):
        output = self(batch["eeg"], use_ema=True)
        if self.task in {"dino", "clip_b32", "synclr"}:
            pred, target = output, batch["target"]
        elif self.task == "edge":
            pred, target = output.pyramid.shape_logits16.sigmoid(), batch["shape16"]
        else:
            pred, target = output.image.rgb, batch["rgb"]
        self.validation_outputs.append(
            (pred.detach().float().cpu(), target.detach().float().cpu(), batch["row"].cpu())
        )

    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return
        pred, target, rows = [torch.cat([x[i] for x in self.validation_outputs]) for i in range(3)]
        if self.task in {"dino", "clip_b32", "synclr"}:
            metrics = retrieval_metrics(pred, target)
            selection = metrics["mrr"]
        elif self.task == "edge":
            concepts = [self.concepts[int(i)] for i in rows]
            metrics = geometry_metrics(pred, target, concepts)
            selection = (
                (
                    metrics["within_category_2afc"]
                    if metrics["within_category_2afc"] is not None
                    else 0.5
                )
                + 0.25 * metrics["shape_iou"]
                - 0.25 * metrics["centroid_error"]
            )
        else:
            from ..models.color import rgb_to_oklab

            metrics = {
                "rgb_mae": float(F.l1_loss(pred, target)),
                "chroma_mae": float(
                    F.l1_loss(rgb_to_oklab(pred)[:, 1:], rgb_to_oklab(target)[:, 1:])
                ),
            }
            selection = -metrics["chroma_mae"] - 0.25 * metrics["rgb_mae"]
        self.log("val/selection", selection, on_epoch=True, prog_bar=True)
        if not self.trainer.sanity_checking:
            record = {
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "selection": selection,
                "metrics": metrics,
                "epoch_seconds": time.monotonic() - self._epoch_started,
                "peak_cuda_mb": torch.cuda.max_memory_allocated() / 2**20
                if torch.cuda.is_available()
                else 0,
            }
            with (Path(self.trainer.default_root_dir) / "history.jsonl").open(
                "a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        self.validation_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/selection"},
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint["experiment_rng"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        checkpoint["protocol_sha256"] = self.hparams.protocol_sha256
        checkpoint["inference_subjects"] = [self.hparams.subject_id]

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("protocol_sha256") != self.hparams.protocol_sha256:
            raise ValueError("checkpoint belongs to a different data protocol")
        if checkpoint.get("inference_subjects") != [self.hparams.subject_id]:
            raise ValueError("checkpoint belongs to a different participant")
        self._resume_rng = checkpoint.get("experiment_rng")

    def on_fit_start(self):
        if self._resume_rng:
            state = self._resume_rng
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(state["torch"].cpu())
            if state["cuda"] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
            self._resume_rng = None
