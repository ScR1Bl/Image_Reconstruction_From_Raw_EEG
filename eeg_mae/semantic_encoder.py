from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class SubjectAdaptiveSemanticEncoder(nn.Module):
    """EEG -> visual-semantic embedding with strong subject-specific anchors.

    The subject-specific linear path is deliberately kept as an exact superset of
    the decisive linear baseline.  Shared temporal/spatial and semantic residuals
    can improve it, while zero initialization makes the initial model identical
    to the already validated baseline rather than a random decoder.
    """

    def __init__(
        self,
        output_dim: int = 384,
        subjects: int = 10,
        channels: int = 17,
        samples: int = 80,
        input_samples: int = 100,
        post_stimulus_start: int = 20,
        temporal_features: int = 48,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if input_samples - post_stimulus_start != samples:
            raise ValueError("input_samples - post_stimulus_start must equal samples")
        self.output_dim = output_dim
        self.subjects = subjects
        self.channels = channels
        self.samples = samples
        self.input_samples = input_samples
        self.post_stimulus_start = post_stimulus_start
        flattened = channels * samples

        # A batched bank of independent LayerNorm + Linear decoders.  This is the
        # part that preserves the proven per-subject performance.
        self.input_norm_weight = nn.Parameter(torch.ones(subjects, flattened))
        self.input_norm_bias = nn.Parameter(torch.zeros(subjects, flattened))
        self.linear_weight = nn.Parameter(torch.empty(subjects, output_dim, flattened))
        self.linear_bias = nn.Parameter(torch.zeros(subjects, output_dim))
        for weight in self.linear_weight:
            nn.init.xavier_uniform_(weight)

        # Subject-wise channel calibration before a shared EEGNet/NICE-like path.
        self.channel_scale = nn.Parameter(torch.ones(subjects, channels))
        self.channel_bias = nn.Parameter(torch.zeros(subjects, channels))
        self.temporal_spatial = nn.Sequential(
            nn.Conv2d(1, temporal_features, (1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(temporal_features),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Conv2d(
                temporal_features,
                temporal_features,
                (channels, 1),
                groups=temporal_features,
                bias=False,
            ),
            nn.BatchNorm2d(temporal_features),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Flatten(),
        )
        feature_dim = temporal_features * (samples // 4)
        self.temporal_project = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, output_dim),
        )
        self.semantic_residual = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
        )
        self.subject_residual_scale = nn.Parameter(torch.ones(subjects, output_dim))
        self.subject_residual_bias = nn.Parameter(torch.zeros(subjects, output_dim))
        self.residual_gate = nn.Parameter(torch.full((subjects, 1), -2.0))

        # Exact baseline at initialization. Gradients first open the output layers
        # and subsequently train the preceding shared feature extractors.
        nn.init.zeros_(self.temporal_project[-1].weight)
        nn.init.zeros_(self.temporal_project[-1].bias)
        nn.init.zeros_(self.semantic_residual[-1].weight)
        nn.init.zeros_(self.semantic_residual[-1].bias)

    def crop(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3 or eeg.shape[1] != self.channels:
            raise ValueError(f"expected EEG [batch,{self.channels},time], got {tuple(eeg.shape)}")
        if eeg.shape[2] == self.input_samples:
            return eeg[:, :, self.post_stimulus_start :]
        if eeg.shape[2] == self.samples:
            return eeg
        raise ValueError(
            f"expected {self.input_samples} or {self.samples} samples, got {eeg.shape[2]}"
        )

    def base_parameters(self) -> list[nn.Parameter]:
        return [self.input_norm_weight, self.input_norm_bias, self.linear_weight, self.linear_bias]

    def residual_parameters(self) -> list[nn.Parameter]:
        base_ids = {id(value) for value in self.base_parameters()}
        return [value for value in self.parameters() if id(value) not in base_ids]

    def forward(self, eeg: torch.Tensor, subject: torch.Tensor) -> torch.Tensor:
        eeg = self.crop(eeg)
        if subject.ndim != 1 or len(subject) != len(eeg):
            raise ValueError("subject must be a [batch] vector")
        if subject.numel() and (int(subject.min()) < 0 or int(subject.max()) >= self.subjects):
            raise ValueError(f"subject IDs must be in 0..{self.subjects - 1}")

        flattened = eeg.flatten(1)
        normalized = F.layer_norm(flattened, (flattened.shape[1],))
        # Do not advanced-index the 3-D weight bank per example: that would
        # replicate a 384x1360 matrix hundreds of times. Batches are normally
        # subject-homogeneous, and this grouped form also supports mixed batches.
        base = None
        for sid in subject.unique(sorted=True):
            indices = torch.where(subject == sid)[0]
            value = normalized.index_select(0, indices)
            value = value * self.input_norm_weight[sid] + self.input_norm_bias[sid]
            value = F.linear(value, self.linear_weight[sid], self.linear_bias[sid])
            if base is None:
                base = value.new_zeros((len(normalized), self.output_dim))
            base = base.index_copy(0, indices, value)
        if base is None:
            return normalized.new_empty((0, self.output_dim))

        calibrated = (
            eeg * self.channel_scale[subject, :, None] + self.channel_bias[subject, :, None]
        )
        temporal = self.temporal_project(self.temporal_spatial(calibrated[:, None]))
        residual = temporal + self.semantic_residual(base)
        residual = (
            residual * self.subject_residual_scale[subject] + self.subject_residual_bias[subject]
        )
        return base + torch.sigmoid(self.residual_gate[subject]) * residual

    def load_linear_baselines(self, directory: str | Path) -> None:
        """Load the ten official-benchmark LinearEEG checkpoints exactly."""
        directory = Path(directory)
        with torch.no_grad():
            for subject in range(self.subjects):
                path = directory / f"linear_subject_{subject + 1:02d}.pt"
                if not path.exists():
                    raise FileNotFoundError(f"missing baseline checkpoint: {path}")
                payload = torch.load(path, map_location="cpu", weights_only=False)
                state = payload["model"]
                expected = {
                    "norm.weight": self.input_norm_weight[subject].shape,
                    "norm.bias": self.input_norm_bias[subject].shape,
                    "project.weight": self.linear_weight[subject].shape,
                    "project.bias": self.linear_bias[subject].shape,
                }
                for name, shape in expected.items():
                    if name not in state or state[name].shape != shape:
                        raise RuntimeError(
                            f"incompatible {path}: {name} has "
                            f"{state.get(name, torch.empty(0)).shape}, expected {shape}"
                        )
                self.input_norm_weight[subject].copy_(state["norm.weight"])
                self.input_norm_bias[subject].copy_(state["norm.bias"])
                self.linear_weight[subject].copy_(state["project.weight"])
                self.linear_bias[subject].copy_(state["project.bias"])

    def config(self) -> dict:
        return {
            "output_dim": self.output_dim,
            "subjects": self.subjects,
            "channels": self.channels,
            "samples": self.samples,
            "input_samples": self.input_samples,
            "post_stimulus_start": self.post_stimulus_start,
            "temporal_features": self.temporal_spatial[0].out_channels,
            "dropout": self.temporal_spatial[-2].p,
        }


def load_semantic_encoder(
    path: str | Path, device: torch.device | str = "cpu"
) -> SubjectAdaptiveSemanticEncoder:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = SubjectAdaptiveSemanticEncoder(**payload["model_config"])
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device)
