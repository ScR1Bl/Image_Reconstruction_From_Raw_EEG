"""Independent raw-EEG structure encoder and stable multi-scale edge refiner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .model_utils import CrossAttentionBlock
from .pyramid_edge import PyramidEdgeOutput, ResidualConv


@dataclass
class StructureCondition:
    grid16: torch.Tensor
    subject_weights: torch.Tensor


@dataclass
class EdgeSpecialistOutput:
    pyramid: PyramidEdgeOutput
    condition: StructureCondition
    gates: dict[str, torch.Tensor]


class MultiscaleEEGStructureEncoder(nn.Module):
    """EEGNet/Conformer-style branch trained only for stimulus geometry."""

    def __init__(
        self,
        subjects: int = 10,
        channels: int = 17,
        input_samples: int = 100,
        post_stimulus_start: int = 20,
        branch_width: int = 24,
        dim: int = 192,
        heads: int = 6,
        temporal_depth: int = 2,
        spatial_depth: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.subjects, self.channels, self.input_samples = subjects, channels, input_samples
        self.post_stimulus_start, self.branch_width = post_stimulus_start, branch_width
        self.dim, self.heads = dim, heads
        self.temporal_depth, self.spatial_depth, self.dropout = (
            temporal_depth,
            spatial_depth,
            dropout,
        )
        self.subject_scale = nn.Parameter(torch.ones(subjects, channels))
        self.subject_bias = nn.Parameter(torch.zeros(subjects, channels))
        self.branches = nn.ModuleList([self._branch(kernel) for kernel in (7, 15, 31)])
        merged = 3 * branch_width
        self.temporal_projection = nn.Sequential(nn.LayerNorm(merged), nn.Linear(merged, dim))
        time_tokens = (input_samples - post_stimulus_start) // 4
        self.temporal_position = nn.Parameter(torch.randn(1, time_tokens, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.temporal_mixer = nn.TransformerEncoder(
            layer, temporal_depth, enable_nested_tensor=False
        )
        self.subject_quality = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1)
        )
        nn.init.zeros_(self.subject_quality[-1].weight)
        nn.init.zeros_(self.subject_quality[-1].bias)
        self.subject_prior = nn.Parameter(torch.zeros(subjects))
        self.spatial_queries = nn.Parameter(torch.randn(1, 256, dim) * 0.02)
        self.cross = nn.ModuleList(
            [CrossAttentionBlock(dim, dim, heads, dropout) for _ in range(2)]
        )
        spatial_layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.spatial_mixer = nn.TransformerEncoder(
            spatial_layer, spatial_depth, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(dim)

    def _branch(self, kernel: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(1, self.branch_width, (1, kernel), padding=(0, kernel // 2), bias=False),
            nn.BatchNorm2d(self.branch_width),
            nn.GELU(),
            nn.Conv2d(
                self.branch_width,
                self.branch_width,
                (self.channels, 1),
                groups=self.branch_width,
                bias=False,
            ),
            nn.BatchNorm2d(self.branch_width),
            nn.GELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout),
        )

    def forward(self, eeg: torch.Tensor) -> StructureCondition:
        if eeg.ndim != 4 or eeg.shape[1:3] != (self.subjects, self.channels):
            raise ValueError(f"expected EEG [B,{self.subjects},{self.channels},time]")
        if eeg.shape[-1] != self.input_samples:
            raise ValueError("unexpected EEG time dimension")
        batch = len(eeg)
        value = eeg[..., self.post_stimulus_start :].float()
        subjects = torch.arange(self.subjects, device=eeg.device).repeat(batch)
        value = value.flatten(0, 1)
        value = value * self.subject_scale[subjects, :, None] + self.subject_bias[subjects, :, None]
        temporal = torch.cat([branch(value[:, None]).squeeze(2) for branch in self.branches], 1)
        temporal = self.temporal_projection(temporal.transpose(1, 2)) + self.temporal_position
        temporal = self.temporal_mixer(temporal).reshape(
            batch, self.subjects, temporal.shape[1], self.dim
        )
        summaries = temporal.mean(2)
        weights = (self.subject_quality(summaries).squeeze(-1) + self.subject_prior).softmax(1)
        fused = torch.einsum("bs,bstd->btd", weights, temporal)
        spatial = self.spatial_queries.expand(batch, -1, -1)
        for block in self.cross:
            spatial = block(spatial, fused)
        spatial = self.output_norm(self.spatial_mixer(spatial))
        return StructureCondition(spatial.reshape(batch, 16, 16, self.dim), weights)

    def config(self) -> dict:
        return {
            "subjects": self.subjects,
            "channels": self.channels,
            "input_samples": self.input_samples,
            "post_stimulus_start": self.post_stimulus_start,
            "branch_width": self.branch_width,
            "dim": self.dim,
            "heads": self.heads,
            "temporal_depth": self.temporal_depth,
            "spatial_depth": self.spatial_depth,
            "dropout": self.dropout,
        }


def _logit(probability: torch.Tensor) -> torch.Tensor:
    return torch.logit(probability.float().clamp(1e-4, 1 - 1e-4))


class StableEdgeRefiner(nn.Module):
    """Predict gated residuals around a frozen EEG edge baseline without deconvolution."""

    def __init__(self, input_dim: int = 192, width: int = 128) -> None:
        super().__init__()
        self.input_dim, self.width = input_dim, width
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim + 2, width, 1), ResidualConv(width), ResidualConv(width)
        )
        self.head16 = nn.Conv2d(width, 2, 3, padding=1)
        self.up32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(width, width, 3, padding=1),
            ResidualConv(width),
        )
        self.head32 = nn.Conv2d(width, 1, 3, padding=1)
        self.up64 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(width, width // 2, 3, padding=1),
            ResidualConv(width // 2),
            ResidualConv(width // 2),
        )
        self.head64 = nn.Conv2d(width // 2, 6, 3, padding=1)
        for head in (self.head16, self.head32, self.head64):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.gate16 = nn.Parameter(torch.tensor(-2.5))
        self.gate32 = nn.Parameter(torch.tensor(-2.5))
        self.gate64 = nn.Parameter(torch.tensor(-2.5))

    def forward(
        self, condition: StructureCondition, baseline: PyramidEdgeOutput
    ) -> EdgeSpecialistOutput:
        grid = condition.grid16.permute(0, 3, 1, 2).float()
        base16 = torch.cat((baseline.shape_logits16.sigmoid(), baseline.distance16), 1)
        level16 = self.stem(torch.cat((grid, base16), 1))
        delta16 = self.head16(level16)
        gate16 = torch.sigmoid(self.gate16)
        shape = baseline.shape_logits16 + gate16 * delta16[:, :1]
        distance = (baseline.distance16 + 0.25 * gate16 * torch.tanh(delta16[:, 1:2])).clamp(0, 1)
        level32 = self.up32(level16)
        delta32 = self.head32(level32)
        gate32 = torch.sigmoid(self.gate32)
        semantic = baseline.semantic_logits32 + gate32 * delta32
        level64 = self.up64(level32)
        delta64 = self.head64(level64)
        gate64 = torch.sigmoid(self.gate64)
        perceptual = baseline.perceptual_logits64 + gate64 * delta64[:, :1]
        crisp = baseline.crisp_logits64 + gate64 * delta64[:, 1:2]
        confidence = torch.sigmoid(_logit(baseline.confidence64) + gate64 * delta64[:, 2:3])
        uncertainty = torch.sigmoid(_logit(baseline.uncertainty64) + gate64 * delta64[:, 3:4])
        orientation = F.normalize(
            baseline.orientation64 + 0.25 * gate64 * torch.tanh(delta64[:, 4:6]), dim=1, eps=1e-5
        )
        pyramid = PyramidEdgeOutput(
            shape, distance, semantic, perceptual, crisp, confidence, uncertainty, orientation
        )
        gates = {"level16": gate16, "level32": gate32, "level64": gate64}
        return EdgeSpecialistOutput(pyramid, condition, gates)

    def config(self) -> dict:
        return {"input_dim": self.input_dim, "width": self.width}


class EEGEdgeSpecialist(nn.Module):
    def __init__(
        self, encoder_config: dict | None = None, refiner_config: dict | None = None
    ) -> None:
        super().__init__()
        self.encoder = MultiscaleEEGStructureEncoder(**(encoder_config or {}))
        self.refiner = StableEdgeRefiner(**(refiner_config or {}))

    def forward(self, eeg: torch.Tensor, baseline: PyramidEdgeOutput) -> EdgeSpecialistOutput:
        return self.refiner(self.encoder(eeg), baseline)

    def config(self) -> dict:
        return {"encoder_config": self.encoder.config(), "refiner_config": self.refiner.config()}


class EEGEdgeEncoderPipeline(nn.Module):
    """Self-contained inference pipeline loaded from ``runs/eeg_edge_encoder``."""

    def __init__(self, specialist, predictor, decoder) -> None:
        super().__init__()
        self.specialist = specialist
        self.predictor = predictor
        self.decoder = decoder

    def forward(self, eeg: torch.Tensor) -> EdgeSpecialistOutput:
        grid, _ = self.predictor.forward_ensemble(eeg)
        baseline = self.decoder(grid)
        return self.specialist(eeg, baseline)


def load_eeg_edge_encoder(
    root: str | Path = "runs/eeg_edge_encoder", device: torch.device | str = "cpu"
) -> tuple[EEGEdgeEncoderPipeline, dict]:
    from .semantic_edge import EEGDINOGridPredictor
    from .semantic_encoder import SubjectAdaptiveSemanticEncoder

    root = Path(root)
    components = root / "components"
    state = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
    specialist = EEGEdgeSpecialist(**state["model_config"])
    specialist.load_state_dict(state.get("ema", state["model"]), strict=True)
    semantic_state = torch.load(
        components / "semantic_encoder.pt", map_location="cpu", weights_only=False
    )
    semantic = SubjectAdaptiveSemanticEncoder(**semantic_state["model_config"])
    semantic.load_state_dict(semantic_state["model"], strict=True)
    baseline_state = torch.load(components / "baseline.pt", map_location="cpu", weights_only=False)
    predictor = EEGDINOGridPredictor(semantic, **baseline_state["model_config"])
    predictor.load_state_dict(baseline_state["model"], strict=True)
    from .pyramid_edge import PyramidEdgeDecoder

    decoder_state = torch.load(
        components / "edge_decoder.pt", map_location="cpu", weights_only=False
    )
    decoder = PyramidEdgeDecoder(**decoder_state["model_config"])
    decoder.load_state_dict(decoder_state["model"], strict=True)
    pipeline = EEGEdgeEncoderPipeline(specialist, predictor, decoder).to(device).eval()
    for parameter in pipeline.parameters():
        parameter.requires_grad_(False)
    return pipeline, state
