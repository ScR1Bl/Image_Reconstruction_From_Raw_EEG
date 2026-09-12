"""Independent raw-EEG color encoder with a direct OKLab renderer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .advanced_color import (
    AdvancedColorOutput,
    ConvResidual,
    linear_rgb_to_srgb,
    oklab_to_linear_rgb,
)
from .color import color_descriptor
from .model_utils import CrossAttentionBlock


@dataclass
class ColorEEGCondition:
    style_tokens: torch.Tensor
    grid16: torch.Tensor
    descriptor: torch.Tensor
    subject_weights: torch.Tensor


@dataclass
class ColorSpecialistOutput:
    image: AdvancedColorOutput
    condition: ColorEEGCondition
    color_grid16: torch.Tensor


class MultiscaleEEGColorEncoder(nn.Module):
    """Independent EEGNet/Conformer-style encoder dedicated to color and style."""

    def __init__(
        self,
        subjects: int = 10,
        channels: int = 17,
        input_samples: int = 100,
        post_stimulus_start: int = 20,
        branch_width: int = 24,
        dim: int = 192,
        heads: int = 6,
        depth: int = 2,
        style_tokens: int = 16,
        descriptor_dim: int = 63,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.subjects, self.channels, self.input_samples = subjects, channels, input_samples
        self.post_stimulus_start, self.branch_width = post_stimulus_start, branch_width
        self.dim, self.heads, self.depth, self.style_tokens = dim, heads, depth, style_tokens
        self.descriptor_dim, self.dropout = descriptor_dim, dropout
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
        self.temporal_mixer = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.subject_quality = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1)
        )
        nn.init.zeros_(self.subject_quality[-1].weight)
        nn.init.zeros_(self.subject_quality[-1].bias)
        self.subject_prior = nn.Parameter(torch.zeros(subjects))
        self.style_queries = nn.Parameter(torch.randn(1, style_tokens, dim) * 0.02)
        self.style_cross = nn.ModuleList(
            [CrossAttentionBlock(dim, dim, heads, dropout) for _ in range(2)]
        )
        self.style_norm = nn.LayerNorm(dim)
        # Perceiver-IO-style learned output queries.  These are color-layout
        # tokens, not DINO tokens copied from the semantic encoder.
        self.spatial_queries = nn.Parameter(torch.randn(1, 256, dim) * 0.02)
        self.spatial_cross = nn.ModuleList(
            [CrossAttentionBlock(dim, dim, heads, dropout) for _ in range(2)]
        )
        spatial_layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.spatial_mixer = nn.TransformerEncoder(spatial_layer, 2, enable_nested_tensor=False)
        self.spatial_norm = nn.LayerNorm(dim)
        self.descriptor_head = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, descriptor_dim),
            nn.Sigmoid(),
        )

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

    def forward(self, eeg: torch.Tensor) -> ColorEEGCondition:
        if eeg.ndim != 4 or eeg.shape[1] != self.subjects or eeg.shape[2] != self.channels:
            raise ValueError(f"expected EEG [B,{self.subjects},{self.channels},time]")
        if eeg.shape[-1] != self.input_samples:
            raise ValueError(f"expected {self.input_samples} time samples")
        batch = len(eeg)
        value = eeg[..., self.post_stimulus_start :].float()
        subject_ids = torch.arange(self.subjects, device=eeg.device).repeat(batch)
        value = value.flatten(0, 1)
        value = (
            value * self.subject_scale[subject_ids, :, None]
            + self.subject_bias[subject_ids, :, None]
        )
        branches = [branch(value[:, None]).squeeze(2) for branch in self.branches]
        temporal = torch.cat(branches, 1).transpose(1, 2)
        temporal = self.temporal_projection(temporal) + self.temporal_position
        temporal = self.temporal_mixer(temporal).reshape(
            batch, self.subjects, temporal.shape[1], self.dim
        )
        summaries = temporal.mean(2)
        logits = self.subject_quality(summaries).squeeze(-1) + self.subject_prior
        weights = logits.softmax(1)
        fused = torch.einsum("bs,bstd->btd", weights, temporal)
        style = self.style_queries.expand(batch, -1, -1)
        for block in self.style_cross:
            style = block(style, fused)
        style = self.style_norm(style)
        spatial = self.spatial_queries.expand(batch, -1, -1)
        for block in self.spatial_cross:
            spatial = block(spatial, fused)
        spatial = self.spatial_norm(self.spatial_mixer(spatial))
        descriptor = self.descriptor_head(torch.cat((style.mean(1), fused.mean(1)), -1))
        return ColorEEGCondition(
            style, spatial.reshape(batch, 16, 16, self.dim), descriptor, weights
        )

    def config(self) -> dict:
        return {
            "subjects": self.subjects,
            "channels": self.channels,
            "input_samples": self.input_samples,
            "post_stimulus_start": self.post_stimulus_start,
            "branch_width": self.branch_width,
            "dim": self.dim,
            "heads": self.heads,
            "depth": self.depth,
            "style_tokens": self.style_tokens,
            "descriptor_dim": self.descriptor_dim,
            "dropout": self.dropout,
        }


class DirectColorDecoder(nn.Module):
    """Render luminance, chroma and palette directly from color-specific EEG tokens."""

    def __init__(
        self,
        style_dim: int = 192,
        width: int = 128,
        heads: int = 4,
        style_tokens: int = 16,
        chroma_bins: int = 32,
        output_size: int = 64,
    ) -> None:
        super().__init__()
        self.style_dim, self.width, self.heads = style_dim, width, heads
        self.style_tokens, self.chroma_bins, self.output_size = (
            style_tokens,
            chroma_bins,
            output_size,
        )
        self.grid_projection = nn.Sequential(
            nn.Conv2d(style_dim, width, 1), ConvResidual(width), ConvResidual(width)
        )
        self.style_projection = nn.Sequential(nn.LayerNorm(style_dim), nn.Linear(style_dim, width))
        self.pixel_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.combined = nn.Sequential(nn.Conv2d(width * 2, width, 1), ConvResidual(width))
        self.palette_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3)
        )
        self.mask_features = nn.Conv2d(width, width, 1)
        self.mask_queries = nn.Linear(width, width)
        self.chroma_head = nn.Sequential(ConvResidual(width), nn.Conv2d(width, 2 * chroma_bins, 1))
        self.palette_categorical_gate = nn.Parameter(torch.tensor(0.0))
        self.pixel16 = nn.Sequential(ConvResidual(width), nn.Conv2d(width, width, 3, padding=1))
        self.up32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(width, width // 2, 3, padding=1),
            ConvResidual(width // 2),
        )
        if output_size == 64:
            self.up64 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(width // 2, width // 4, 3, padding=1),
                ConvResidual(width // 4),
            )
            final_width = width // 4
        elif output_size == 32:
            self.up64 = nn.Identity()
            final_width = width // 2
        else:
            raise ValueError("output_size must be 32 or 64")
        self.residual_head = nn.Conv2d(final_width, 3, 3, padding=1)
        self.register_buffer(
            "chroma_centers", torch.linspace(-0.4, 0.4, chroma_bins), persistent=False
        )

    def forward(
        self, condition: ColorEEGCondition, hard_chroma: bool = False
    ) -> ColorSpecialistOutput:
        structure = self.grid_projection(condition.grid16.permute(0, 3, 1, 2).float())
        style = self.style_projection(condition.style_tokens.float())
        pixels = structure.flatten(2).transpose(1, 2)
        attended, _ = self.pixel_attention(pixels, style, style, need_weights=False)
        batch = len(condition.grid16)
        color_features = attended.transpose(1, 2).reshape(batch, self.width, 16, 16)
        combined = self.combined(torch.cat((structure, color_features), 1))

        palette_raw = self.palette_head(style)
        palette = torch.cat((palette_raw[..., :1].sigmoid(), 0.4 * palette_raw[..., 1:].tanh()), -1)
        masks = (
            torch.einsum(
                "bchw,bkc->bkhw",
                self.mask_features(combined),
                F.normalize(self.mask_queries(style), dim=-1),
            )
            / self.width**0.5
        )
        masks = masks.softmax(1)
        palette_map = torch.einsum("bkhw,bkc->bchw", masks, palette)
        logits = self.chroma_head(color_features).reshape(batch, 2, self.chroma_bins, 16, 16)
        probabilities = logits.softmax(2)
        if hard_chroma:
            categorical = self.chroma_centers[probabilities.argmax(2)]
        else:
            categorical = (probabilities * self.chroma_centers[None, None, :, None, None]).sum(2)
        mixture = torch.sigmoid(self.palette_categorical_gate)
        chroma16 = (1 - mixture) * palette_map[:, 1:] + mixture * categorical
        base16 = torch.cat((palette_map[:, :1], chroma16), 1)
        base = F.interpolate(
            base16, (self.output_size, self.output_size), mode="bilinear", align_corners=False
        )
        residual = torch.tanh(self.residual_head(self.up64(self.up32(self.pixel16(combined)))))
        scale = residual.new_tensor([0.18, 0.16, 0.16])[None, :, None, None]
        oklab = base + scale * residual
        oklab = torch.cat((oklab[:, :1].clamp(0, 1), oklab[:, 1:].clamp(-0.5, 0.5)), 1)
        linear_rgb = oklab_to_linear_rgb(oklab)
        rgb = linear_rgb_to_srgb(linear_rgb)
        image = AdvancedColorOutput(
            rgb,
            oklab,
            linear_rgb,
            color_descriptor(rgb),
            palette,
            masks,
            logits,
            rgb.new_empty((0, 0)),
        )
        return ColorSpecialistOutput(image, condition, condition.grid16)

    def config(self) -> dict:
        return {
            "style_dim": self.style_dim,
            "width": self.width,
            "heads": self.heads,
            "style_tokens": self.style_tokens,
            "chroma_bins": self.chroma_bins,
            "output_size": self.output_size,
        }


class EEGColorSpecialist(nn.Module):
    def __init__(
        self, encoder_config: dict | None = None, refiner_config: dict | None = None
    ) -> None:
        super().__init__()
        self.encoder = MultiscaleEEGColorEncoder(**(encoder_config or {}))
        self.decoder = DirectColorDecoder(**(refiner_config or {}))

    def forward(self, eeg: torch.Tensor, hard_chroma: bool = False) -> ColorSpecialistOutput:
        return self.decoder(self.encoder(eeg), hard_chroma)

    def config(self) -> dict:
        return {"encoder_config": self.encoder.config(), "refiner_config": self.decoder.config()}


class EEGColorEncoderPipeline(EEGColorSpecialist):
    """Deployment name for the self-contained direct EEG-to-color model."""


def load_eeg_color_encoder(
    root: str | Path = "runs/eeg_color_encoder", device: torch.device | str = "cpu"
) -> tuple[EEGColorEncoderPipeline, dict]:
    root = Path(root)
    state = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
    if state.get("architecture") != "independent_direct_color_v1":
        raise RuntimeError(
            "legacy baseline-dependent color checkpoint; train the independent model with "
            "eeg-train-color --init-from <legacy-best.pt>"
        )
    specialist = EEGColorEncoderPipeline(**state["model_config"])
    specialist.load_state_dict(state.get("ema", state["model"]), strict=True)
    pipeline = specialist.to(device).eval()
    for parameter in specialist.parameters():
        parameter.requires_grad_(False)
    return pipeline, state
