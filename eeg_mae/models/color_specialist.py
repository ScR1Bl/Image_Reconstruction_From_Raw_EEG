"""Color-specialized EEG encoder and structure-preserving chroma refiner."""

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
from .color import color_descriptor, rgb_to_oklab
from .model_utils import CrossAttentionBlock


@dataclass
class ColorEEGCondition:
    style_tokens: torch.Tensor
    descriptor: torch.Tensor
    subject_weights: torch.Tensor


@dataclass
class ColorSpecialistOutput:
    image: AdvancedColorOutput
    condition: ColorEEGCondition
    alpha16: torch.Tensor
    chroma_delta16: torch.Tensor


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
        descriptor = self.descriptor_head(torch.cat((style.mean(1), fused.mean(1)), -1))
        return ColorEEGCondition(style, descriptor, weights)

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


class StructurePreservingColorRefiner(nn.Module):
    """Change chroma only; baseline luminance and spatial stability are immutable."""

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
        self.structure = nn.Sequential(
            nn.Conv2d(4, width, 3, padding=1), ConvResidual(width), ConvResidual(width)
        )
        self.style_projection = nn.Sequential(nn.LayerNorm(style_dim), nn.Linear(style_dim, width))
        self.pixel_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.combined = nn.Sequential(nn.Conv2d(width * 2, width, 1), ConvResidual(width))
        self.palette_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2)
        )
        self.mask_features = nn.Conv2d(width, width, 1)
        self.mask_queries = nn.Linear(width, width)
        self.chroma_head = nn.Sequential(ConvResidual(width), nn.Conv2d(width, 2 * chroma_bins, 1))
        self.palette_categorical_gate = nn.Parameter(torch.tensor(0.0))
        self.residual_head = nn.Sequential(ConvResidual(width), nn.Conv2d(width, 2, 3, padding=1))
        self.alpha_head = nn.Sequential(ConvResidual(width), nn.Conv2d(width, 1, 1))
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.alpha_head[-1].weight)
        nn.init.zeros_(self.alpha_head[-1].bias)
        self.global_alpha = nn.Parameter(torch.tensor(-3.0))
        self.register_buffer(
            "chroma_centers", torch.linspace(-0.4, 0.4, chroma_bins), persistent=False
        )

    def forward(
        self, baseline_rgb: torch.Tensor, condition: ColorEEGCondition, hard_chroma: bool = False
    ) -> ColorSpecialistOutput:
        baseline = F.interpolate(
            baseline_rgb.float(),
            (self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        baseline_oklab = rgb_to_oklab(baseline)
        low = F.adaptive_avg_pool2d(baseline_oklab, 16)
        edges = (low[:, :1, :, 1:] - low[:, :1, :, :-1]).abs()
        edges = F.pad(edges, (0, 1, 0, 0))
        structure = self.structure(torch.cat((low, edges), 1))
        style = self.style_projection(condition.style_tokens.float())
        pixels = structure.flatten(2).transpose(1, 2)
        attended, _ = self.pixel_attention(pixels, style, style, need_weights=False)
        color_features = attended.transpose(1, 2).reshape(len(baseline), self.width, 16, 16)
        combined = self.combined(torch.cat((structure, color_features), 1))

        palette_ab = 0.4 * self.palette_head(style).tanh()
        masks = (
            torch.einsum(
                "bchw,bkc->bkhw",
                self.mask_features(combined),
                F.normalize(self.mask_queries(style), dim=-1),
            )
            / self.width**0.5
        )
        masks = masks.softmax(1)
        palette_chroma = torch.einsum("bkhw,bkc->bchw", masks, palette_ab)
        logits = self.chroma_head(color_features).reshape(
            len(baseline), 2, self.chroma_bins, 16, 16
        )
        probabilities = logits.softmax(2)
        if hard_chroma:
            categorical = self.chroma_centers[probabilities.argmax(2)]
        else:
            categorical = (probabilities * self.chroma_centers[None, None, :, None, None]).sum(2)
        mixture = torch.sigmoid(self.palette_categorical_gate)
        proposal = (1 - mixture) * palette_chroma + mixture * categorical
        proposal = proposal + 0.10 * torch.tanh(self.residual_head(color_features))
        baseline_chroma = low[:, 1:]
        alpha = torch.sigmoid(self.global_alpha) * torch.sigmoid(self.alpha_head(combined))
        chroma16 = baseline_chroma + alpha * (proposal - baseline_chroma)
        chroma = F.interpolate(
            chroma16, (self.output_size, self.output_size), mode="bilinear", align_corners=False
        )
        oklab = torch.cat((baseline_oklab[:, :1], chroma.clamp(-0.5, 0.5)), 1)
        linear_rgb = oklab_to_linear_rgb(oklab)
        rgb = linear_rgb_to_srgb(linear_rgb)
        mass = masks.sum((2, 3)).clamp_min(1e-6)
        palette_l = torch.einsum("bkhw,bhw->bk", masks, low[:, 0]) / mass
        palette = torch.cat((palette_l[..., None], palette_ab), -1)
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
        return ColorSpecialistOutput(image, condition, alpha, chroma16 - baseline_chroma)

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
        self.refiner = StructurePreservingColorRefiner(**(refiner_config or {}))

    def forward(
        self, eeg: torch.Tensor, baseline_rgb: torch.Tensor, hard_chroma: bool = False
    ) -> ColorSpecialistOutput:
        return self.refiner(baseline_rgb, self.encoder(eeg), hard_chroma)

    def config(self) -> dict:
        return {"encoder_config": self.encoder.config(), "refiner_config": self.refiner.config()}


class EEGColorEncoderPipeline(nn.Module):
    """Self-contained inference pipeline loaded from ``runs/eeg_color_encoder``."""

    def __init__(self, specialist, predictor, adapter, decoder) -> None:
        super().__init__()
        self.specialist = specialist
        self.predictor = predictor
        self.adapter = adapter
        self.decoder = decoder

    def forward(self, eeg: torch.Tensor, hard_chroma: bool = False) -> ColorSpecialistOutput:
        batch, subjects = eeg.shape[:2]
        grid, contexts = self.predictor.forward_ensemble(eeg)
        ids = torch.arange(subjects, device=eeg.device).repeat(batch)
        semantic = self.predictor.encoder(eeg.flatten(0, 1), ids)
        semantic = F.normalize(semantic.reshape(batch, subjects, -1).mean(1), dim=-1)
        adapted = self.adapter(grid, contexts, semantic)
        baseline = self.decoder(adapted.grid, adapted.global_feature).rgb32
        return self.specialist(eeg, baseline, hard_chroma)


def load_eeg_color_encoder(
    root: str | Path = "runs/eeg_color_encoder", device: torch.device | str = "cpu"
) -> tuple[EEGColorEncoderPipeline, dict]:
    from .color import ColorDecoder
    from .semantic_edge import EEGDINOGridPredictor
    from .semantic_encoder import SubjectAdaptiveSemanticEncoder
    from .task_adapter import TaskResidualAdapter

    root = Path(root)
    components = root / "components"
    state = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
    specialist = EEGColorSpecialist(**state["model_config"])
    specialist.load_state_dict(state.get("ema", state["model"]), strict=True)
    semantic_state = torch.load(
        components / "semantic_encoder.pt", map_location="cpu", weights_only=False
    )
    semantic = SubjectAdaptiveSemanticEncoder(**semantic_state["model_config"])
    semantic.load_state_dict(semantic_state["model"], strict=True)
    adapter_state = torch.load(components / "adapter.pt", map_location="cpu", weights_only=False)
    predictor = EEGDINOGridPredictor(semantic, **adapter_state["predictor_config"])
    predictor.load_state_dict(adapter_state["predictor"], strict=True)
    adapter = TaskResidualAdapter(**adapter_state["adapter_config"])
    adapter.load_state_dict(adapter_state["adapter"], strict=True)
    decoder_state = torch.load(
        components / "color_decoder.pt", map_location="cpu", weights_only=False
    )
    decoder = ColorDecoder(**decoder_state["model_config"])
    decoder.load_state_dict(decoder_state["model"], strict=True)
    pipeline = EEGColorEncoderPipeline(specialist, predictor, adapter, decoder).to(device).eval()
    for parameter in pipeline.parameters():
        parameter.requires_grad_(False)
    return pipeline, state
