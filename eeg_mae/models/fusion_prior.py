"""Rectified-flow prior fusing semantic, color, and geometry EEG conditions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .hierarchical_semantic import BRANCH_DIMS
from .model_utils import CrossAttentionBlock


@dataclass
class FusionPriorCondition:
    tokens: torch.Tensor
    semantic_tokens: torch.Tensor
    spatial_tokens: torch.Tensor
    semantic_prediction: torch.Tensor


class FlowBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.cross = CrossAttentionBlock(width, width, heads, dropout)
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            width * 4,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_attention = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)

    def forward(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.self_attention(self.cross(latent, condition))


def sinusoidal_time_embedding(time: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    frequency = torch.exp(
        -math.log(10_000)
        * torch.arange(half, device=time.device, dtype=time.dtype)
        / max(half - 1, 1)
    )
    phase = time[:, None] * frequency[None] * 1000
    value = torch.cat((phase.sin(), phase.cos()), -1)
    return torch.nn.functional.pad(value, (0, width - value.shape[-1]))


class EEGFusionPrior(nn.Module):
    """Generator-facing prior with explicit semantic and spatial condition tokens.

    The model never consumes an image at inference. During training, the frozen
    SDXL-VAE latent is the endpoint of a conditional rectified-flow path.
    """

    def __init__(
        self,
        semantic_dim: int = 1024,
        spatial_input_dim: int = 192,
        width: int = 256,
        heads: int = 8,
        condition_depth: int = 2,
        flow_depth: int = 4,
        dropout: float = 0.10,
        condition_dropout: float = 0.12,
        latent_channels: int = 4,
        grid_size: int = 16,
    ) -> None:
        super().__init__()
        self.semantic_dim = semantic_dim
        self.spatial_input_dim = spatial_input_dim
        self.width = width
        self.heads = heads
        self.condition_depth = condition_depth
        self.flow_depth = flow_depth
        self.dropout = dropout
        self.condition_dropout = condition_dropout
        self.latent_channels = latent_channels
        self.grid_size = grid_size
        self.semantic_names = ("fused", *BRANCH_DIMS)

        self.semantic_projections = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(semantic_dim), nn.Linear(semantic_dim, width))
                for name in self.semantic_names
            }
        )
        self.color_projection = nn.Sequential(
            nn.LayerNorm(spatial_input_dim), nn.Linear(spatial_input_dim, width)
        )
        self.edge_projection = nn.Sequential(
            nn.LayerNorm(spatial_input_dim), nn.Linear(spatial_input_dim, width)
        )
        self.spatial_queries = nn.Parameter(torch.randn(1, grid_size**2, width) * 0.02)
        self.spatial_position = nn.Parameter(torch.randn(1, grid_size**2, width) * 0.02)
        self.modality_embedding = nn.Parameter(torch.randn(2, 1, width) * 0.02)
        self.spatial_cross = CrossAttentionBlock(width, width, heads, dropout)
        layer = nn.TransformerEncoderLayer(
            width, heads, width * 4, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.condition_mixer = nn.TransformerEncoder(
            layer, condition_depth, enable_nested_tensor=False
        )
        self.condition_norm = nn.LayerNorm(width)
        self.semantic_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, semantic_dim))

        self.latent_input = nn.Linear(latent_channels, width)
        self.latent_position = nn.Parameter(torch.randn(1, grid_size**2, width) * 0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(width, width * 4), nn.SiLU(), nn.Linear(width * 4, width)
        )
        self.flow_blocks = nn.ModuleList(
            [FlowBlock(width, heads, dropout) for _ in range(flow_depth)]
        )
        self.velocity_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, latent_channels))
        self.null_condition = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        nn.init.zeros_(self.velocity_head[-1].weight)
        nn.init.zeros_(self.velocity_head[-1].bias)

    def _drop(self, value: torch.Tensor, enabled: bool) -> torch.Tensor:
        if not enabled or not self.training or self.condition_dropout <= 0:
            return value
        keep = torch.rand(len(value), 1, 1, device=value.device) >= self.condition_dropout
        return value * keep.to(value.dtype)

    def encode_condition(
        self,
        semantic_fused: torch.Tensor,
        semantic_branches: dict[str, torch.Tensor],
        color_grid16: torch.Tensor,
        edge_grid16: torch.Tensor,
        *,
        drop_modalities: bool = False,
        ablate: str | None = None,
    ) -> FusionPriorCondition:
        expected = (self.grid_size, self.grid_size, self.spatial_input_dim)
        if tuple(color_grid16.shape[1:]) != expected or tuple(edge_grid16.shape[1:]) != expected:
            raise ValueError(f"color and edge grids must have shape [B,{expected}]")
        semantic_values = {"fused": semantic_fused, **semantic_branches}
        if set(semantic_values) != set(self.semantic_names):
            raise KeyError(f"expected semantic tokens {self.semantic_names}")
        semantic = torch.stack(
            [
                self.semantic_projections[name](semantic_values[name].float())
                for name in self.semantic_names
            ],
            1,
        )
        color = self.color_projection(color_grid16.float().flatten(1, 2))
        edge = self.edge_projection(edge_grid16.float().flatten(1, 2))
        if ablate in {"semantic", "all"}:
            semantic = torch.zeros_like(semantic)
        if ablate in {"color", "all"}:
            color = torch.zeros_like(color)
        if ablate in {"edge", "all"}:
            edge = torch.zeros_like(edge)
        semantic = self._drop(semantic, drop_modalities)
        color = self._drop(color, drop_modalities) + self.modality_embedding[0]
        edge = self._drop(edge, drop_modalities) + self.modality_embedding[1]
        spatial = self.spatial_queries.expand(len(color), -1, -1) + self.spatial_position
        spatial = self.spatial_cross(spatial, torch.cat((color, edge), 1))
        tokens = self.condition_mixer(torch.cat((semantic, spatial), 1))
        tokens = self.condition_norm(tokens)
        semantic_tokens, spatial_tokens = (
            tokens[:, : len(self.semantic_names)],
            tokens[:, len(self.semantic_names) :],
        )
        semantic_prediction = self.semantic_head(semantic_tokens[:, 0])
        return FusionPriorCondition(tokens, semantic_tokens, spatial_tokens, semantic_prediction)

    def predict_velocity(
        self,
        noisy_latent: torch.Tensor,
        time: torch.Tensor,
        condition: FusionPriorCondition,
        *,
        unconditional: bool = False,
    ) -> torch.Tensor:
        expected = (self.latent_channels, self.grid_size, self.grid_size)
        if tuple(noisy_latent.shape[1:]) != expected:
            raise ValueError(f"latent must have shape [B,{expected}]")
        tokens = noisy_latent.flatten(2).transpose(1, 2).float()
        tokens = self.latent_input(tokens) + self.latent_position
        tokens = (
            tokens + self.time_mlp(sinusoidal_time_embedding(time.float(), self.width))[:, None]
        )
        context = (
            self.null_condition.expand(len(tokens), -1, -1) if unconditional else condition.tokens
        )
        for block in self.flow_blocks:
            tokens = block(tokens, context)
        velocity = self.velocity_head(tokens).transpose(1, 2)
        return velocity.reshape(len(tokens), self.latent_channels, self.grid_size, self.grid_size)

    @torch.no_grad()
    def sample(
        self,
        condition: FusionPriorCondition,
        *,
        steps: int = 24,
        guidance: float = 1.8,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if steps < 1:
            raise ValueError("steps must be positive")
        shape = (len(condition.tokens), self.latent_channels, self.grid_size, self.grid_size)
        value = (
            torch.randn(shape, device=condition.tokens.device) if noise is None else noise.clone()
        )
        delta = 1.0 / steps
        for index in range(steps):
            time = value.new_full((len(value),), index / steps)
            conditional = self.predict_velocity(value, time, condition)
            if guidance != 1:
                null = self.predict_velocity(value, time, condition, unconditional=True)
                conditional = null + guidance * (conditional - null)
            # Heun integration is materially more stable than a same-cost-size Euler path.
            proposal = value + delta * conditional
            if index + 1 < steps:
                next_time = value.new_full((len(value),), (index + 1) / steps)
                corrected = self.predict_velocity(proposal, next_time, condition)
                if guidance != 1:
                    null = self.predict_velocity(proposal, next_time, condition, unconditional=True)
                    corrected = null + guidance * (corrected - null)
                value = value + 0.5 * delta * (conditional + corrected)
            else:
                value = proposal
        return value

    def config(self) -> dict:
        return {
            "semantic_dim": self.semantic_dim,
            "spatial_input_dim": self.spatial_input_dim,
            "width": self.width,
            "heads": self.heads,
            "condition_depth": self.condition_depth,
            "flow_depth": self.flow_depth,
            "dropout": self.dropout,
            "condition_dropout": self.condition_dropout,
            "latent_channels": self.latent_channels,
            "grid_size": self.grid_size,
        }


def load_fusion_prior(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[EEGFusionPrior, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("architecture") != "eeg_rectified_flow_fusion_prior_v1":
        raise RuntimeError("incompatible fusion-prior checkpoint")
    model = EEGFusionPrior(**payload["model_config"])
    model.load_state_dict(payload.get("ema", payload["model"]), strict=True)
    model = model.to(device).eval()
    return model, payload


__all__ = ["EEGFusionPrior", "FusionPriorCondition", "load_fusion_prior"]
