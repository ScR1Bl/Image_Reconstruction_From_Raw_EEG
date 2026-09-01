"""Task-specific residual adapters on top of a frozen EEG-to-DINO model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .model_utils import CrossAttentionBlock


@dataclass
class AdaptedFeatures:
    grid: torch.Tensor
    global_feature: torch.Tensor | None
    descriptor: torch.Tensor | None
    grid_delta: torch.Tensor
    global_delta: torch.Tensor | None
    subject_weights: torch.Tensor


class TaskResidualAdapter(nn.Module):
    """Learn task information unavailable in the generic DINO grid.

    The adapter starts as an exact identity. It learns a residual grid and,
    for color, a residual global condition. Subject fusion is sample-adaptive
    instead of being a fixed arithmetic mean.
    """

    def __init__(
        self,
        grid_dim: int = 192,
        context_dim: int = 192,
        subjects: int = 10,
        heads: int = 6,
        depth: int = 2,
        dropout: float = 0.10,
        global_dim: int | None = None,
        descriptor_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.grid_dim, self.context_dim, self.subjects = grid_dim, context_dim, subjects
        self.heads, self.depth, self.dropout = heads, depth, dropout
        self.global_dim, self.descriptor_dim = global_dim, descriptor_dim
        self.subject_bias = nn.Parameter(torch.zeros(subjects))
        self.subject_quality = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, context_dim // 2),
            nn.GELU(),
            nn.Linear(context_dim // 2, 1),
        )
        nn.init.zeros_(self.subject_quality[-1].weight)
        nn.init.zeros_(self.subject_quality[-1].bias)
        self.grid_projection = nn.Linear(grid_dim, context_dim)
        self.position = nn.Parameter(torch.randn(1, 64, context_dim) * 0.01)
        self.cross = nn.ModuleList(
            [CrossAttentionBlock(context_dim, context_dim, heads, dropout) for _ in range(depth)]
        )
        layer = nn.TransformerEncoderLayer(
            context_dim,
            heads,
            context_dim * 4,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mixer = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.grid_delta_head = nn.Sequential(
            nn.LayerNorm(context_dim), nn.Linear(context_dim, grid_dim)
        )
        nn.init.zeros_(self.grid_delta_head[-1].weight)
        nn.init.zeros_(self.grid_delta_head[-1].bias)
        self.grid_gate = nn.Parameter(torch.tensor(-1.5))

        if (global_dim is None) != (descriptor_dim is None):
            raise ValueError("global_dim and descriptor_dim must be supplied together")
        if global_dim is not None:
            self.descriptor_head = nn.Sequential(
                nn.LayerNorm(context_dim * 2),
                nn.Linear(context_dim * 2, context_dim),
                nn.GELU(),
                nn.Linear(context_dim, descriptor_dim),
                nn.Sigmoid(),
            )
            self.global_delta_head = nn.Sequential(
                nn.LayerNorm(context_dim * 2 + descriptor_dim),
                nn.Linear(context_dim * 2 + descriptor_dim, context_dim * 2),
                nn.GELU(),
                nn.Linear(context_dim * 2, global_dim),
            )
            nn.init.zeros_(self.global_delta_head[-1].weight)
            nn.init.zeros_(self.global_delta_head[-1].bias)
            self.global_gate = nn.Parameter(torch.tensor(-1.5))
        else:
            self.descriptor_head = None
            self.global_delta_head = None
            self.register_parameter("global_gate", None)

    def fuse_subjects(self, contexts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if contexts.ndim != 4:
            raise ValueError("contexts must have shape [batch,subjects,tokens,features]")
        if contexts.shape[1] != self.subjects:
            raise ValueError(f"expected {self.subjects} subjects, got {contexts.shape[1]}")
        summaries = contexts.mean(2)
        logits = self.subject_quality(summaries).squeeze(-1) + self.subject_bias
        weights = logits.softmax(1)
        fused = torch.einsum("bs,bstd->btd", weights, contexts)
        return fused, weights

    def forward(
        self,
        base_grid: torch.Tensor,
        contexts: torch.Tensor,
        base_global: torch.Tensor | None = None,
    ) -> AdaptedFeatures:
        fused, subject_weights = self.fuse_subjects(contexts)
        tokens = self.grid_projection(base_grid.flatten(1, 2)) + self.position
        for block in self.cross:
            tokens = block(tokens, fused)
        tokens = self.mixer(tokens)
        grid_delta = self.grid_delta_head(tokens).reshape_as(base_grid)
        grid = F.normalize(base_grid + torch.sigmoid(self.grid_gate) * grid_delta, dim=-1)

        descriptor = global_delta = global_feature = None
        if self.global_delta_head is not None:
            if base_global is None:
                raise ValueError("base_global is required by a color adapter")
            summary = torch.cat((tokens.mean(1), fused.mean(1)), -1)
            descriptor = self.descriptor_head(summary)
            global_delta = self.global_delta_head(torch.cat((summary, descriptor), -1))
            global_feature = F.normalize(
                base_global + torch.sigmoid(self.global_gate) * global_delta, dim=-1
            )
        elif base_global is not None:
            global_feature = base_global
        return AdaptedFeatures(
            grid, global_feature, descriptor, grid_delta, global_delta, subject_weights
        )

    def config(self) -> dict:
        return {
            "grid_dim": self.grid_dim,
            "context_dim": self.context_dim,
            "subjects": self.subjects,
            "heads": self.heads,
            "depth": self.depth,
            "dropout": self.dropout,
            "global_dim": self.global_dim,
            "descriptor_dim": self.descriptor_dim,
        }


def cvar(values: torch.Tensor, fraction: float = 0.30) -> torch.Tensor:
    """Mean of the hardest samples in a batch."""
    count = max(1, round(len(values) * fraction))
    return values.topk(count).values.mean()


def improvement_loss(
    new_error: torch.Tensor, baseline_error: torch.Tensor, margin: float, temperature: float = 0.02
) -> torch.Tensor:
    """Smoothly require every prediction to beat its frozen baseline."""
    difference = new_error - baseline_error.detach() + margin
    return (F.softplus(difference / temperature) * temperature).mean()


def shuffled_ranking_loss(
    true_error: torch.Tensor, shuffled_error: torch.Tensor, margin: float
) -> torch.Tensor:
    return F.relu(true_error - shuffled_error + margin).mean()
