"""Small neural-network primitives shared by the maintained EEG/edge pipeline."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CrossAttentionBlock(nn.Module):
    """Pre-normalized cross-attention followed by a residual MLP."""

    def __init__(self, query_dim: int, context_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(query_dim)
        self.kv_norm = nn.LayerNorm(context_dim)
        self.attn = nn.MultiheadAttention(
            query_dim,
            heads,
            dropout=dropout,
            batch_first=True,
            kdim=context_dim,
            vdim=context_dim,
        )
        self.ff_norm = nn.LayerNorm(query_dim)
        self.ff = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 4, query_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        normalized_query = self.q_norm(query)
        normalized_context = self.kv_norm(context)
        attended, _ = self.attn(
            normalized_query, normalized_context, normalized_context, need_weights=False
        )
        query = query + attended
        return query + self.ff(self.ff_norm(query))


def image_gradients(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return value[..., 1:, :] - value[..., :-1, :], value[..., :, 1:] - value[..., :, :-1]


def focal_map_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    probability = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    return (bce * (probability - target).abs().pow(gamma)).mean()


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction, target = logits.sigmoid().flatten(1), target.float().flatten(1)
    return (
        1 - (2 * (prediction * target).sum(1) + 1) / (prediction.sum(1) + target.sum(1) + 1)
    ).mean()


def correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction, target = prediction.float().flatten(1), target.float().flatten(1)
    prediction = F.normalize(prediction - prediction.mean(1, keepdim=True), dim=1, eps=1e-6)
    target = F.normalize(target - target.mean(1, keepdim=True), dim=1, eps=1e-6)
    return (1 - (prediction * target).sum(1)).mean()


def grid_contrastive(
    prediction: torch.Tensor, target: torch.Tensor, temperature: float = 0.08
) -> torch.Tensor:
    prediction = F.normalize(prediction.flatten(1), dim=-1)
    target = F.normalize(target.flatten(1), dim=-1)
    logits = prediction @ target.T / temperature
    labels = torch.arange(len(prediction), device=prediction.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def grid_structure_losses(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Losses that a spatially uniform or class-average grid cannot satisfy."""
    pred_residual = F.normalize(prediction - prediction.mean((1, 2), keepdim=True), dim=-1)
    target_residual = F.normalize(target - target.mean((1, 2), keepdim=True), dim=-1)
    residual = (1 - F.cosine_similarity(pred_residual, target_residual, dim=-1)).mean()
    gradients = []
    for pred_delta, target_delta in (
        (prediction[:, 1:] - prediction[:, :-1], target[:, 1:] - target[:, :-1]),
        (prediction[:, :, 1:] - prediction[:, :, :-1], target[:, :, 1:] - target[:, :, :-1]),
    ):
        gradients.append((1 - F.cosine_similarity(pred_delta, target_delta, dim=-1)).mean())
    return residual, torch.stack(gradients).mean()
