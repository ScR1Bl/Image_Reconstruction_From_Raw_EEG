"""Permutation-invariant subject fusion (Step 1b).

Both variants consume RAW per-subject embeddings ``[batch, n, dim]`` plus the
TRUE subject identities ``[n]`` (or ``[batch, n]``) and return a RAW fused
embedding — normalization happens inside the metric/loss, exactly like the
200-way gate (mean BEFORE normalization; decision 2026-09-01, amendment #1).

``AttentionSubjectPooling`` starts as an exact arithmetic mean (zero-init score
head and prior -> uniform softmax), following the identity-at-init house style.
Identity indexing (not position) keeps arbitrary subject subsets correct —
unlike the legacy ``forward_ensemble`` (known bug, tests/test_known_issues.py).
"""

from __future__ import annotations

import torch
from torch import nn


class MeanSubjectPooling(nn.Module):
    """Masked-free arithmetic mean over the subject axis — the legacy gate fusion."""

    def forward(
        self, embeddings: torch.Tensor, subject_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, count = embeddings.shape[:2]
        weights = embeddings.new_full((batch, count), 1.0 / count)
        return embeddings.mean(1), weights


class AttentionSubjectPooling(nn.Module):
    """Learned attention pooling over subjects; N never enters weight shapes."""

    def __init__(self, dim: int = 384, hidden: int = 128, roster: int = 10) -> None:
        super().__init__()
        self.dim, self.hidden, self.roster = dim, hidden, roster
        self.score = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)
        self.subject_prior = nn.Parameter(torch.zeros(roster))

    def forward(
        self, embeddings: torch.Tensor, subject_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [batch, subjects, dim]")
        logits = self.score(embeddings).squeeze(-1)
        prior = self.subject_prior[subject_ids]
        if prior.ndim == 1:
            prior = prior[None, :]
        weights = (logits + prior).softmax(1)
        return torch.einsum("bn,bnd->bd", weights, embeddings), weights


def build_fusion(kind: str, dim: int, roster: int = 10) -> nn.Module:
    if kind == "mean":
        return MeanSubjectPooling()
    if kind == "attention":
        return AttentionSubjectPooling(dim=dim, roster=roster)
    raise ValueError(f"unknown fusion kind: {kind}")
