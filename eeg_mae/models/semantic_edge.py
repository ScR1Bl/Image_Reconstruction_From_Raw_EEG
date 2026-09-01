"""EEG encoder specialized for predicting a spatial DINO feature grid."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .model_utils import CrossAttentionBlock
from .semantic_encoder import SubjectAdaptiveSemanticEncoder


class EEGDINOGridPredictor(nn.Module):
    """Predict local 8x8 DINO features from global semantics and raw EEG."""

    def __init__(
        self,
        encoder: SubjectAdaptiveSemanticEncoder,
        grid_dim: int = 192,
        dim: int = 192,
        heads: int = 6,
        depth: int = 3,
        context_tokens: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        self.grid_dim, self.dim, self.heads, self.depth = grid_dim, dim, heads, depth
        self.context_tokens, self.dropout = context_tokens, dropout
        subjects = encoder.subjects
        flattened = encoder.channels * encoder.samples
        # This second subject-specific bank starts from the semantic solution,
        # then specializes toward spatial information without changing it.
        self.spatial_norm_weight = nn.Parameter(encoder.input_norm_weight.detach().clone())
        self.spatial_norm_bias = nn.Parameter(encoder.input_norm_bias.detach().clone())
        self.spatial_weight = nn.Parameter(encoder.linear_weight.detach().clone())
        self.spatial_bias = nn.Parameter(encoder.linear_bias.detach().clone())
        self.subjects, self.flattened = subjects, flattened
        self.global_context = nn.Sequential(
            nn.LayerNorm(encoder.output_dim),
            nn.Linear(encoder.output_dim, context_tokens * dim),
            nn.GELU(),
        )
        self.spatial_context = nn.Sequential(
            nn.LayerNorm(encoder.output_dim),
            nn.Linear(encoder.output_dim, context_tokens * dim),
            nn.GELU(),
        )
        self.queries = nn.Parameter(torch.randn(1, 64, dim) * 0.02)
        self.cross = nn.ModuleList(
            [CrossAttentionBlock(dim, dim, heads, dropout) for _ in range(depth)]
        )
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.mixer = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, grid_dim))

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def spatial_anchor(self, eeg: torch.Tensor, subject: torch.Tensor) -> torch.Tensor:
        eeg = self.encoder.crop(eeg)
        normalized = F.layer_norm(eeg.flatten(1), (self.flattened,))
        result = None
        for sid in subject.unique(sorted=True):
            indices = torch.where(subject == sid)[0]
            value = normalized.index_select(0, indices)
            value = value * self.spatial_norm_weight[sid] + self.spatial_norm_bias[sid]
            value = F.linear(value, self.spatial_weight[sid], self.spatial_bias[sid])
            if result is None:
                result = value.new_zeros((len(eeg), value.shape[1]))
            result = result.index_copy(0, indices, value)
        if result is None:
            return normalized.new_empty((0, self.encoder.output_dim))
        return result

    def forward(self, eeg: torch.Tensor, subject: torch.Tensor) -> torch.Tensor:
        return self.decode_context(self.encode_context(eeg, subject))

    def encode_context(self, eeg: torch.Tensor, subject: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        with torch.no_grad():
            global_semantic = self.encoder(eeg, subject)
        spatial = self.spatial_anchor(eeg, subject)
        return torch.cat(
            (
                self.global_context(global_semantic).reshape(
                    len(eeg), self.context_tokens, self.dim
                ),
                self.spatial_context(spatial).reshape(len(eeg), self.context_tokens, self.dim),
            ),
            1,
        )

    def decode_context(self, context: torch.Tensor) -> torch.Tensor:
        queries = self.queries.expand(len(context), -1, -1)
        for block in self.cross:
            queries = block(queries, context)
        queries = self.mixer(queries)
        return self.output(queries).reshape(len(context), 8, 8, self.grid_dim)

    def forward_ensemble(self, eeg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse subject contexts before decoding them into a visual grid."""
        batch, subjects = eeg.shape[:2]
        sid = torch.arange(subjects, device=eeg.device).repeat(batch)
        context = self.encode_context(eeg.flatten(0, 1), sid).reshape(
            batch, subjects, 2 * self.context_tokens, self.dim
        )
        return self.decode_context(context.mean(1)), context

    def config(self) -> dict:
        return {
            "grid_dim": self.grid_dim,
            "dim": self.dim,
            "heads": self.heads,
            "depth": self.depth,
            "context_tokens": self.context_tokens,
            "dropout": self.dropout,
        }
