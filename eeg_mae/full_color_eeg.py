"""Raw EEG encoder for the full 16x16 antiring color latent."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .model_utils import CrossAttentionBlock
from .semantic_encoder import SubjectAdaptiveSemanticEncoder


@dataclass
class EEGFullColorPrediction:
    grid: torch.Tensor
    global_feature: torch.Tensor
    contexts: torch.Tensor


class EEGFullColorLatentPredictor(nn.Module):
    """Predict a standardized 16x16 oracle latent directly from raw EEG."""

    def __init__(
        self,
        encoder: SubjectAdaptiveSemanticEncoder,
        latent_dim: int = 192,
        dim: int = 192,
        heads: int = 6,
        depth: int = 3,
        context_tokens: int = 6,
        mixer_depth: int = 3,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        self.latent_dim, self.dim, self.heads, self.depth = latent_dim, dim, heads, depth
        self.context_tokens, self.mixer_depth, self.dropout = context_tokens, mixer_depth, dropout
        self.subjects = encoder.subjects
        self.flattened = encoder.channels * encoder.samples
        # Independent spatial anchor: semantic features stay frozen and cannot be forgotten.
        self.spatial_norm_weight = nn.Parameter(encoder.input_norm_weight.detach().clone())
        self.spatial_norm_bias = nn.Parameter(encoder.input_norm_bias.detach().clone())
        self.spatial_weight = nn.Parameter(encoder.linear_weight.detach().clone())
        self.spatial_bias = nn.Parameter(encoder.linear_bias.detach().clone())
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
        self.queries = nn.Parameter(torch.randn(1, 256, dim) * 0.02)
        self.cross = nn.ModuleList(
            [CrossAttentionBlock(dim, dim, heads, dropout) for _ in range(depth)]
        )
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.mixer = nn.TransformerEncoder(layer, mixer_depth, enable_nested_tensor=False)
        self.grid_output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, latent_dim))
        self.global_output = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, latent_dim),
        )
        # Stay close to the standardized train mean, but never start at exactly
        # zero: cosine/structure losses have singular half-precision gradients there.
        nn.init.normal_(self.grid_output[-1].weight, std=1e-3)
        nn.init.zeros_(self.grid_output[-1].bias)
        nn.init.normal_(self.global_output[-1].weight, std=1e-3)
        nn.init.zeros_(self.global_output[-1].bias)

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
                result = value.new_zeros((len(eeg), self.encoder.output_dim))
            result = result.index_copy(0, indices, value)
        return result if result is not None else normalized.new_empty((0, self.encoder.output_dim))

    def encode_context(self, eeg: torch.Tensor, subject: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        with torch.no_grad():
            semantic = self.encoder(eeg, subject)
        spatial = self.spatial_anchor(eeg, subject)
        return torch.cat(
            (
                self.global_context(semantic).reshape(len(eeg), self.context_tokens, self.dim),
                self.spatial_context(spatial).reshape(len(eeg), self.context_tokens, self.dim),
            ),
            1,
        )

    def decode_context(self, context: torch.Tensor) -> EEGFullColorPrediction:
        queries = self.queries.expand(len(context), -1, -1)
        for block in self.cross:
            queries = block(queries, context)
        queries = self.mixer(queries)
        grid = self.grid_output(queries).reshape(len(context), 16, 16, self.latent_dim)
        global_feature = self.global_output(context.mean(1))
        return EEGFullColorPrediction(grid, global_feature, context)

    def forward(self, eeg: torch.Tensor, subject: torch.Tensor) -> EEGFullColorPrediction:
        return self.decode_context(self.encode_context(eeg, subject))

    def forward_ensemble(self, eeg: torch.Tensor) -> EEGFullColorPrediction:
        if eeg.ndim != 4 or eeg.shape[1] != self.subjects:
            raise ValueError(f"expected EEG [batch,{self.subjects},channels,time]")
        batch, subjects = eeg.shape[:2]
        sid = torch.arange(subjects, device=eeg.device).repeat(batch)
        contexts = self.encode_context(eeg.flatten(0, 1), sid).reshape(
            batch, subjects, 2 * self.context_tokens, self.dim
        )
        prediction = self.decode_context(contexts.mean(1))
        return EEGFullColorPrediction(prediction.grid, prediction.global_feature, contexts)

    @torch.no_grad()
    def initialize_from_spatial(self, state: dict) -> dict[str, int]:
        """Reuse compatible 8x8 EEG spatial weights without importing its output head."""
        source = state.get("model", state)
        own = self.state_dict()
        copied = 0
        excluded = ("queries", "output.", "grid_output.", "global_output.", "encoder.")
        for name, value in source.items():
            if name.startswith(excluded) or name not in own or own[name].shape != value.shape:
                continue
            own[name].copy_(value)
            copied += 1
        if (
            "queries" in source
            and source["queries"].shape[1] == 64
            and source["queries"].shape[2] == self.dim
        ):
            query = source["queries"].reshape(1, 8, 8, self.dim).permute(0, 3, 1, 2)
            own["queries"].copy_(
                F.interpolate(query.float(), (16, 16), mode="bicubic", align_corners=False)
                .permute(0, 2, 3, 1)
                .reshape(1, 256, self.dim)
            )
            copied += 1
        self.load_state_dict(own)
        return {"copied_tensors": copied, "source_tensors": len(source)}

    def config(self) -> dict:
        return {
            "latent_dim": self.latent_dim,
            "dim": self.dim,
            "heads": self.heads,
            "depth": self.depth,
            "context_tokens": self.context_tokens,
            "mixer_depth": self.mixer_depth,
            "dropout": self.dropout,
        }
