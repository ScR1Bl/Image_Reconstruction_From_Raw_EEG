"""Hierarchical EEG semantics aligned to SynCLR, CLIP, and SDXL-VAE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .semantic_encoder import SubjectAdaptiveSemanticEncoder

BRANCH_DIMS = {"synclr": 768, "clip_b32": 512, "vae_sdxl": 1024}


class ResidualMLP(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.layers = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.layers(self.norm(value))


class HierarchicalVisualFusion(nn.Module):
    """Brain-HIVE-style learned projection and fusion of frozen teachers."""

    def __init__(
        self,
        fusion_dim: int = 1024,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
        modality_dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.modality_dropout = modality_dropout
        self.projections = nn.ModuleDict(
            {
                key: nn.Sequential(nn.LayerNorm(size), nn.Linear(size, fusion_dim))
                for key, size in BRANCH_DIMS.items()
            }
        )
        self.branch_norms = nn.ModuleDict({key: nn.LayerNorm(fusion_dim) for key in BRANCH_DIMS})
        self.fusion = ResidualMLP(fusion_dim, hidden_dim, dropout)
        self.output_norm = nn.LayerNorm(fusion_dim)

    def project_branches(self, targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if set(targets) != set(BRANCH_DIMS):
            raise KeyError(f"expected targets {tuple(BRANCH_DIMS)}, got {tuple(targets)}")
        branches = {}
        for key, expected_dim in BRANCH_DIMS.items():
            value = targets[key].flatten(1)
            if value.shape[1] != expected_dim:
                raise ValueError(f"{key} must flatten to {expected_dim}, got {value.shape[1]}")
            branches[key] = self.branch_norms[key](self.projections[key](value))
        return branches

    def _keep_mask(self, batch: int, device: torch.device) -> torch.Tensor:
        keep = torch.rand(batch, len(BRANCH_DIMS), device=device) >= self.modality_dropout
        empty = ~keep.any(1)
        if empty.any():
            rows = torch.where(empty)[0]
            columns = torch.randint(len(BRANCH_DIMS), (len(rows),), device=device)
            keep[rows, columns] = True
        return keep

    def combine(
        self, branches: dict[str, torch.Tensor], apply_modality_dropout: bool = False
    ) -> torch.Tensor:
        values = torch.stack([branches[key] for key in BRANCH_DIMS], 1)
        if apply_modality_dropout and self.training and self.modality_dropout > 0:
            keep = self._keep_mask(len(values), values.device).to(values.dtype)
            values = values * keep[:, :, None]
            count = keep.sum(1, keepdim=True).clamp_min(1)
        else:
            count = values.new_full((len(values), 1), len(BRANCH_DIMS))
        fused = values.sum(1) / count.sqrt()
        return self.output_norm(self.fusion(fused))

    def forward(
        self, targets: dict[str, torch.Tensor], apply_modality_dropout: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        branches = self.project_branches(targets)
        return self.combine(branches, apply_modality_dropout), branches

    def config(self) -> dict:
        return {
            "fusion_dim": self.fusion_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "modality_dropout": self.modality_dropout,
        }


@dataclass
class HierarchicalEEGOutput:
    fused: torch.Tensor
    branches: dict[str, torch.Tensor]
    dino_anchor: torch.Tensor


class HierarchicalSemanticEEGEncoder(nn.Module):
    """Existing DINO EEG encoder extended with three complementary token heads."""

    def __init__(
        self,
        anchor_dim: int = 384,
        fusion_dim: int = 1024,
        hidden_dim: int = 1024,
        branch_hidden_dim: int = 256,
        dropout: float = 0.15,
        backbone_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.anchor_dim = anchor_dim
        self.fusion_dim = fusion_dim
        self.hidden_dim = hidden_dim
        self.branch_hidden_dim = branch_hidden_dim
        self.dropout = dropout
        config = dict(backbone_config or {})
        config["output_dim"] = anchor_dim
        self.backbone = SubjectAdaptiveSemanticEncoder(**config)
        self.shared = nn.Sequential(
            nn.LayerNorm(anchor_dim),
            nn.Linear(anchor_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.branch_adapters = nn.ModuleDict(
            {
                key: nn.Sequential(
                    nn.LayerNorm(fusion_dim),
                    nn.Linear(fusion_dim, branch_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(branch_hidden_dim, fusion_dim),
                )
                for key in BRANCH_DIMS
            }
        )
        self.branch_norms = nn.ModuleDict({key: nn.LayerNorm(fusion_dim) for key in BRANCH_DIMS})
        self.fusion = ResidualMLP(fusion_dim, hidden_dim, dropout)
        self.output_norm = nn.LayerNorm(fusion_dim)
        for adapter in self.branch_adapters.values():
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)

    def load_dino_anchor(self, checkpoint: str | Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(payload.get("model_config", {}).get("output_dim", -1)) != self.anchor_dim:
            raise RuntimeError("DINO anchor checkpoint has an incompatible output dimension")
        self.backbone.load_state_dict(payload["model"], strict=True)

    def head_parameters(self) -> list[nn.Parameter]:
        backbone_ids = {id(parameter) for parameter in self.backbone.parameters()}
        return [parameter for parameter in self.parameters() if id(parameter) not in backbone_ids]

    def forward(self, eeg: torch.Tensor, subject: torch.Tensor) -> HierarchicalEEGOutput:
        anchor = self.backbone(eeg, subject)
        shared = self.shared(anchor)
        branches = {
            key: self.branch_norms[key](shared + adapter(shared))
            for key, adapter in self.branch_adapters.items()
        }
        fused = torch.stack([branches[key] for key in BRANCH_DIMS], 1).sum(1)
        fused = fused / (len(BRANCH_DIMS) ** 0.5)
        fused = self.output_norm(self.fusion(fused))
        return HierarchicalEEGOutput(fused, branches, anchor)

    def config(self) -> dict:
        return {
            "anchor_dim": self.anchor_dim,
            "fusion_dim": self.fusion_dim,
            "hidden_dim": self.hidden_dim,
            "branch_hidden_dim": self.branch_hidden_dim,
            "dropout": self.dropout,
            "backbone_config": self.backbone.config(),
        }


def load_hierarchical_semantic_encoder(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[HierarchicalSemanticEEGEncoder, HierarchicalVisualFusion, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    encoder = HierarchicalSemanticEEGEncoder(**payload["encoder_config"])
    fusion = HierarchicalVisualFusion(**payload["fusion_config"])
    encoder.load_state_dict(payload["encoder"], strict=True)
    fusion.load_state_dict(payload["visual_fusion"], strict=True)
    return encoder.to(device), fusion.to(device), payload
