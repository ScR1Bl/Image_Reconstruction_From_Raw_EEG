from __future__ import annotations

from pathlib import Path

import torch

from eeg_mae.models.hierarchical_semantic import (
    BRANCH_DIMS,
    HierarchicalSemanticEEGEncoder,
    HierarchicalVisualFusion,
    load_hierarchical_semantic_encoder,
)


def teacher_targets(batch: int) -> dict[str, torch.Tensor]:
    return {
        "synclr": torch.randn(batch, 768),
        "clip_b32": torch.randn(batch, 512),
        "vae_sdxl": torch.randn(batch, 4, 16, 16),
    }


def test_hierarchical_encoder_has_three_token_heads() -> None:
    model = HierarchicalSemanticEEGEncoder(
        fusion_dim=64,
        hidden_dim=96,
        branch_hidden_dim=24,
        backbone_config={"subjects": 2, "temporal_features": 8},
    )
    output = model(torch.randn(3, 17, 100), torch.tensor([0, 1, 0]))
    assert output.fused.shape == (3, 64)
    assert set(output.branches) == set(BRANCH_DIMS)
    assert all(value.shape == (3, 64) for value in output.branches.values())
    assert output.dino_anchor.shape == (3, 384)


def test_visual_fusion_projects_all_teacher_families() -> None:
    fusion = HierarchicalVisualFusion(fusion_dim=64, hidden_dim=96)
    fused, branches = fusion(teacher_targets(5))
    assert fused.shape == (5, 64)
    assert set(branches) == set(BRANCH_DIMS)
    assert torch.isfinite(fused).all()


def test_modality_dropout_never_drops_every_teacher() -> None:
    fusion = HierarchicalVisualFusion(fusion_dim=32, hidden_dim=48, modality_dropout=1.0).train()
    fused, _ = fusion(teacher_targets(20), apply_modality_dropout=True)
    assert torch.isfinite(fused).all()
    assert fused.std() > 0


def test_hierarchical_checkpoint_round_trip(tmp_path: Path) -> None:
    encoder = HierarchicalSemanticEEGEncoder(
        fusion_dim=32,
        hidden_dim=48,
        branch_hidden_dim=16,
        backbone_config={"subjects": 2, "temporal_features": 8},
    )
    fusion = HierarchicalVisualFusion(fusion_dim=32, hidden_dim=48)
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "visual_fusion": fusion.state_dict(),
            "encoder_config": encoder.config(),
            "fusion_config": fusion.config(),
        },
        path,
    )

    restored, restored_fusion, _ = load_hierarchical_semantic_encoder(path)

    assert restored.fusion_dim == 32
    assert restored_fusion.fusion_dim == 32
