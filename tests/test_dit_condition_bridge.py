import pytest
import torch

pytest.importorskip("lightning")

from eeg_mae.data.dit_ten_subjects import normalize_clip
from eeg_mae.lightning.dit_condition_bridge import DiTConditionBridge


def test_bridge_starts_at_normalized_eeg_baseline_and_ignores_absent_modalities():
    torch.set_num_threads(2)
    torch.manual_seed(19)
    batch = {"S": torch.randn(3, 768), "E": torch.randn(3, 12), "C": torch.randn(3, 12)}
    module = DiTConditionBridge(variant="S", structural_dim=12, hidden=16).eval()
    expected = normalize_clip(batch["S"], module.hparams.target_norm)
    torch.testing.assert_close(module(batch), expected)
    with torch.no_grad():
        module.head[-1].weight.normal_()
    first = module(batch)
    batch["E"] *= 100
    batch["C"] += 50
    torch.testing.assert_close(module(batch), first, rtol=0, atol=0)


def test_structural_variants_have_equal_parameter_count_and_use_structure():
    counts = []
    for variant in ("S", "S_E", "S_C", "S_E_C"):
        module = DiTConditionBridge(variant=variant, structural_dim=12, hidden=16).eval()
        counts.append(sum(p.numel() for p in module.parameters()))
    assert len(set(counts)) == 1
    batch = {"S": torch.randn(2, 768), "E": torch.randn(2, 12), "C": torch.randn(2, 12)}
    with torch.no_grad():
        module.head[-1].weight.normal_()
    first = module(batch)
    changed = {**batch, "E": torch.randn_like(batch["E"])}
    assert not torch.allclose(first, module(changed))
    torch.testing.assert_close(
        module(batch).norm(dim=-1), torch.full((2,), module.hparams.target_norm)
    )


def test_generated_image_scoring_uses_full_true_gallery_with_fair_ties():
    from eeg_mae.evaluation.score_dit_generation import gallery_metrics

    gallery = torch.eye(4)
    result = gallery_metrics(gallery[[1, 3]], gallery, [1, 3])
    assert result["top1_true_gallery"] == 1
    null = gallery_metrics(torch.zeros(2, 4), gallery, [1, 3])
    assert null["top1_true_gallery"] == 0.25


def test_render_resume_rejects_changed_weights_or_seed(tmp_path):
    from eeg_mae.evaluation.render_dit_conditions import validate_render_signature

    signature = {"seed": 42, "checkpoint_sha256": "example"}
    validate_render_signature(tmp_path, signature)
    validate_render_signature(tmp_path, signature)
    with pytest.raises(ValueError, match="changed"):
        validate_render_signature(tmp_path, {**signature, "seed": 43})
