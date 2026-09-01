import torch

from eeg_mae.models.advanced_color import (
    DINOColorOracleV2,
    FullColorLatent,
    advanced_color_loss,
    color_artifact_metrics,
    oklab_to_linear_rgb,
    oklab_to_rgb,
)
from eeg_mae.models.color import rgb_to_oklab


def test_oklab_roundtrip_is_accurate():
    torch.manual_seed(4)
    rgb = torch.rand(2, 3, 16, 16)
    restored = oklab_to_rgb(rgb_to_oklab(rgb))
    torch.testing.assert_close(restored, rgb, atol=2e-5, rtol=2e-5)


def test_advanced_oracle_shapes_and_gradients():
    torch.manual_seed(5)
    model = DINOColorOracleV2(
        input_dim=24,
        layers=3,
        width=48,
        heads=4,
        transformer_depth=1,
        palette_queries=9,
        chroma_bins=12,
        output_size=32,
        dropout=0,
    )
    patches = torch.randn(2, 3, 16, 16, 24)
    cls = torch.randn(2, 3, 24)
    target = torch.rand(2, 3, 32, 32)
    output = model(patches, cls)
    loss, parts = advanced_color_loss(output, target)
    loss.backward()
    assert output.rgb.shape == (2, 3, 32, 32)
    assert output.palette.shape == (2, 9, 3)
    assert output.masks16.shape == (2, 9, 16, 16)
    assert output.chroma_logits16.shape == (2, 2, 12, 16, 16)
    torch.testing.assert_close(output.masks16.sum(1), torch.ones(2, 16, 16), atol=1e-5, rtol=1e-5)
    assert model.palette_head[-1].weight.grad is not None
    assert torch.isfinite(loss) and "categorical" in parts
    assert {
        "chroma_gradient",
        "hard_oklab",
        "chroma_edge_artifact",
        "gamut_violation",
    } <= parts.keys()


def test_hard_chroma_path_is_finite_and_bounded():
    model = DINOColorOracleV2(
        input_dim=16,
        layers=2,
        width=32,
        heads=4,
        transformer_depth=1,
        palette_queries=4,
        chroma_bins=8,
        output_size=32,
        dropout=0,
    ).eval()
    output = model(torch.randn(1, 2, 16, 16, 16), torch.randn(1, 2, 16), hard_chroma=True)
    assert torch.isfinite(output.rgb).all()
    assert output.rgb.min() >= 0 and output.rgb.max() <= 1


def test_artifact_metrics_reward_exact_color_and_detect_green_ringing():
    target = torch.full((2, 3, 32, 32), 0.45)
    exact_oklab = rgb_to_oklab(target)
    exact = color_artifact_metrics(exact_oklab, target, oklab_to_linear_rgb(exact_oklab))
    ringing = target.clone()
    ringing[:, 1, :, ::2] = 0.9
    ringing[:, 0, :, ::2] = 0.05
    bad_oklab = rgb_to_oklab(ringing)
    bad = color_artifact_metrics(bad_oklab, target, oklab_to_linear_rgb(bad_oklab))
    assert exact["chroma_edge_artifact"] < 1e-6
    assert bad["chroma_edge_artifact"] > exact["chroma_edge_artifact"]
    assert bad["oklab_delta_p95"] > exact["oklab_delta_p95"]


def test_full_latent_can_be_encoded_and_rendered_without_dino_bypass():
    model = DINOColorOracleV2(
        input_dim=16,
        layers=2,
        width=32,
        heads=4,
        transformer_depth=1,
        palette_queries=4,
        chroma_bins=8,
        output_size=32,
        dropout=0,
    ).eval()
    patches = torch.randn(2, 2, 16, 16, 16)
    cls = torch.randn(2, 2, 16)
    latent = model.encode_full_latent(patches, cls)
    assert latent.grid.shape == (2, 16, 16, 32)
    assert latent.global_feature.shape == (2, 32)
    output = model.decode_full_latent(FullColorLatent(latent.grid, latent.global_feature))
    assert output.rgb.shape == (2, 3, 32, 32) and torch.isfinite(output.rgb).all()
