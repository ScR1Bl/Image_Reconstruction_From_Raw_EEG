from __future__ import annotations

import torch

from eeg_mae.color import (
    ColorDecoder,
    color_descriptor,
    color_loss,
    color_metrics,
    rgb_to_ycbcr,
)


def test_color_decoder_outputs_rgb_and_global_descriptor():
    model = ColorDecoder(grid_dim=24, global_dim=32, width=32)
    grid = torch.randn(3, 8, 8, 24)
    global_feature = torch.randn(3, 32)
    target = torch.rand(3, 3, 32, 32)
    descriptor = color_descriptor(target).detach()
    output = model(grid, global_feature)
    loss, parts = color_loss(output, target, descriptor)
    loss.backward()
    assert output.rgb32.shape == (3, 3, 32, 32)
    assert output.descriptor.shape == (3, 63)
    assert output.rgb32.min() >= 0 and output.rgb32.max() <= 1
    assert model.stem[0].weight.grad is not None
    assert torch.isfinite(loss) and "chroma" in parts


def test_color_metrics_reward_exact_color_and_detect_permutation():
    torch.manual_seed(8)
    target = torch.rand(12, 3, 32, 32)
    descriptor = color_descriptor(target)
    exact = type("Output", (), {"rgb32": target, "descriptor": descriptor})()
    shuffled = type(
        "Output", (), {"rgb32": target.roll(1, 0), "descriptor": descriptor.roll(1, 0)}
    )()
    exact_metrics = color_metrics(exact, target, descriptor, target.mean(0, keepdim=True))
    shuffled_metrics = color_metrics(shuffled, target, descriptor, target.mean(0, keepdim=True))
    assert exact_metrics["rgb_mae"] == 0
    assert exact_metrics["descriptor_top1"] == 1
    assert exact_metrics["palette_pearson"] > shuffled_metrics["palette_pearson"]
    assert exact_metrics["rgb_gain_vs_train_mean"] > shuffled_metrics["rgb_gain_vs_train_mean"]


def test_ycbcr_separates_gray_luminance_from_chroma():
    gray = torch.full((2, 3, 8, 8), 0.3)
    ycbcr = rgb_to_ycbcr(gray)
    torch.testing.assert_close(ycbcr[:, :1], torch.full((2, 1, 8, 8), 0.3))
    torch.testing.assert_close(ycbcr[:, 1:], torch.full((2, 2, 8, 8), 0.5))
