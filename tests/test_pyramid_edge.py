from __future__ import annotations

import torch

from eeg_mae.data.build_pyramid_edge_targets import derive_targets
from eeg_mae.models.pyramid_edge import PyramidEdgeDecoder, pyramid_edge_loss


def test_target_derivation_produces_true_resolution_pyramid():
    semantic = torch.zeros(2, 1, 64, 64)
    semantic[:, :, 15:49, 15] = 1
    semantic[:, :, 15:49, 48] = 1
    semantic[:, :, 15, 15:49] = 1
    semantic[:, :, 48, 15:49] = 1
    distance = torch.zeros_like(semantic)
    distance[:, :, 16:48, 16:48] = 0.75
    perceptual = semantic.clone()
    geometry = semantic.clone()
    target = derive_targets(semantic, distance, perceptual, geometry)
    assert target["shape16"].shape == (2, 1, 16, 16)
    assert target["semantic32"].shape == (2, 1, 32, 32)
    assert target["crisp64"].shape == (2, 1, 64, 64)
    assert target["orientation64"].shape == (2, 2, 64, 64)
    assert target["crisp64"].sum() > 0


def test_pyramid_decoder_is_deeply_supervised_and_trainable():
    model = PyramidEdgeDecoder(input_dim=24, widths=(48, 32, 24, 16))
    grid = torch.randn(2, 8, 8, 24)
    semantic = torch.rand(2, 1, 64, 64)
    distance = torch.rand(2, 1, 64, 64)
    target = derive_targets(semantic, distance, semantic, semantic)
    output = model(grid)
    loss, parts = pyramid_edge_loss(output, target)
    loss.backward()
    assert output.shape_logits16.shape == (2, 1, 16, 16)
    assert output.semantic_logits32.shape == (2, 1, 32, 32)
    assert output.crisp_logits64.shape == (2, 1, 64, 64)
    assert torch.isfinite(loss)
    assert model.stem[0].weight.grad is not None
    assert "orientation" in parts
