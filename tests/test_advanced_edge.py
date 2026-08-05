import torch
from torch.nn import functional as F

from eeg_mae.advanced_edge import DINOEdgeOracleV2, PixelDifferenceConv, advanced_edge_loss


def edge_targets(batch=2):
    return {
        "shape16": torch.rand(batch, 1, 16, 16),
        "distance16": torch.rand(batch, 1, 16, 16),
        "semantic32": torch.rand(batch, 1, 32, 32),
        "perceptual64": torch.rand(batch, 1, 64, 64),
        "crisp64": torch.rand(batch, 1, 64, 64),
        "confidence64": torch.rand(batch, 1, 64, 64),
        "uncertainty64": torch.rand(batch, 1, 64, 64),
        "orientation64": F.normalize(torch.rand(batch, 2, 64, 64), dim=1),
    }


def test_pixel_difference_rejects_constant_interior():
    layer = PixelDifferenceConv(4)
    output = layer(torch.ones(1, 4, 12, 12))
    assert output[:, :, 2:-2, 2:-2].abs().max() < 1e-5


def test_advanced_edge_shapes_loss_and_gradients():
    torch.manual_seed(7)
    model = DINOEdgeOracleV2(
        input_dim=24,
        layers=3,
        width=40,
        heads=5,
        transformer_depth=1,
        refinement_steps=2,
        dropout=0,
    )
    output = model(torch.randn(2, 3, 16, 16, 24), torch.randn(2, 3, 24), mask_ratio=0.3)
    value = output.pyramid
    loss, parts = advanced_edge_loss(output, edge_targets())
    loss.backward()
    assert value.shape_logits16.shape == (2, 1, 16, 16)
    assert value.semantic_logits32.shape == (2, 1, 32, 32)
    assert value.crisp_logits64.shape == (2, 1, 64, 64)
    assert value.orientation64.shape == (2, 2, 64, 64)
    assert len(output.side_crisp64) == 2
    assert model.local_difference[0].weight.grad is not None
    assert torch.isfinite(loss) and "topology" in parts and "frequency" in parts


def test_edge_eval_refinement_is_deterministic():
    model = DINOEdgeOracleV2(
        input_dim=16,
        layers=2,
        width=40,
        heads=5,
        transformer_depth=1,
        refinement_steps=2,
        dropout=0,
    ).eval()
    patches = torch.randn(1, 2, 16, 16, 16)
    cls = torch.randn(1, 2, 16)
    first = model(patches, cls).pyramid.crisp_logits64
    second = model(patches, cls).pyramid.crisp_logits64
    torch.testing.assert_close(first, second)
