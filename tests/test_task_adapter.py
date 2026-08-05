import torch
from torch.nn import functional as F

from eeg_mae.color import ColorOutput, color_sample_errors
from eeg_mae.pyramid_edge import PyramidEdgeDecoder, pyramid_edge_sample_errors
from eeg_mae.task_adapter import TaskResidualAdapter, improvement_loss, shuffled_ranking_loss


def test_adapter_starts_as_exact_normalized_baseline_and_uniform_subjects():
    torch.manual_seed(1)
    base = F.normalize(torch.randn(2, 8, 8, 24), dim=-1)
    contexts = torch.randn(2, 3, 5, 24)
    global_feature = F.normalize(torch.randn(2, 32), dim=-1)
    model = TaskResidualAdapter(24, 24, 3, 4, 1, 0.0, 32, 7).eval()
    output = model(base, contexts, global_feature)
    assert torch.allclose(output.grid, base, atol=1e-6)
    assert torch.allclose(output.global_feature, global_feature, atol=1e-6)
    assert torch.allclose(output.subject_weights, torch.full((2, 3), 1 / 3), atol=1e-6)


def test_adapter_delta_heads_receive_gradient():
    model = TaskResidualAdapter(24, 24, 3, 4, 1, 0.0)
    output = model(F.normalize(torch.randn(2, 8, 8, 24), dim=-1), torch.randn(2, 3, 5, 24))
    output.grid[..., 0].mean().backward()
    assert model.grid_delta_head[-1].weight.grad.abs().sum() > 0


def test_improvement_and_ranking_losses_reward_correct_direction():
    baseline = torch.tensor([0.5, 0.5])
    assert improvement_loss(torch.tensor([0.3, 0.3]), baseline, 0.01) < improvement_loss(
        torch.tensor([0.6, 0.6]), baseline, 0.01
    )
    assert shuffled_ranking_loss(torch.tensor([0.2]), torch.tensor([0.6]), 0.1) == 0
    assert shuffled_ranking_loss(torch.tensor([0.6]), torch.tensor([0.2]), 0.1) > 0


def test_color_sample_error_is_zero_for_exact_render():
    rgb = torch.rand(2, 3, 32, 32)
    from eeg_mae.color import color_descriptor

    error, _ = color_sample_errors(
        ColorOutput(rgb, color_descriptor(rgb)), rgb, color_descriptor(rgb)
    )
    assert error.max() < 1e-6


def test_edge_per_sample_error_has_batch_shape():
    decoder = PyramidEdgeDecoder(input_dim=24, widths=(32, 24, 16, 8))
    output = decoder(torch.randn(2, 8, 8, 24))
    target = {
        "shape16": torch.rand(2, 1, 16, 16),
        "distance16": torch.rand(2, 1, 16, 16),
        "semantic32": torch.rand(2, 1, 32, 32),
        "perceptual64": torch.rand(2, 1, 64, 64),
        "crisp64": torch.rand(2, 1, 64, 64),
        "confidence64": torch.rand(2, 1, 64, 64),
        "uncertainty64": torch.rand(2, 1, 64, 64),
        "orientation64": F.normalize(torch.rand(2, 2, 64, 64), dim=1),
    }
    error, parts = pyramid_edge_sample_errors(output, target)
    assert error.shape == (2,)
    assert all(value.shape == (2,) for value in parts.values())
