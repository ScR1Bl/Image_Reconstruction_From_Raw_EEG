import torch

from eeg_mae.edge_specialist import EEGEdgeSpecialist
from eeg_mae.pyramid_edge import PyramidEdgeOutput, pyramid_edge_loss


def baseline(batch=2):
    return PyramidEdgeOutput(
        torch.randn(batch, 1, 16, 16),
        torch.rand(batch, 1, 16, 16),
        torch.randn(batch, 1, 32, 32),
        torch.randn(batch, 1, 64, 64),
        torch.randn(batch, 1, 64, 64),
        torch.rand(batch, 1, 64, 64),
        torch.rand(batch, 1, 64, 64),
        torch.nn.functional.normalize(torch.randn(batch, 2, 64, 64), dim=1),
    )


def targets(batch=2):
    return {
        "shape16": torch.rand(batch, 1, 16, 16),
        "distance16": torch.rand(batch, 1, 16, 16),
        "semantic32": torch.rand(batch, 1, 32, 32),
        "perceptual64": torch.rand(batch, 1, 64, 64),
        "crisp64": torch.rand(batch, 1, 64, 64),
        "confidence64": torch.rand(batch, 1, 64, 64),
        "uncertainty64": torch.rand(batch, 1, 64, 64),
        "orientation64": torch.nn.functional.normalize(torch.randn(batch, 2, 64, 64), dim=1),
    }


def tiny_model():
    return EEGEdgeSpecialist(
        {
            "subjects": 3,
            "channels": 4,
            "input_samples": 20,
            "post_stimulus_start": 4,
            "branch_width": 4,
            "dim": 32,
            "heads": 4,
            "temporal_depth": 1,
            "spatial_depth": 1,
            "dropout": 0,
        },
        {"input_dim": 32, "width": 32},
    )


def test_edge_specialist_starts_as_exact_baseline_and_trains():
    model = tiny_model()
    base = baseline()
    result = model(torch.randn(2, 3, 4, 20), base)
    for field in PyramidEdgeOutput.__dataclass_fields__:
        torch.testing.assert_close(
            getattr(result.pyramid, field), getattr(base, field), atol=1e-4, rtol=1e-4
        )
    loss, _ = pyramid_edge_loss(result.pyramid, targets())
    loss.backward()
    assert model.encoder.branches[0][0].weight.grad is not None
    assert model.refiner.head64.weight.grad is not None
    assert result.condition.grid16.shape == (2, 16, 16, 32)


def test_edge_specialist_rejects_incomplete_subject_ensemble():
    model = tiny_model()
    try:
        model(torch.randn(1, 2, 4, 20), baseline(1))
    except ValueError as error:
        assert "3" in str(error)
    else:
        raise AssertionError("incomplete EEG ensemble was accepted")
