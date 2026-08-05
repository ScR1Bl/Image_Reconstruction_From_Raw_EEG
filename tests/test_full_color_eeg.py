import torch

from eeg_mae.full_color_eeg import EEGFullColorLatentPredictor
from eeg_mae.semantic_encoder import SubjectAdaptiveSemanticEncoder


def tiny_encoder():
    return SubjectAdaptiveSemanticEncoder(
        output_dim=32,
        subjects=3,
        channels=4,
        samples=16,
        input_samples=20,
        post_stimulus_start=4,
        temporal_features=8,
        dropout=0,
    )


def test_raw_eeg_predicts_full_grid_and_global_latent():
    model = EEGFullColorLatentPredictor(
        tiny_encoder(),
        latent_dim=24,
        dim=32,
        heads=4,
        depth=1,
        context_tokens=2,
        mixer_depth=1,
        dropout=0,
    )
    eeg = torch.randn(2, 3, 4, 20)
    output = model.forward_ensemble(eeg)
    assert output.grid.shape == (2, 16, 16, 24)
    assert output.global_feature.shape == (2, 24)
    assert output.contexts.shape == (2, 3, 4, 32)
    loss = output.grid.square().mean() + output.global_feature.square().mean()
    loss.backward()
    assert model.grid_output[-1].weight.grad is not None
    assert all(parameter.grad is None for parameter in model.encoder.parameters())


def test_full_predictor_requires_all_subjects_for_ensemble():
    model = EEGFullColorLatentPredictor(
        tiny_encoder(),
        latent_dim=16,
        dim=32,
        heads=4,
        depth=1,
        context_tokens=2,
        mixer_depth=1,
        dropout=0,
    )
    try:
        model.forward_ensemble(torch.randn(1, 2, 4, 20))
    except ValueError as error:
        assert "3" in str(error)
    else:
        raise AssertionError("incomplete subject ensemble was accepted")
