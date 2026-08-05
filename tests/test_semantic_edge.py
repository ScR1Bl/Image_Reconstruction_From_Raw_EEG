from __future__ import annotations

import torch

from eeg_mae.models.semantic_edge import EEGDINOGridPredictor
from eeg_mae.models.semantic_encoder import SubjectAdaptiveSemanticEncoder


def test_eeg_grid_predictor_uses_frozen_semantic_encoder():
    encoder = SubjectAdaptiveSemanticEncoder(output_dim=24, subjects=2, temporal_features=8)
    predictor = EEGDINOGridPredictor(
        encoder, grid_dim=16, dim=48, heads=4, depth=1, context_tokens=2
    )
    output = predictor(torch.randn(3, 17, 100), torch.tensor([0, 1, 0]))
    assert output.shape == (3, 8, 8, 16)
    output.mean().backward()
    assert predictor.spatial_weight.grad is not None
    assert all(parameter.grad is None for parameter in encoder.parameters())


def test_eeg_grid_predictor_fuses_subjects_before_decoding():
    encoder = SubjectAdaptiveSemanticEncoder(output_dim=24, subjects=2, temporal_features=8)
    predictor = EEGDINOGridPredictor(
        encoder, grid_dim=16, dim=48, heads=4, depth=1, context_tokens=2
    )
    eeg = torch.randn(3, encoder.subjects, 17, 100)
    grid, contexts = predictor.forward_ensemble(eeg)
    assert grid.shape == (3, 8, 8, 16)
    assert contexts.shape == (3, encoder.subjects, 4, 48)
