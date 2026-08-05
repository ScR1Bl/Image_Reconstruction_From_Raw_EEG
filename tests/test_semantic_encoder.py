from __future__ import annotations

import torch

from eeg_mae.evaluation.official_200way_benchmark import LinearEEG
from eeg_mae.models.semantic_encoder import SubjectAdaptiveSemanticEncoder
from eeg_mae.training.train_semantic_encoder import semantic_loss


def test_encoder_accepts_full_and_cropped_windows():
    model = SubjectAdaptiveSemanticEncoder(output_dim=32, subjects=3, temporal_features=8)
    subject = torch.tensor([0, 2])
    assert model(torch.randn(2, 17, 100), subject).shape == (2, 32)
    assert model(torch.randn(2, 17, 80), subject).shape == (2, 32)


def test_zero_residual_is_exact_subject_linear_baseline():
    torch.manual_seed(4)
    model = SubjectAdaptiveSemanticEncoder(output_dim=24, subjects=2, temporal_features=8)
    baselines = [LinearEEG(80, 24) for _ in range(2)]
    with torch.no_grad():
        for subject, baseline in enumerate(baselines):
            model.input_norm_weight[subject].copy_(baseline.norm.weight)
            model.input_norm_bias[subject].copy_(baseline.norm.bias)
            model.linear_weight[subject].copy_(baseline.project.weight)
            model.linear_bias[subject].copy_(baseline.project.bias)
    eeg = torch.randn(6, 17, 80)
    subject = torch.tensor([0, 1, 0, 1, 1, 0])
    expected = torch.stack([baselines[int(s)](x[None])[0] for x, s in zip(eeg, subject)])
    torch.testing.assert_close(model(eeg, subject), expected, atol=1e-6, rtol=1e-5)


def test_semantic_loss_rewards_matching_pairs():
    torch.manual_seed(5)
    target = torch.randn(16, 32)
    matched = semantic_loss(target, target)[0]
    shuffled = semantic_loss(target.roll(1, 0), target)[0]
    assert matched < shuffled
