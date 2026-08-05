from __future__ import annotations

import zipfile

import torch

from eeg_mae.build_official_test_targets import ordered_test_members
from eeg_mae.official_200way_benchmark import (
    LinearEEG,
    TSConvEEG,
    retrieval_metrics,
    shuffled_metrics,
)


def test_official_member_order_is_numeric(tmp_path):
    path = tmp_path / "test.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for index in reversed(range(1, 201)):
            archive.writestr(f"test_images/{index:05d}_concept/image_{index}.jpg", b"x")
    with zipfile.ZipFile(path) as archive:
        members = ordered_test_members(archive)
    assert "00001_concept" in members[0]
    assert "00200_concept" in members[-1]


def test_retrieval_metrics_detect_identity_and_shuffle():
    generator = torch.Generator().manual_seed(9)
    target = torch.randn(20, 32, generator=generator)
    perfect, scores = retrieval_metrics(target, target)
    assert perfect["top1"] == 1.0
    shuffled = shuffled_metrics(scores, permutations=200, seed=4)
    assert shuffled["top1_mean"] < 0.12
    assert 0.15 < shuffled["top5_mean"] < 0.40


def test_baseline_shapes_for_poststimulus_window():
    eeg = torch.randn(3, 17, 80)
    assert LinearEEG(80, 384)(eeg).shape == (3, 384)
    assert TSConvEEG(80, 384)(eeg).shape == (3, 384)
    assert LinearEEG(100, 384)(torch.randn(3, 17, 100)).shape == (3, 384)
