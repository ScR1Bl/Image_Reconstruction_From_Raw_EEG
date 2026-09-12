import pytest
import torch

from eeg_mae.data.single_subject import make_protocol
from eeg_mae.evaluation.geometry_metrics import (
    geometry_metrics,
    retrieval_metrics,
    same_category_2afc,
)


def test_protocol_is_reproducible_and_grouped_by_image():
    rows = [
        {
            "filename": f"{c}_{i:02}.jpg",
            "image_index": 10 * n + i,
            "position": 10 * n + i,
            "split": "train",
        }
        for n, c in enumerate(("air_plane", "chair"))
        for i in range(10)
    ]
    a, b = make_protocol(rows), make_protocol(list(reversed(rows)))
    assert a == b
    for concept in ("air_plane", "chair"):
        subset = [r for r in a["rows"] if r["concept"] == concept]
        assert [sum(r["split"] == s for r in subset) for s in ("train", "val", "test")] == [6, 2, 2]
    assert not a["sequence_disjoint_verified"]
    with pytest.raises(ValueError, match="duplicate"):
        make_protocol(rows + rows[:1])


def test_constant_retrieval_is_chance_not_perfect():
    metrics = retrieval_metrics(torch.zeros(10, 4), torch.randn(10, 4), chunk_size=3)
    assert metrics["top1"] == pytest.approx(0.1)
    assert metrics["top5"] == pytest.approx(0.5)
    expected_mrr = sum(1 / i for i in range(1, 11)) / 10
    assert metrics["mrr"] == pytest.approx(expected_mrr)


def test_geometry_distinguishes_same_category_locations_and_ties():
    target = torch.zeros(2, 1, 16, 16)
    target[0, 0, 3:13, 2:5] = 1
    target[1, 0, 3:13, 11:14] = 1
    metrics = geometry_metrics(target, target, ["plane", "plane"])
    assert metrics["within_category_2afc"] == 1
    assert metrics["centroid_error"] == 0
    assert metrics["axis_error_degrees"] == 0
    wrong = geometry_metrics(target.flip(0), target, ["plane", "plane"])
    assert wrong["within_category_2afc"] == 0
    mean = target.mean(0, keepdim=True).expand_as(target)
    assert same_category_2afc(mean, target, ["plane", "plane"])["accuracy"] == 0.5
