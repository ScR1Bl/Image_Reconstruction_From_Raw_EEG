import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eeg_mae.training.evaluate_neural_geometry import fit_map, load_predictions


def test_oof_loader_rejects_in_sample_or_misaligned_predictions(tmp_path):
    bank = SimpleNamespace(
        rows=[
            {"image_index": i, "concept": "plane", "split": "train" if i < 6 else "val"}
            for i in range(8)
        ],
        protocol={"sha256": "test"},
        subject=0,
    )
    for fold in range(3):
        path = tmp_path / "oof" / "synclr" / f"fold_{fold}"
        path.mkdir(parents=True)
        (path / "report.json").write_text(
            json.dumps({"protocol_sha256": "test", "subject_id": 0, "oof_fold": fold})
        )
        (path / "config.json").write_text(
            json.dumps({"end_ms": 400, "task": "synclr", "init_ridge": None})
        )
        np.savez(
            path / "predictions_oof_r4.npz",
            rows=[fold, fold + 3],
            prediction=np.ones((2, 4), dtype=np.float32),
        )
    predictions = load_predictions(tmp_path, "synclr", bank, oof=True)
    assert set(predictions) == set(range(6))
    path = tmp_path / "oof" / "synclr" / "fold_0"
    np.savez(
        path / "predictions_oof_r4.npz", rows=[1, 3], prediction=np.ones((2, 4), dtype=np.float32)
    )
    with pytest.raises(ValueError, match="held out"):
        load_predictions(tmp_path, "synclr", bank, oof=True)


def test_conditional_probe_can_recover_a_held_out_linear_signal():
    torch.set_num_threads(2)
    rng = torch.Generator().manual_seed(22)
    x = torch.rand(64, 4, generator=rng)
    truth = 0.2 + x[:, :2] * 0.5
    output, _, state = fit_map(x[:48], truth[:48], x[48:], truth[48:])
    assert torch.mean((output - truth[48:]).square()) < 1e-6
    assert torch.isfinite(state[0]).all()


def test_dit_adapter_checkpoint_excludes_lazily_loaded_frozen_vae():
    L = pytest.importorskip("lightning")
    from eeg_mae.lightning.dit_adapter import DiTAdapterLightning

    # Exercise real serialization methods without downloading a diffusion model.
    model = DiTAdapterLightning.__new__(DiTAdapterLightning)
    L.LightningModule.__init__(model)
    model.transformer = torch.nn.Linear(3, 3)
    model._vae_model = torch.nn.Linear(3, 3)
    model.projection = torch.nn.Linear(3, 3)
    state = model.state_dict()
    assert set(state) == {"projection.weight", "projection.bias"}
    result = model.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    with pytest.raises(RuntimeError, match="missing"):
        model.load_state_dict({"projection.weight": state["projection.weight"]}, strict=True)
