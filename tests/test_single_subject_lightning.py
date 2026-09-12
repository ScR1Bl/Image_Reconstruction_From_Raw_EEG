import pytest
import torch

pytest.importorskip("lightning")

from eeg_mae.lightning.single_subject import SingleSubjectLightning
from eeg_mae.training.train_fusion_prior import gradient_loss
from eeg_mae.training.train_geometry_probes import fit_from_basis, prepare_eeg, ridge_basis


@pytest.mark.parametrize("task", ["dino", "edge", "color"])
def test_single_subject_forward_gradients_and_checkpoint(task):
    torch.set_num_threads(2)
    model = SingleSubjectLightning(task=task, end_ms=200, compact=True, protocol_sha256="test")
    eeg = torch.randn(2, 17, 40)
    output = model(eeg)
    value = (
        output
        if task == "dino"
        else output.pyramid.shape_logits16
        if task == "edge"
        else output.image.rgb
    )
    value.square().mean().backward()
    assert any(p.grad is not None for p in model.model.parameters())
    assert all(p.grad is None for p in model.ema.parameters())
    restored = SingleSubjectLightning(task=task, end_ms=200, compact=True, protocol_sha256="test")
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval()
    restored.eval()
    expected = model(eeg, use_ema=True)
    actual = restored(eeg, use_ema=True)
    if task == "edge":
        expected, actual = expected.condition.grid16, actual.condition.grid16
    elif task == "color":
        expected, actual = expected.image.rgb, actual.image.rgb
    torch.testing.assert_close(expected, actual)
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    restored.on_load_checkpoint(checkpoint)
    checkpoint["protocol_sha256"] = "other"
    with pytest.raises(ValueError, match="protocol"):
        restored.on_load_checkpoint(checkpoint)


def test_gradient_loss_ignores_constant_offset_but_detects_changed_edges():
    target = torch.randn(2, 4, 8, 8)
    assert gradient_loss(target + 3, target) < 1e-12
    changed = target.clone()
    changed[..., 4:, :] += 1
    assert gradient_loss(changed, target) > 0


def test_ridge_can_initialize_existing_semantic_linear_path(tmp_path):
    torch.manual_seed(21)
    eeg = torch.randn(12, 17, 40)
    x = prepare_eeg(eeg[:, :, 20:])
    weight, bias = fit_from_basis(ridge_basis(x), torch.randn(12, 384), 10)
    checkpoint = tmp_path / "ridge.pt"
    torch.save(
        {
            "target": "dino",
            "subject_id": 0,
            "end_ms": 200,
            "protocol_sha256": "test",
            "weight": weight,
            "bias": bias,
        },
        checkpoint,
    )
    model = SingleSubjectLightning(task="dino", end_ms=200, protocol_sha256="test").eval()
    model.initialize_ridge(checkpoint)
    torch.testing.assert_close(
        model(eeg), torch.nn.functional.linear(x, weight, bias), rtol=1e-4, atol=1e-5
    )
