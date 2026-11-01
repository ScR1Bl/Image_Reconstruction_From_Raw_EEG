import torch

from eeg_mae.models.advanced_color import advanced_color_loss
from eeg_mae.models.color import rgb_to_oklab
from eeg_mae.models.color_specialist import EEGColorSpecialist


def tiny_model():
    return EEGColorSpecialist(
        encoder_config={
            "subjects": 3,
            "channels": 4,
            "input_samples": 20,
            "post_stimulus_start": 4,
            "branch_width": 4,
            "dim": 32,
            "heads": 4,
            "depth": 1,
            "style_tokens": 4,
            "descriptor_dim": 63,
            "dropout": 0,
        },
        refiner_config={
            "style_dim": 32,
            "width": 32,
            "heads": 4,
            "style_tokens": 4,
            "chroma_bins": 8,
            "output_size": 32,
        },
    )


def test_color_specialist_preserves_baseline_luminance_and_has_gradients():
    torch.manual_seed(9)
    model = tiny_model()
    eeg = torch.randn(2, 3, 4, 20)
    baseline = torch.rand(2, 3, 16, 16)
    target = torch.rand(2, 3, 32, 32)
    output = model(eeg, baseline)
    loss, _ = advanced_color_loss(output.image, target)
    loss.backward()
    expected = rgb_to_oklab(
        torch.nn.functional.interpolate(baseline, (32, 32), mode="bilinear", align_corners=False)
    )[:, :1]
    torch.testing.assert_close(output.image.oklab[:, :1], expected)
    assert output.image.rgb.shape == (2, 3, 32, 32)
    assert output.condition.style_tokens.shape == (2, 4, 32)
    assert output.condition.subject_weights.shape == (2, 3)
    assert output.alpha16.max() < 0.1
    assert model.encoder.branches[0][0].weight.grad is not None
    assert model.refiner.palette_head[-1].weight.grad is not None


def test_hard_specialist_output_is_finite():
    model = tiny_model().eval()
    output = model(torch.randn(1, 3, 4, 20), torch.rand(1, 3, 16, 16), hard_chroma=True)
    assert torch.isfinite(output.image.rgb).all()
    assert output.image.rgb.min() >= 0 and output.image.rgb.max() <= 1
