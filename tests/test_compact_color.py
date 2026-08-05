import torch

from eeg_mae.advanced_color import advanced_color_loss
from eeg_mae.compact_color import (
    CompactColorAutoencoder,
    CompactColorTokens,
    RichDINOColorTokenizer,
)


def test_compact_color_uses_explicit_scene_and_color_bottleneck():
    torch.manual_seed(8)
    model = CompactColorAutoencoder(
        input_dim=24,
        layers=3,
        width=48,
        heads=4,
        tokenizer_depth=1,
        color_tokens=7,
        chroma_bins=12,
        output_size=32,
        dropout=0,
    )
    patches = torch.randn(2, 3, 16, 16, 24)
    cls = torch.randn(2, 3, 24)
    target = torch.rand(2, 3, 32, 32)
    forward = model(patches, cls, token_noise=0.01, token_dropout=0.02)
    loss, _ = advanced_color_loss(forward.image, target)
    loss.backward()
    assert forward.tokens.scene.shape == (2, 8, 8, 48)
    assert forward.tokens.color.shape == (2, 7, 48)
    assert forward.image.rgb.shape == (2, 3, 32, 32)
    assert forward.scene_lightness.shape == (2, 1, 16, 16)
    assert model.tokenizer.color_queries.grad is not None
    assert model.renderer.palette_head[-1].weight.grad is not None
    assert torch.isfinite(loss)


def test_renderer_can_decode_tokens_without_access_to_dino():
    model = CompactColorAutoencoder(
        input_dim=16,
        layers=2,
        width=32,
        heads=4,
        tokenizer_depth=1,
        color_tokens=4,
        chroma_bins=8,
        output_size=32,
        dropout=0,
    ).eval()
    tokens = CompactColorTokens(torch.randn(1, 8, 8, 32), torch.randn(1, 4, 32))
    soft, _ = model.decode(tokens)
    hard, _ = model.decode(tokens, hard_chroma=True)
    assert soft.rgb.shape == hard.rgb.shape == (1, 3, 32, 32)
    assert torch.isfinite(soft.rgb).all() and torch.isfinite(hard.rgb).all()
    assert soft.rgb.min() >= 0 and soft.rgb.max() <= 1


def test_tokenizer_rejects_wrong_spatial_shape():
    tokenizer = RichDINOColorTokenizer(
        input_dim=16, layers=2, width=32, heads=4, depth=1, color_tokens=4
    )
    try:
        tokenizer(torch.randn(1, 2, 8, 8, 16), torch.randn(1, 2, 16))
    except ValueError as error:
        assert "16,16" in str(error)
    else:
        raise AssertionError("invalid DINO grid was accepted")
