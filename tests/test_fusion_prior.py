import torch

from eeg_mae.models.fusion_prior import EEGFusionPrior


def inputs(batch=2):
    return (
        torch.randn(batch, 1024),
        {name: torch.randn(batch, 1024) for name in ("synclr", "clip_b32", "vae_sdxl")},
        torch.randn(batch, 16, 16, 192),
        torch.randn(batch, 16, 16, 192),
    )


def test_fusion_prior_shapes_and_gradient():
    model = EEGFusionPrior(width=64, heads=4, condition_depth=1, flow_depth=1, dropout=0)
    condition = model.encode_condition(*inputs())
    assert condition.tokens.shape == (2, 260, 64)
    assert condition.semantic_tokens.shape == (2, 4, 64)
    assert condition.spatial_tokens.shape == (2, 256, 64)
    assert condition.semantic_prediction.shape == (2, 1024)
    latent = torch.randn(2, 4, 16, 16)
    velocity = model.predict_velocity(latent, torch.tensor([0.2, 0.8]), condition)
    assert velocity.shape == latent.shape
    velocity.square().mean().backward()
    assert model.velocity_head[-1].weight.grad is not None


def test_fusion_prior_ablation_changes_condition_and_sampling_is_finite():
    model = EEGFusionPrior(width=32, heads=4, condition_depth=1, flow_depth=1, dropout=0).eval()
    values = inputs(batch=1)
    complete = model.encode_condition(*values)
    no_edges = model.encode_condition(*values, ablate="edge")
    no_conditions = model.encode_condition(*values, ablate="all")
    assert not torch.allclose(complete.tokens, no_edges.tokens)
    assert not torch.allclose(complete.tokens, no_conditions.tokens)
    noise = torch.randn(1, 4, 16, 16)
    result = model.sample(complete, steps=2, guidance=1.0, noise=noise)
    assert result.shape == noise.shape
    assert torch.isfinite(result).all()
