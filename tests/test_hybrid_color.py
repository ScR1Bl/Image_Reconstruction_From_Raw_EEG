import pytest
import torch

from eeg_mae.models.hybrid_color import EEGHybridColorEncoder, load_hybrid_color_encoder
from eeg_mae.training.train_hybrid_color_encoder import safe_routing_terms, spatial_color_terms


def tiny_model():
    encoder = {
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
    }
    specialist = {
        "encoder_config": encoder,
        "refiner_config": {
            "style_dim": 32,
            "width": 32,
            "heads": 4,
            "style_tokens": 4,
            "chroma_bins": 8,
            "output_size": 32,
        },
    }
    return EEGHybridColorEncoder(
        direct_config=specialist,
        semantic_config={
            "output_dim": 32,
            "subjects": 3,
            "channels": 4,
            "samples": 16,
            "input_samples": 20,
            "post_stimulus_start": 4,
            "temporal_features": 4,
            "dropout": 0,
        },
        predictor_config={
            "grid_dim": 32,
            "dim": 32,
            "heads": 4,
            "depth": 1,
            "context_tokens": 2,
            "dropout": 0,
        },
        adapter_config={
            "grid_dim": 32,
            "context_dim": 32,
            "subjects": 3,
            "heads": 4,
            "depth": 1,
            "dropout": 0,
            "global_dim": 32,
            "descriptor_dim": 63,
        },
        prior_decoder_config={
            "grid_dim": 32,
            "global_dim": 32,
            "width": 32,
            "descriptor_dim": 63,
        },
        legacy_specialist_config=specialist,
        fusion_config={
            "direct_dim": 32,
            "prior_dim": 32,
            "width": 32,
            "heads": 4,
            "chroma_bins": 8,
            "output_size": 32,
            "dropout": 0,
        },
    )


def test_hybrid_routes_each_oklab_channel_over_three_experts():
    torch.manual_seed(4)
    model = tiny_model()
    output = model(torch.randn(2, 3, 4, 20))
    assert output.image.rgb.shape == (2, 3, 32, 32)
    assert output.expert_weights.shape == (2, 3, 3, 32, 32)
    torch.testing.assert_close(
        output.expert_weights.sum(2), torch.ones(2, 3, 32, 32), atol=1e-6, rtol=1e-6
    )
    envelope = output.proposal_oklab.new_tensor([0.08, 0.06, 0.06])[None, :, None, None]
    lower = torch.minimum(output.legacy_image.oklab, output.direct_image.oklab) - envelope
    upper = torch.maximum(output.legacy_image.oklab, output.direct_image.oklab) + envelope
    assert torch.all(output.proposal_oklab >= lower - 1e-6)
    assert torch.all(output.proposal_oklab <= upper + 1e-6)
    output.image.rgb.mean().backward()
    assert model.fusion.router_head.weight.grad is not None
    assert model.fusion.direct_to_prior.attn.in_proj_weight.grad is not None


def test_spatial_color_terms_reward_exact_color_locations():
    target = torch.zeros(2, 3, 32, 32)
    target[:, 0, 8:16, 4:12] = 1
    exact = spatial_color_terms(target, target)
    shifted = spatial_color_terms(target.roll(12, -1), target)
    assert exact["salient_chroma_mae"].mean() < shifted["salient_chroma_mae"].mean()
    assert exact["chroma_centroid_error"].mean() < shifted["chroma_centroid_error"].mean()


def test_safe_router_loss_is_finite_and_trains_router_and_proposal():
    model = tiny_model()
    output = model(torch.randn(2, 3, 4, 20))
    terms = safe_routing_terms(output, torch.rand(2, 3, 32, 32))
    loss = sum(terms.values())
    assert torch.isfinite(loss)
    loss.backward()
    assert model.fusion.router_head.weight.grad is not None
    assert model.fusion.proposal_head.weight.grad is not None


def test_production_loader_rejects_checkpoint_from_old_objective(tmp_path):
    torch.save(
        {
            "objective_version": 1,
            "record": {"qualified": True},
            "architecture": "independent_hybrid_color_v1",
        },
        tmp_path / "best_qualified.pt",
    )
    with pytest.raises(RuntimeError, match="safely qualified v2"):
        load_hybrid_color_encoder(tmp_path)
