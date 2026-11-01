"""Step 1b: fusion-module invariants (CI-safe, CPU)."""

import torch

from eeg_mae.lightning.subject_fusion import AttentionSubjectPooling, MeanSubjectPooling


def test_attention_starts_as_exact_mean():
    """Zero-init score head + zero prior -> uniform weights == arithmetic mean
    (identity-at-init: the untrained attention variant IS the gate fusion)."""
    torch.manual_seed(0)
    fusion = AttentionSubjectPooling(dim=16, roster=10).eval()
    embeddings = torch.randn(4, 7, 16)
    ids = torch.tensor([0, 2, 3, 5, 6, 8, 9])
    fused, weights = fusion(embeddings, ids)
    torch.testing.assert_close(fused, embeddings.mean(1), atol=1e-6, rtol=0)
    torch.testing.assert_close(weights, torch.full((4, 7), 1 / 7), atol=1e-7, rtol=0)


def test_attention_is_permutation_invariant():
    torch.manual_seed(1)
    fusion = AttentionSubjectPooling(dim=16, roster=10).eval()
    with torch.no_grad():
        for parameter in fusion.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)  # odejscie od init
    embeddings = torch.randn(3, 5, 16)
    ids = torch.tensor([1, 4, 6, 7, 9])
    fused, _ = fusion(embeddings, ids)
    perm = torch.randperm(5)
    fused_perm, _ = fusion(embeddings[:, perm], ids[perm])
    torch.testing.assert_close(fused, fused_perm, atol=1e-6, rtol=0)


def test_fusion_accepts_any_subset_size():
    fusion_mean = MeanSubjectPooling()
    fusion_att = AttentionSubjectPooling(dim=8, roster=10).eval()
    for count in (1, 3, 10):
        embeddings = torch.randn(2, count, 8)
        ids = torch.randperm(10)[:count]
        for fusion in (fusion_mean, fusion_att):
            fused, weights = fusion(embeddings, ids)
            assert fused.shape == (2, 8)
            torch.testing.assert_close(weights.sum(1), torch.ones(2), atol=1e-6, rtol=0)
