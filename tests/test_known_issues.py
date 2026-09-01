"""Documented KNOWN BUGS in the legacy code (Step 1b). These tests assert the
buggy behavior on purpose: they pass today, and will flip (fail) the moment
someone fixes the legacy path — which is the desired alarm. Do not "fix" a test
here without reading its docstring.
"""

import torch

from eeg_mae.models.semantic_edge import EEGDINOGridPredictor
from eeg_mae.models.semantic_encoder import SubjectAdaptiveSemanticEncoder

TRUNK_CONFIG = dict(
    output_dim=32,
    subjects=10,
    channels=17,
    samples=8,
    input_samples=10,
    post_stimulus_start=2,
    temporal_features=8,
    dropout=0.0,
)
PREDICTOR_CONFIG = dict(grid_dim=16, dim=16, heads=2, depth=1, context_tokens=2, dropout=0.0)


def _ensemble_with_true_ids(predictor, stacked, subject_ids):
    """Reference computation of forward_ensemble with CORRECT identity indexing."""
    batch, count = stacked.shape[:2]
    flat = stacked.flatten(0, 1)
    ids = subject_ids.repeat(batch)
    context = predictor.encode_context(flat, ids).reshape(
        batch, count, 2 * predictor.context_tokens, predictor.dim
    )
    return predictor.decode_context(context.mean(1))


def test_forward_ensemble_misindexes_nonprefix_subset_KNOWN_BUG():
    """semantic_edge.py:110: forward_ensemble derives subject ids from POSITION
    (``torch.arange(subjects)``), not identity. A non-prefix subset of subjects —
    e.g. (2, 5, 9) stacked as a 3-subject ensemble — is silently indexed with the
    per-subject banks of subjects (0, 1, 2). This test documents the bug: the
    buggy output must DIFFER from the correct identity-indexed computation, while
    a prefix subset (0, 1, 2) must agree. The Step 1b ensemble path indexes by ID
    and is not affected; the legacy path stays untouched by decision (2026-09-01).
    """
    torch.manual_seed(3)
    trunk = SubjectAdaptiveSemanticEncoder(**TRUNK_CONFIG)
    predictor = EEGDINOGridPredictor(trunk, **PREDICTOR_CONFIG).eval()
    eeg = torch.randn(2, 10, 17, 10)

    with torch.no_grad():
        prefix = eeg[:, :3]
        buggy_prefix, _ = predictor.forward_ensemble(prefix)
        correct_prefix = _ensemble_with_true_ids(predictor, prefix, torch.tensor([0, 1, 2]))
        torch.testing.assert_close(buggy_prefix, correct_prefix, atol=1e-6, rtol=0)

        subset = eeg[:, [2, 5, 9]]
        buggy_subset, _ = predictor.forward_ensemble(subset)
        correct_subset = _ensemble_with_true_ids(predictor, subset, torch.tensor([2, 5, 9]))
    assert not torch.allclose(buggy_subset, correct_subset, atol=1e-4), (
        "forward_ensemble zaczal poprawnie indeksowac podzbiory po ID - "
        "usun ten test-dokumentacje i zaktualizuj wpis o znanym bugu"
    )
