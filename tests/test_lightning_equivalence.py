"""Step 1a equivalence: old modules and their Lightning shells produce
identical outputs from identical weights and inputs (CPU, eval, atol=1e-6).

Small random configurations keep this CI-safe (no artifact files needed);
real-weights parity is covered by the conversion smoke + the 200-way gate.
"""

import torch

from eeg_mae.lightning.color import ColorLightning
from eeg_mae.lightning.common import freeze
from eeg_mae.lightning.edges import EdgesLightning
from eeg_mae.lightning.eeg_encoder import SemanticEncoderLightning
from eeg_mae.models.color import ColorDecoder
from eeg_mae.models.color_specialist import EEGColorEncoderPipeline, EEGColorSpecialist
from eeg_mae.models.edge_specialist import EEGEdgeEncoderPipeline, EEGEdgeSpecialist
from eeg_mae.models.pyramid_edge import PyramidEdgeDecoder, PyramidEdgeOutput
from eeg_mae.models.semantic_edge import EEGDINOGridPredictor
from eeg_mae.models.semantic_encoder import SubjectAdaptiveSemanticEncoder
from eeg_mae.models.task_adapter import TaskResidualAdapter

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


def small_color_configs():
    specialist = dict(
        encoder_config=dict(
            subjects=10,
            channels=17,
            input_samples=10,
            post_stimulus_start=2,
            branch_width=4,
            dim=16,
            heads=2,
            depth=1,
            style_tokens=4,
            descriptor_dim=63,
            dropout=0.0,
        ),
        refiner_config=dict(
            style_dim=16, width=16, heads=2, style_tokens=4, chroma_bins=8, output_size=16
        ),
    )
    adapter = dict(
        grid_dim=16,
        context_dim=16,
        subjects=10,
        heads=2,
        depth=1,
        dropout=0.0,
        global_dim=32,
        descriptor_dim=63,
    )
    decoder = dict(grid_dim=16, global_dim=32, width=16, descriptor_dim=63)
    return specialist, adapter, decoder


def test_semantic_encoder_equivalence():
    torch.manual_seed(0)
    old = SubjectAdaptiveSemanticEncoder(**TRUNK_CONFIG).eval()
    new = SemanticEncoderLightning(model_config=TRUNK_CONFIG).eval()
    new.model.load_state_dict(old.state_dict(), strict=True)
    eeg = torch.randn(4, 17, 10)
    subject = torch.tensor([0, 3, 7, 9])
    with torch.no_grad():
        torch.testing.assert_close(new(eeg, subject), old(eeg, subject), atol=1e-6, rtol=0)


def test_color_pipeline_equivalence():
    torch.manual_seed(1)
    specialist_cfg, adapter_cfg, decoder_cfg = small_color_configs()
    trunk = SubjectAdaptiveSemanticEncoder(**TRUNK_CONFIG)
    old = EEGColorEncoderPipeline(
        EEGColorSpecialist(**specialist_cfg),
        EEGDINOGridPredictor(trunk, **PREDICTOR_CONFIG),
        TaskResidualAdapter(**adapter_cfg),
        ColorDecoder(**decoder_cfg),
    ).eval()
    new = ColorLightning(
        specialist_config=specialist_cfg,
        predictor_config=PREDICTOR_CONFIG,
        adapter_config=adapter_cfg,
        decoder_config=decoder_cfg,
        trunk_config=TRUNK_CONFIG,
    ).eval()
    new.pipeline.load_state_dict(old.state_dict(), strict=True)
    # Production inference semantics: the legacy loaders freeze the whole
    # pipeline, and requires_grad flags affect CPU kernel selection (~1e-6 ULP),
    # so equivalence is defined on identically frozen modules.
    freeze(old)
    freeze(new)
    eeg = torch.randn(2, 10, 17, 10)
    with torch.no_grad():
        old_out = old(eeg)
        new_out = new(eeg)
    for field in (
        "rgb",
        "oklab",
        "linear_rgb",
        "descriptor",
        "palette",
        "masks16",
        "chroma_logits16",
    ):
        torch.testing.assert_close(
            getattr(new_out.image, field), getattr(old_out.image, field), atol=1e-6, rtol=0
        )
    torch.testing.assert_close(new_out.alpha16, old_out.alpha16, atol=1e-6, rtol=0)
    torch.testing.assert_close(new_out.chroma_delta16, old_out.chroma_delta16, atol=1e-6, rtol=0)


def test_edges_pipeline_equivalence():
    torch.manual_seed(2)
    specialist_cfg = dict(
        encoder_config=dict(
            subjects=10,
            channels=17,
            input_samples=10,
            post_stimulus_start=2,
            branch_width=4,
            dim=16,
            heads=2,
            temporal_depth=1,
            spatial_depth=1,
            dropout=0.0,
        ),
        refiner_config=dict(input_dim=16, width=16),
    )
    decoder_cfg = dict(input_dim=16, widths=(16, 16, 8, 8))
    trunk = SubjectAdaptiveSemanticEncoder(**TRUNK_CONFIG)
    old = EEGEdgeEncoderPipeline(
        EEGEdgeSpecialist(**specialist_cfg),
        EEGDINOGridPredictor(trunk, **PREDICTOR_CONFIG),
        PyramidEdgeDecoder(**decoder_cfg),
    ).eval()
    new = EdgesLightning(
        specialist_config=specialist_cfg,
        predictor_config=PREDICTOR_CONFIG,
        decoder_config=decoder_cfg,
        trunk_config=TRUNK_CONFIG,
    ).eval()
    new.pipeline.load_state_dict(old.state_dict(), strict=True)
    freeze(old)
    freeze(new)
    eeg = torch.randn(2, 10, 17, 10)
    with torch.no_grad():
        old_out = old(eeg)
        new_out = new(eeg)
    for field in PyramidEdgeOutput.__dataclass_fields__:
        torch.testing.assert_close(
            getattr(new_out.pyramid, field), getattr(old_out.pyramid, field), atol=1e-6, rtol=0
        )
