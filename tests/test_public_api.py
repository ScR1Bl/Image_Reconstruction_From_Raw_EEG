from eeg_mae import (
    EEGColorEncoder,
    EEGEdgeEncoder,
    SubjectAdaptiveSemanticEncoder,
    load_color_encoder,
    load_edge_encoder,
)


def test_final_models_have_a_small_public_api():
    assert EEGColorEncoder.__name__ == "EEGColorEncoderPipeline"
    assert EEGEdgeEncoder.__name__ == "EEGEdgeEncoderPipeline"
    assert SubjectAdaptiveSemanticEncoder.__name__ == "SubjectAdaptiveSemanticEncoder"
    assert callable(load_color_encoder)
    assert callable(load_edge_encoder)
