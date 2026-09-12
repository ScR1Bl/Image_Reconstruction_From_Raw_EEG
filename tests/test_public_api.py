from eeg_mae import (
    EEGColorEncoder,
    EEGEdgeEncoder,
    EEGHybridColorEncoder,
    SubjectAdaptiveSemanticEncoder,
    load_color_encoder,
    load_edge_encoder,
    load_hybrid_color_encoder,
)


def test_final_models_have_a_small_public_api():
    assert EEGColorEncoder.__name__ == "EEGColorEncoderPipeline"
    assert EEGEdgeEncoder.__name__ == "EEGEdgeEncoderPipeline"
    assert EEGHybridColorEncoder.__name__ == "EEGHybridColorEncoder"
    assert SubjectAdaptiveSemanticEncoder.__name__ == "SubjectAdaptiveSemanticEncoder"
    assert callable(load_color_encoder)
    assert callable(load_edge_encoder)
    assert callable(load_hybrid_color_encoder)
