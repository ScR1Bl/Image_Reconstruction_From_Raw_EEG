"""Public API for the EEG color and edge encoders."""

from .models.color_encoder import EEGColorEncoder, load_color_encoder
from .models.edge_encoder import EEGEdgeEncoder, load_edge_encoder
from .models.semantic_encoder import SubjectAdaptiveSemanticEncoder

__all__ = [
    "EEGColorEncoder",
    "EEGEdgeEncoder",
    "SubjectAdaptiveSemanticEncoder",
    "load_color_encoder",
    "load_edge_encoder",
]
