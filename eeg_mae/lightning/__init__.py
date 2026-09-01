"""Lightning shells around the existing eeg_mae models (Step 1a refactor).

Composition, not rewriting: every LightningModule here holds the ORIGINAL
``eeg_mae.models`` modules and delegates forward to them verbatim. The old
trainers in ``eeg_mae.training`` remain the reference implementation; loss
functions are imported from them, never copied by hand where importable.
"""

from .color import ColorLightning
from .edges import EdgesLightning
from .eeg_encoder import SemanticEncoderLightning

__all__ = ["ColorLightning", "EdgesLightning", "SemanticEncoderLightning"]
