"""Stable public interface for the final EEG color pipeline."""

from pathlib import Path

import torch

from .color_specialist import (
    ColorSpecialistOutput,
    EEGColorEncoderPipeline,
    load_eeg_color_encoder,
)

EEGColorEncoder = EEGColorEncoderPipeline


def load_color_encoder(
    root: str | Path = "runs/eeg_color_encoder",
    device: torch.device | str = "cpu",
) -> tuple[EEGColorEncoder, dict]:
    """Load the complete EEG-to-color pipeline and its checkpoint metadata."""

    return load_eeg_color_encoder(root=root, device=device)


__all__ = ["ColorSpecialistOutput", "EEGColorEncoder", "load_color_encoder"]
