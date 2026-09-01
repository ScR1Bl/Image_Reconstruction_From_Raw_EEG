"""Stable public interface for the final EEG structure pipeline."""

from pathlib import Path

import torch

from .edge_specialist import (
    EdgeSpecialistOutput,
    EEGEdgeEncoderPipeline,
    load_eeg_edge_encoder,
)

EEGEdgeEncoder = EEGEdgeEncoderPipeline


def load_edge_encoder(
    root: str | Path = "runs/eeg_edge_encoder",
    device: torch.device | str = "cpu",
) -> tuple[EEGEdgeEncoder, dict]:
    """Load the complete EEG-to-edge pipeline and its checkpoint metadata."""

    return load_eeg_edge_encoder(root=root, device=device)


__all__ = ["EEGEdgeEncoder", "EdgeSpecialistOutput", "load_edge_encoder"]
