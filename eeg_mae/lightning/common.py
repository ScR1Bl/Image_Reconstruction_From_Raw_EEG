"""Shared helpers for the Lightning shells: trunk-by-path loading and converted
checkpoint IO with explicit key accounting (never a silent ``strict=False``)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def sha256_of_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trunk_state(trunk_checkpoint: str | Path, expected_sha256: str | None) -> dict:
    """Read trunk weights from the converted Lightning trunk artifact.

    The trunk artifact stores the ``SemanticEncoderLightning`` state_dict, whose
    keys carry the ``model.`` composition prefix; strip it for the bare module.
    """

    if expected_sha256:
        actual = sha256_of_file(trunk_checkpoint)
        if actual != expected_sha256:
            raise RuntimeError(
                f"trunk checkpoint {trunk_checkpoint} has sha256 {actual}, "
                f"expected {expected_sha256} — refusing to load a different trunk"
            )
    payload = torch.load(trunk_checkpoint, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    prefix = "model."
    stripped = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    if len(stripped) != len(state):
        raise RuntimeError(f"unexpected keys in trunk checkpoint {trunk_checkpoint}")
    return stripped
