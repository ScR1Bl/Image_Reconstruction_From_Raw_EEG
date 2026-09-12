"""Shared helpers for the Lightning shells: trunk-by-path loading and converted
checkpoint IO with explicit key accounting (never a silent ``strict=False``)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

# The shared semantic trunk lives inside the frozen grid predictor of both
# branch pipelines under this state_dict prefix. Branch checkpoints exclude it;
# the single trunk artifact is loaded by path instead (Step 1a design).
TRUNK_KEY_PREFIX = "pipeline.predictor.encoder."


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


def load_converted_branch(cls, path: str | Path, map_location: str = "cpu"):
    """Load a converted branch checkpoint whose state_dict excludes the trunk.

    ``strict=False`` is used ONLY with full accounting: the missing keys must be
    exactly the trunk keys (already loaded by path in ``__init__``) and there
    must be no unexpected keys. Anything else is a hard error.
    """

    payload = torch.load(path, map_location=map_location, weights_only=False)
    module = cls(**payload["hyper_parameters"])
    missing, unexpected = module.load_state_dict(payload["state_dict"], strict=False)
    non_trunk_missing = [key for key in missing if not key.startswith(TRUNK_KEY_PREFIX)]
    if unexpected or non_trunk_missing:
        raise RuntimeError(
            f"converted checkpoint {path} does not match the module: "
            f"unexpected={unexpected} missing(non-trunk)={non_trunk_missing}"
        )
    # Mirror the legacy inference loaders (load_eeg_color_encoder /
    # load_eeg_edge_encoder), which freeze the WHOLE pipeline. This is not
    # cosmetic: requires_grad flags select different CPU kernels at the ~1e-6
    # ULP level, so inference parity with the old loaders requires identical
    # flag semantics. Training re-enables specialist grads in on_fit_start.
    return freeze(module)


def freeze(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module
