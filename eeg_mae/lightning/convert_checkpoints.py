"""Convert the baseline (legacy) checkpoints to the Lightning layout.

Reads ``artifacts/<model>/baseline/checkpoints`` (NEVER modified) and writes
``artifacts/<model>/baseline-lightning/``. Handles both legacy key schemas of
the frozen grid predictor (audit flag C13): ``predictor/predictor_config`` in
the color bundle's ``adapter.pt`` and ``model/model_config`` in the edge
bundle's ``baseline.pt``. Specialist weights use the EMA variant, exactly like
``load_eeg_color_encoder`` / ``load_eeg_edge_encoder``.

The shared semantic trunk is written ONCE (eeg_encoder artifact); branch
checkpoints exclude the trunk tensors and reference the trunk artifact by path
+ sha256 in their hyperparameters and config.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning
import torch
import yaml

from .color import ColorLightning
from .common import TRUNK_KEY_PREFIX, sha256_of_file
from .edges import EdgesLightning
from .eeg_encoder import SemanticEncoderLightning

CODE_COMMIT = "b9360cc"
CONVERTED_ON = "2026-09-01"


def lightning_payload(module, epoch: int, state_dict: dict) -> dict:
    return {
        "state_dict": state_dict,
        "hyper_parameters": dict(module.hparams),
        "epoch": epoch,
        "global_step": 0,
        "pytorch-lightning_version": lightning.__version__,
    }


def write_config(directory: Path, config: dict) -> None:
    (directory / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


def convert_trunk(artifacts: Path) -> tuple[Path, str]:
    source = artifacts / "eeg_encoder" / "baseline" / "checkpoints" / "semantic_encoder.pt"
    payload = torch.load(source, map_location="cpu", weights_only=False)
    module = SemanticEncoderLightning(model_config=payload["model_config"])
    module.model.load_state_dict(payload["model"], strict=True)

    target_dir = artifacts / "eeg_encoder" / "baseline-lightning"
    (target_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    target = target_dir / "checkpoints" / "semantic_encoder.ckpt"
    torch.save(
        lightning_payload(module, int(payload.get("epoch", -1)), module.state_dict()), target
    )
    trunk_sha = sha256_of_file(target)
    write_config(
        target_dir,
        {
            "model": "eeg_encoder",
            "run": "baseline-lightning",
            "converted_from": str(source),
            "source_sha256": sha256_of_file(source),
            "artifact_sha256": trunk_sha,
            "code_commit": CODE_COMMIT,
            "converted_on": CONVERTED_ON,
        },
    )
    print(f"trunk -> {target}")
    return target, trunk_sha


def branch_state_without_trunk(module) -> dict:
    full = module.state_dict()
    kept = {key: value for key, value in full.items() if not key.startswith(TRUNK_KEY_PREFIX)}
    dropped = len(full) - len(kept)
    trunk_params = len(module.pipeline.predictor.encoder.state_dict())
    if dropped != trunk_params:
        raise RuntimeError(
            f"expected to drop exactly the {trunk_params} trunk tensors, dropped {dropped}"
        )
    return kept


def assert_same_trunk(artifacts: Path, bundle: str) -> Path:
    canonical = artifacts / "eeg_encoder" / "baseline" / "checkpoints" / "semantic_encoder.pt"
    duplicate = (
        artifacts / bundle / "baseline" / "checkpoints" / "components" / "semantic_encoder.pt"
    )
    if sha256_of_file(canonical) != sha256_of_file(duplicate):
        raise RuntimeError(
            f"{duplicate} differs from the canonical trunk {canonical} — "
            "trunk-by-path conversion would change behavior; aborting"
        )
    return duplicate


def convert_color(artifacts: Path, trunk_artifact: Path, trunk_sha: str) -> Path:
    root = artifacts / "color" / "baseline" / "checkpoints"
    components = root / "components"
    assert_same_trunk(artifacts, "color")
    state = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
    semantic_state = torch.load(
        components / "semantic_encoder.pt", map_location="cpu", weights_only=False
    )
    adapter_state = torch.load(components / "adapter.pt", map_location="cpu", weights_only=False)
    decoder_state = torch.load(
        components / "color_decoder.pt", map_location="cpu", weights_only=False
    )

    module = ColorLightning(
        specialist_config=state["model_config"],
        predictor_config=adapter_state["predictor_config"],  # C13: schemat "predictor"
        adapter_config=adapter_state["adapter_config"],
        decoder_config=decoder_state["model_config"],
        trunk_config=semantic_state["model_config"],
        trunk_checkpoint=str(trunk_artifact).replace("\\", "/"),
        trunk_sha256=trunk_sha,
        oracle_checkpoint=str(components / "color_oracle.pt").replace("\\", "/"),
    )
    specialist_weights = state.get("ema", state["model"])
    module.pipeline.specialist.load_state_dict(specialist_weights, strict=True)
    module.ema_specialist.load_state_dict(specialist_weights, strict=True)
    module.pipeline.predictor.load_state_dict(adapter_state["predictor"], strict=True)
    module.pipeline.adapter.load_state_dict(adapter_state["adapter"], strict=True)
    module.pipeline.decoder.load_state_dict(decoder_state["model"], strict=True)

    target_dir = artifacts / "color" / "baseline-lightning"
    (target_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    target = target_dir / "checkpoints" / "color.ckpt"
    torch.save(
        lightning_payload(module, int(state.get("epoch", -1)), branch_state_without_trunk(module)),
        target,
    )
    write_config(
        target_dir,
        {
            "model": "color",
            "run": "baseline-lightning",
            "weights_variant": "ema",
            "converted_from": {
                "best.pt": sha256_of_file(root / "best.pt"),
                "components/adapter.pt": sha256_of_file(components / "adapter.pt"),
                "components/color_decoder.pt": sha256_of_file(components / "color_decoder.pt"),
                "components/semantic_encoder.pt": sha256_of_file(
                    components / "semantic_encoder.pt"
                ),
            },
            "predictor_key_schema": "predictor/predictor_config (C13)",
            "trunk_checkpoint": str(trunk_artifact).replace("\\", "/"),
            "trunk_sha256": trunk_sha,
            "code_commit": CODE_COMMIT,
            "converted_on": CONVERTED_ON,
        },
    )
    print(f"color -> {target}")
    return target


def convert_edges(artifacts: Path, trunk_artifact: Path, trunk_sha: str) -> Path:
    root = artifacts / "edges" / "baseline" / "checkpoints"
    components = root / "components"
    assert_same_trunk(artifacts, "edges")
    state = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
    semantic_state = torch.load(
        components / "semantic_encoder.pt", map_location="cpu", weights_only=False
    )
    baseline_state = torch.load(components / "baseline.pt", map_location="cpu", weights_only=False)
    decoder_state = torch.load(
        components / "edge_decoder.pt", map_location="cpu", weights_only=False
    )

    module = EdgesLightning(
        specialist_config=state["model_config"],
        predictor_config=baseline_state["model_config"],  # C13: schemat "model"
        decoder_config=decoder_state["model_config"],
        trunk_config=semantic_state["model_config"],
        trunk_checkpoint=str(trunk_artifact).replace("\\", "/"),
        trunk_sha256=trunk_sha,
    )
    specialist_weights = state.get("ema", state["model"])
    module.pipeline.specialist.load_state_dict(specialist_weights, strict=True)
    module.ema_specialist.load_state_dict(specialist_weights, strict=True)
    module.pipeline.predictor.load_state_dict(baseline_state["model"], strict=True)
    module.pipeline.decoder.load_state_dict(decoder_state["model"], strict=True)

    target_dir = artifacts / "edges" / "baseline-lightning"
    (target_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    target = target_dir / "checkpoints" / "edges.ckpt"
    torch.save(
        lightning_payload(module, int(state.get("epoch", -1)), branch_state_without_trunk(module)),
        target,
    )
    write_config(
        target_dir,
        {
            "model": "edges",
            "run": "baseline-lightning",
            "weights_variant": "ema",
            "converted_from": {
                "best.pt": sha256_of_file(root / "best.pt"),
                "components/baseline.pt": sha256_of_file(components / "baseline.pt"),
                "components/edge_decoder.pt": sha256_of_file(components / "edge_decoder.pt"),
                "components/semantic_encoder.pt": sha256_of_file(
                    components / "semantic_encoder.pt"
                ),
            },
            "predictor_key_schema": "model/model_config (C13)",
            "trunk_checkpoint": str(trunk_artifact).replace("\\", "/"),
            "trunk_sha256": trunk_sha,
            "code_commit": CODE_COMMIT,
            "converted_on": CONVERTED_ON,
        },
    )
    print(f"edges -> {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert baseline checkpoints to Lightning")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--only", choices=("trunk", "color", "edges"), nargs="+", default=None)
    args = parser.parse_args()
    wanted = set(args.only or ("trunk", "color", "edges"))

    trunk_artifact = (
        args.artifacts
        / "eeg_encoder"
        / "baseline-lightning"
        / "checkpoints"
        / "semantic_encoder.ckpt"
    )
    if "trunk" in wanted:
        trunk_artifact, trunk_sha = convert_trunk(args.artifacts)
    else:
        trunk_sha = sha256_of_file(trunk_artifact)
    if "color" in wanted:
        convert_color(args.artifacts, trunk_artifact, trunk_sha)
    if "edges" in wanted:
        convert_edges(args.artifacts, trunk_artifact, trunk_sha)


if __name__ == "__main__":
    main()
