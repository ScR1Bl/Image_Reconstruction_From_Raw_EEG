"""Bounded, sequential local GPU queue; fail closed on any unsuccessful stage.

No test evaluation, model downloads or diffusion training are started here.
Completed reports are reused, but partial runs require explicit manual resume.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/single_subject_geometry/sub-01"))
    parser.add_argument("--wait-for-edge", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "pipeline_status.json"
    lock = root / "pipeline.lock"
    # Exclusive creation prevents two queues from competing for the GPU.
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    status = {
        "pid": os.getpid(),
        "state": "starting",
        "completed": [],
        "failed": None,
        "test_evaluated": False,
        "maximum_epochs_per_neural_run": 35,
    }

    def update(**values):
        status.update(values, updated_utc=datetime.now(timezone.utc).isoformat())
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temporary.replace(status_path)

    def stage(name, command, report):
        if report.exists():
            parsed = json.loads(report.read_text(encoding="utf-8"))
            if parsed.get("test_evaluated", False):
                raise RuntimeError(f"unexpected test evaluation in {report}")
            status["completed"].append({"stage": name, "reused": True})
            update()
            return
        log = root / "queue_logs" / f"{name}.log"
        log.parent.mkdir(exist_ok=True)
        update(state="running", stage=name, command=command, log=str(log))
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                command, stdout=handle, stderr=subprocess.STDOUT, timeout=7200, check=False
            )
        if result.returncode or not report.exists():
            raise RuntimeError(f"{name} failed (exit {result.returncode}); see {log}")
        status["completed"].append({"stage": name, "reused": False})
        update()

    def train(task, output, fold=-1):
        semantic = task in {"dino", "clip_b32", "synclr"}
        command = [
            sys.executable,
            "-u",
            "-m",
            "eeg_mae.lightning.train_single_subject",
            "--task",
            task,
            "--end-ms",
            "400",
            "--batch-size",
            "256" if semantic else "32" if task == "edge" else "16",
            "--epochs",
            "35" if semantic else "30",
            "--patience",
            "7",
            "--threads",
            "2",
            "--output",
            str(output),
        ]
        if fold >= 0:
            command += ["--oof-fold", str(fold)]
        if semantic:
            probes = json.loads((root / "probes" / "report.json").read_text(encoding="utf-8"))
            command += ["--ridge-alpha", str(probes["windows"]["400"][task]["alpha"])]
        stage(f"{task}_{'full' if fold < 0 else f'fold_{fold}'}", command, output / "report.json")

    try:
        if args.wait_for_edge and not (root / "edge_400" / "report.json").exists():
            update(state="waiting", stage="existing_edge_400")
            started = time.monotonic()
            while not (root / "edge_400" / "report.json").exists():
                history = root / "edge_400" / "history.jsonl"
                if time.monotonic() - started > 2700 or (
                    history.exists() and time.time() - history.stat().st_mtime > 300
                ):
                    raise RuntimeError("existing edge run stalled or failed; inspect its log")
                time.sleep(10)
        reports = {
            name: json.loads((root / f"{name}_400" / "report.json").read_text(encoding="utf-8"))
            for name in ("dino", "clip_b32", "synclr")
        }
        semantic = max(reports, key=lambda name: reports[name]["evaluation"]["val_r4"]["mrr"])
        update(selected_semantic=semantic)
        # Cheap semantic cross-fitting first; then profile and train color.
        for fold in range(3):
            train(semantic, root / "oof" / semantic / f"fold_{fold}", fold)
        profile = root / "profile_color"
        stage(
            "color_profile",
            [
                sys.executable,
                "-u",
                "-m",
                "eeg_mae.lightning.train_single_subject",
                "--task",
                "color",
                "--end-ms",
                "400",
                "--batch-size",
                "8",
                "--epochs",
                "1",
                "--limit-images",
                "128",
                "--threads",
                "2",
                "--output",
                str(profile),
            ],
            profile / "report.json",
        )
        train("color", root / "color_400")
        for task in ("edge", "color"):
            for fold in range(3):
                train(task, root / "oof" / task / f"fold_{fold}", fold)
        stage(
            "neural_complementarity",
            [
                sys.executable,
                "-u",
                "-m",
                "eeg_mae.training.evaluate_neural_geometry",
                "--root",
                str(root),
                "--semantic",
                semantic,
            ],
            root / "neural_complementarity" / "report.json",
        )
        update(
            state="complete",
            stage=None,
            next_step="review validation controls and missing generator artifacts",
        )
    except BaseException as error:
        update(state="failed", failed=str(error))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
