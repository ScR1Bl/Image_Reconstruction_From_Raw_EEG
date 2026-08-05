# Contributing

1. Create a virtual environment and install `python -m pip install -e ".[dev]"`.
2. Keep raw data, derived banks, and checkpoints outside Git.
3. Format changes with `ruff format eeg_mae tests`.
4. Run `python -m pytest -q` before opening a pull request.

New experiments should write to a separate directory under `runs/`. Promote an
experiment to one of the two canonical model bundles only after validation on an
image-disjoint split and a shuffled-pair leakage control.
