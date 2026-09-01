# EEG Color and Edge Encoders

PyTorch pipelines that recover complementary visual conditions from
[THINGS-EEG2](https://osf.io/3jk45/) EEG recordings:

- a **color encoder** for palette, chroma, luminance, and coarse color layout;
- an **edge encoder** for multi-scale shape, boundaries, distance, and orientation.

Both production pipelines take EEG as their only inference input. Image encoders
and oracle decoders are frozen training teachers or bundled rendering components;
the target image is never available during EEG inference.

## Architecture

```text
EEG [batch, 10 subjects, 17 channels, 100 samples]
  |
  +-- shared subject-adaptive semantic encoder
  |
  +-- color-specific EEG branch --> color condition --> RGB 64 x 64
  |
  `-- edge-specific EEG branch  --> spatial grid --> edge pyramid 16/32/64
```

The two branches are deliberately separate. Color and geometry overlap, but they
are not interchangeable signals: a shared bottleneck encouraged average-looking
outputs, while task-specific branches can retain complementary information.
See [the architecture notes](docs/architecture.md) for the full data flow.

## Installation

Python 3.10+ and PyTorch 2.2+ are required.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev,visual-teachers]"
```

## Model bundles

Model weights are not stored in Git because the complete bundles are about
400 MB. Download both archives from the
[models-v1 release](https://github.com/ScR1Bl/Image_Reconstruction_From_Raw_EEG/releases/tag/models-v1)
and unpack them into:

```text
runs/
  eeg_color_encoder/
    best.pt
    components/
  eeg_edge_encoder/
    best.pt
    components/
```

Each bundle is self-contained and also includes its training configuration and
best-checkpoint preview. The selected checkpoints are epoch 22 for color and
epoch 15 for edges. SHA-256 hashes are recorded in
[`models/manifest.json`](models/manifest.json).

## Inference

```python
import torch

from eeg_mae import load_color_encoder, load_edge_encoder

eeg = torch.randn(1, 10, 17, 100, device="cuda")

color_encoder, color_metadata = load_color_encoder(device="cuda")
edge_encoder, edge_metadata = load_edge_encoder(device="cuda")

with torch.inference_mode():
    color = color_encoder(eeg)
    edges = edge_encoder(eeg)

rgb = color.image.rgb                 # [1, 3, 64, 64], range [0, 1]
semantic_edges = edges.pyramid.semantic32  # [1, 1, 32, 32]
crisp_edges = edges.pyramid.crisp64        # [1, 1, 64, 64]
```

The ten-subject dimension represents an ensemble of aligned responses to the
same stimulus. Subject IDs are not inferred from tensor order; use the canonical
THINGS-EEG2 order (`sub-01` through `sub-10`).

## Dataset

This project uses **THINGS-EEG2**, not THINGS-EEG1. The source dataset contains
10 participants and 16,740 image conditions from the THINGS image database.
Our final training split is image-disjoint within the official 1,654-concept
training partition:

| Split | Images per concept | Unique images | EEG rows across 10 subjects |
|---|---:|---:|---:|
| train | 8 | 13,232 | 529,280 |
| validation | 1 | 1,654 | 66,160 |
| untouched test | 1 | 1,654 | 66,160 |

Every image has four EEG repetitions per subject. Repetitions of the same image
never cross splits. We use the authors' preprocessed representation: 17 posterior
channels, 100 samples from -200 ms to 800 ms at 100 Hz. The encoders consume the
80 post-stimulus samples after the first 20 baseline samples.

See [data preparation and citation](docs/data.md). Dataset files and stimulus
images are intentionally excluded from this repository.

## Training

After preparing the dataset and derived teacher banks, train or continue the
final encoders with the stable command names:

```bash
eeg-train-color --resume runs/eeg_color_encoder/best.pt
eeg-train-edges --resume runs/eeg_edge_encoder/best.pt
```

Run `--help` for all paths and hyperparameters. The defaults point to the two
canonical bundle directories.

## Source layout

```text
eeg_mae/
  models/       encoders, decoders, losses, and public inference pipelines
  training/     final trainers and reproducible component-training stages
  data/         THINGS-EEG2 indexing, caching, and teacher-bank builders
  evaluation/   official 200-way benchmark and leakage controls
```

Historical compact/full-color adapters and abandoned versioned edge/color
experiments have been removed. Every remaining module belongs to inference,
final training, component reproduction, dataset preparation, or evaluation.

## Tests

```bash
python -m pytest -q
```

The test suite covers the semantic encoder, spatial predictor, color and edge
losses, refiners, adapters, public API, and anti-collapse controls.

## Repository policy

The repository contains source code and documentation only. Raw EEG, stimulus
images, derived feature banks, checkpoints, logs, and caches are ignored. This
keeps Git history small and avoids redistributing third-party data accidentally.
