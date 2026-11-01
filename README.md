# EEG Image Reconstruction

PyTorch pipelines for decoding visual information from THINGS-EEG2 recordings
and generating images with PixArt. The project includes a semantic EEG encoder,
an image generation adapter, and dedicated color and edge reconstruction models.

## Overview

```text
EEG → Semantic encoder → CLIP features → PixArt adapter → Generated image
EEG → Spatial DINO predictor + Color specialist → Color reconstruction
EEG → Spatial DINO predictor + Edge specialist  → Edge maps
```

The semantic encoder maps EEG to a 768-dimensional CLIP representation. An
adapter converts these features into image-conditioning tokens for a frozen
PixArt transformer, and a VAE decodes the generated latents into an image.

The color and edge models produce separate outputs: RGB images at 64 × 64 and
structural maps at 16 × 16, 32 × 32, and 64 × 64. They use frozen DINO-based
predictions together with task-specific EEG features. These maps are not inputs
to the PixArt adapter.

## Installation

Requires Python 3.10+ and PyTorch 2.2+. A CUDA-capable GPU is recommended for
training and image generation.

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` on Windows or
`source .venv/bin/activate` on Linux and macOS. Then install the project:

```bash
python -m pip install -e ".[visual-teachers,dit,dev]"
```

Run the commands below from the repository root.

## Dataset

The project uses the authors' preprocessed **THINGS-EEG2** recordings:

- 10 participants, with known subject identities.
- 17 EEG channels and 100 time samples per recording at 100 Hz.
- Four repetitions per training image and participant.
- The encoders use the final 80 samples, covering approximately 0–800 ms after stimulus onset.

The official training partition is divided by image within each of its 1,654
categories:

| Split | Images per category | Total images |
|---|---:|---:|
| Training | 8 | 13,232 |
| Validation | 1 | 1,654 |
| Internal test | 1 | 1,654 |

All participants and repetitions for a given image remain in the same split.
The official 200-image test partition is evaluated separately.

See [dataset preparation](docs/data.md) for the expected file layout and dataset
citation. After placing the data in `data/things_eeg2_osf/`, build the index and
EEG cache:

```bash
python -m eeg_mae.data.build_dataset_index
python -m eeg_mae.data.build_eeg_cache
```

## Model checkpoints

Inference expects the following local files:

```text
runs/
  eeg_encoder/
    best.ckpt
  dit_adapter/
    best.ckpt
  eeg_color_encoder/
    best.pt
    components/
  eeg_edge_encoder/
    best.pt
    components/
data/derived/
  pixart_null.pt
```

The color and edge bundles require their complete `components/` directories.
Model weights and datasets are excluded from Git and must be supplied separately
or produced by training. Checkpoint checksums are listed in
[models/manifest.json](models/manifest.json).

PixArt model files are loaded from Hugging Face by default. Pass `--repo` with
a local model snapshot directory to use an existing copy.

## Generate images from EEG

Save preprocessed EEG as a NumPy array with shape **`[N, S, R, 17, 100]`**:

| Dimension | Meaning |
|---|---|
| `N` | Number of image conditions |
| `S` | Number of participants |
| `R` | Number of repetitions per participant |
| `17` | EEG channels in the dataset's channel order |
| `100` | Time samples, including the 20 pre-stimulus samples |

For one participant, `sub-01`:

```bash
python -m eeg_mae.evaluation.reconstruct_eeg --eeg input.npy --subject-ids 0 --device cuda --output runs/reconstruction
```

For all ten participants, provide `--subject-ids 0 1 2 3 4 5 6 7 8 9`.
Subject IDs are zero-based and must match the order of participants in the input
array. Repetitions are averaged before encoding; participant features are combined
using the checkpoint's fusion model.

The output directory must not already exist. It will contain generated PNGs,
`conditions.pt`, and `report.json` with the input hashes and generation settings.
Use `--no-render --device cpu` to export predicted CLIP features without loading
PixArt. Override `--encoder`, `--adapter`, and `--null-embeddings` to use different
checkpoint paths.

## Reconstruct color and edges

These models expect all ten participants in canonical order, `sub-01` through
`sub-10`, with repetitions averaged into `[N, 10, 17, 100]`.

```python
import numpy as np
import torch

from eeg_mae import load_color_encoder, load_edge_encoder

# input.npy: [N, 10, R, 17, 100], preprocessed THINGS-EEG2 recordings
eeg = torch.from_numpy(np.load("input.npy").mean(axis=2)).float()

color_model, _ = load_color_encoder("runs/eeg_color_encoder", "cpu")
edge_model, _ = load_edge_encoder("runs/eeg_edge_encoder", "cpu")

with torch.inference_mode():
    rgb = color_model(eeg).image.rgb                         # [N, 3, 64, 64]
    edges = edge_model(eeg).pyramid.crisp_logits64.sigmoid() # [N, 1, 64, 64]
```

## Training

Prepare CLIP targets, VAE latent moments, and the fixed PixArt text condition:

```bash
python -m eeg_mae.data.build_clip_targets
python -m eeg_mae.data.build_vae_latents
python -m eeg_mae.data.build_pixart_null
```

Skip existing complete banks; the builders protect their output files against
overwriting. Train the EEG encoder and image adapter in separate runs:

```bash
python -m eeg_mae.lightning.train_eeg_encoder --run-name clip-encoder
python -m eeg_mae.lightning.train_dit_adapter --run-name image-adapter
```

The EEG encoder learns from paired EEG and image features. The adapter learns
from image CLIP features and VAE latents, while the base PixArt model stays frozen.
New checkpoints and logs are written under `artifacts/<model>/<run-name>/`.
Use `--resume` with a full training checkpoint to continue training, or `--help`
to inspect model-specific initialization and training options.

The color and edge specialists have separate training entry points:

```bash
python -m eeg_mae.training.train_color_encoder --output runs/color-training
python -m eeg_mae.training.train_edge_encoder --output runs/edge-training
```

They require their component checkpoints and visual target banks. The color
pipeline's base adapter is a supplied prerequisite; its training script is not
included. See the [detailed reproduction guide](docs/reproduce.md) (in Polish)
for component preparation, initialization, and evaluation commands.

## Evaluation

- `eeg_mae.evaluation.eval_200way` evaluates EEG feature retrieval on the official
  200-image test partition.
- `eeg_mae.evaluation.evaluate_reconstruction` scores generated images against
  explicitly paired CLIP targets after generation.

Retrieval accuracy and image-generation quality measure different tasks.
Results depend on the participant set, repetition count, data split, and candidate
gallery. The appearance of a generated image alone does not establish recovery
of the stimulus's exact pose or geometry.

## Repository structure

```text
eeg_mae/
  data/          Dataset indexing, EEG caching, and visual target builders
  models/        EEG encoders, image decoders, and loss functions
  training/      DINO, color, edge, and component training
  lightning/     Semantic encoder and PixArt adapter training
  evaluation/    Retrieval evaluation, reconstruction, and image scoring
docs/            Dataset and reproduction instructions
models/          Checkpoint manifests
```

Local datasets, weights, caches, and generated outputs live in `data/`, `runs/`,
and `artifacts/` and are not tracked by Git.

## Development

```bash
python -m compileall -q eeg_mae
ruff format --check eeg_mae
```
