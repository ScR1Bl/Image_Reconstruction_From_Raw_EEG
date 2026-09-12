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
  +-- independent semantic encoder --> global DINO condition
  +-- independent color encoder ----> color grid --> OKLab/RGB 64 x 64
  `-- independent edge encoder -----> geometry grid --> edge pyramid 16/32/64
```

All three encoders have separate parameters. The color and edge paths no longer
consume the semantic encoder, an 8x8 DINO grid, or a frozen image-like baseline
at inference. Color and geometry overlap in real images, but their supervision
and output representations serve different reconstruction roles.
See [the architecture notes](docs/architecture.md) for the full data flow.

The separate [single-subject geometry experiment](docs/single_subject_geometry.md)
adds a frozen 6/2/2 image split, Lightning training of one participant, semantic
teacher comparisons and out-of-fold tests of extra geometry from edge/color
predictions. Its checkpoints live under `runs/single_subject_geometry/` and are
not promoted production bundles. Install `.[training]` for these entry points.

## Installation

Python 3.10+ and PyTorch 2.2+ are required.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev,visual-teachers]"
```

## Model bundles

Model weights are not stored in Git. The `models-v1` release contains the older
baseline-dependent checkpoints and is intentionally rejected by the new direct
loaders. Train the direct models below; publish a new release only after the
untouched-test evaluation succeeds. New bundles have the following minimal form:

```text
runs/
  eeg_color_encoder/
    best.pt
  eeg_edge_encoder/
    best.pt
```

Each new `best.pt` is self-contained. Frozen image teachers are used during
training only and are not part of inference.

## Inference

```python
import torch

from eeg_mae import load_color_encoder, load_edge_encoder

eeg = torch.randn(1, 10, 17, 100, device="cuda")

color_encoder, color_metadata = load_color_encoder("runs/eeg_color_encoder_direct", "cuda")
edge_encoder, edge_metadata = load_edge_encoder("runs/eeg_edge_encoder_direct", "cuda")

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

### Hierarchical visual teachers

Before training the next semantic EEG encoder, build the frozen visual target
bank. This reproduces the Brain-HIVE teacher combination with pinned checkpoints:
SynCLR ViT-B/16, LAION CLIP ViT-B/32, and 128-pixel SDXL VAE latents.

```powershell
python -m eeg_mae.data.build_hierarchical_visual_targets `
  --output data/derived/hierarchical_visual_targets `
  --batch-size 16
```

The teachers run one after another and are removed from GPU memory between
stages. An interrupted teacher resumes from its last flushed batch. The output
contains memory-mapped arrays, an image/split index, pinned model revisions,
source fingerprints, and SHA-256 checksums. Use `--max-images 64` only for a
pipeline smoke test; a production bank must report 16,540 images and the expected
13,232/1,654/1,654 split in `manifest.json`.

The SDXL VAE posterior is sampled deterministically per image, preserving the
Brain-HIVE target distribution without making the bank depend on batch size or
resume position. CLIP and SynCLR features are intentionally stored before L2
normalization; normalization belongs after learned fusion, as in the reference
method.

This stage reads images only. It does not train on EEG, train a prior, or expose
test targets to an optimizer. `HierarchicalTargetBank(root, split=...)` requires
one explicit split and prevents a training iterator from silently spanning the
validation or untouched-test rows.

Train the hierarchical semantic EEG stage against that bank with:

```powershell
python -m eeg_mae.training.train_hierarchical_semantic_encoder `
  --target-bank data/derived/hierarchical_visual_targets `
  --dino-anchor runs/eeg_color_encoder/components/semantic_encoder.pt `
  --output runs/eeg_hierarchical_encoder `
  --epochs 50 `
  --batch-size 128
```

Resume it without resetting the optimizer or checkpoint selection:

```powershell
python -m eeg_mae.training.train_hierarchical_semantic_encoder `
  --resume runs/eeg_hierarchical_encoder/last.pt `
  --output runs/eeg_hierarchical_encoder
```

The model retains the validated 384-dimensional DINO EEG encoder as a warm
anchor, predicts separate 1024-dimensional SynCLR-, CLIP-, and VAE-aligned
tokens, and emits one 1024-dimensional fused token. Training presents each
subject separately with its known ID; validation and reporting additionally use
the existing ten-subject ensemble. `best.pt` is selected only on validation.
The untouched test is evaluated after training with shuffled-target and zero-EEG
controls.

### Fusion prior

After selecting the hierarchical `best.pt`, cache the outputs of all three frozen
EEG encoders. This is a resumable one-time pass and prevents re-running the large
encoders during every prior epoch:

```powershell
python -m eeg_mae.data.build_fusion_condition_bank `
  --semantic-checkpoint runs/eeg_hierarchical_encoder/best.pt `
  --color-root runs/eeg_color_encoder_direct `
  --edge-root runs/eeg_edge_encoder_direct `
  --output data/derived/eeg_fusion_conditions `
  --batch-size 12
```

Then train the conditional rectified-flow prior:

```powershell
python -m eeg_mae.training.train_fusion_prior `
  --condition-bank data/derived/eeg_fusion_conditions `
  --semantic-checkpoint runs/eeg_hierarchical_encoder/best.pt `
  --output runs/eeg_fusion_prior `
  --epochs 80 `
  --batch-size 48
```

The prior maps EEG-derived semantic, color, and edge tokens to the distribution
of scaled SDXL VAE latents. Its validation report includes modality ablations, so
a branch cannot be credited merely because it duplicates another branch. Images
are training targets only; the saved prior accepts cached or live EEG conditions.
Resume with `--resume runs/eeg_fusion_prior/last.pt`.

Keep the validated legacy checkpoints intact and warm-start only matching EEG
feature-extractor weights into separate direct-model directories:

```bash
eeg-train-color --init-from runs/eeg_color_encoder/best.pt --output runs/eeg_color_encoder_direct
eeg-train-edges --init-from runs/eeg_edge_encoder/best.pt --output runs/eeg_edge_encoder_direct
```

After both color parents have finished, train the feature-level production hybrid:

```bash
python -m eeg_mae.training.train_hybrid_color_encoder \
  --legacy-root runs/eeg_color_encoder \
  --direct-root runs/eeg_color_encoder_direct \
  --output runs/eeg_color_encoder_hybrid
```

Resume it with:

```bash
python -m eeg_mae.training.train_hybrid_color_encoder \
  --resume runs/eeg_color_encoder_hybrid/last.pt \
  --output runs/eeg_color_encoder_hybrid
```

The production loader accepts only `best_qualified.pt`. The qualification guard
requires the hybrid to retain parent-level global color, diversity, salient-color
localization, and parent-level chromatic-edge cleanliness instead of selecting a
visually smooth checkpoint solely by MAE. Candidate checkpoints remain available
for diagnostics with the explicit `allow_unqualified=True` loader option.

The safe routing objective supervises the expert choice independently for each
OKLab channel and pixel, penalizes the worst 10% of parent-relative errors, and
bounds the learned proposal near the envelope of the two validated parents. When
resuming a checkpoint trained with the earlier objective, checkpoint selection is
reset automatically while the learned weights and optimizer state are retained.

Use `--resume <direct-run>/last.pt` only after the first direct run has started;
legacy checkpoints are not resume-compatible. Run `--help` for all parameters.

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
