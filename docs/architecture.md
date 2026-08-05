# Architecture

## Shared semantic path

`SubjectAdaptiveSemanticEncoder` receives the canonical ten-participant EEG
ensemble. It combines subject-specific linear anchors with shared temporal and
spatial residuals and returns a normalized 384-dimensional semantic condition.
The residual path starts at zero, so training begins from the validated linear
baseline instead of a random mapping.

## Color path

1. `EEGDINOGridPredictor` predicts an 8 x 8 visual feature grid from EEG.
2. `TaskResidualAdapter` combines the grid, per-subject context, and shared
   semantic condition.
3. The frozen color decoder produces a conservative baseline reconstruction.
4. `MultiscaleEEGColorEncoder` independently extracts temporal color/style cues
   from raw EEG at several receptive-field sizes.
5. `StructurePreservingColorRefiner` changes chroma and palette while preserving
   stable luminance structure.

The training objective combines color reconstruction in OKLab, palette and
descriptor agreement, spatial gradients, hard-pixel CVaR, ranking against
shuffled EEG, and penalties for chromatic ringing and out-of-gamut values.

## Edge path

1. `EEGDINOGridPredictor` maps EEG to a spatial visual feature grid.
2. The frozen pyramid decoder renders a multi-scale baseline.
3. `MultiscaleEEGStructureEncoder` extracts independent geometry evidence from
   temporal EEG features.
4. `StableEdgeRefiner` applies gated residual corrections rather than replacing
   the baseline outright.

The output pyramid contains coarse shape and signed distance at 16 x 16,
semantic boundaries at 32 x 32, and perceptual/crisp boundaries, confidence,
uncertainty, and orientation at 64 x 64.

## Why the bundles contain several checkpoints

`best.pt` is the final task-specific EEG encoder. Files under `components/` are frozen
dependencies needed to execute it. Keeping those dependencies inside each model
bundle makes inference reproducible and prevents accidental coupling to an old
experiment directory.

The same semantic checkpoint appears in both bundles by design. This small
duplication makes each downloaded bundle independently usable.
