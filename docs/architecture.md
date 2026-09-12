# Architecture

The project now uses three parameter-independent EEG encoders. They receive the
same preprocessed EEG, but do not call one another and do not share a visual
baseline at inference.

| Encoder | Supervision | Output used by reconstruction |
|---|---|---|
| semantic | global DINO embedding and contrastive retrieval | object/category context |
| color | OKLab, palette, chroma classes and color descriptors | luminance, palette and color layout |
| edge | shape, distance, boundaries and orientation | geometry and high-frequency structure |

```text
raw EEG
  +-- semantic EEG encoder --> global semantic vector
  +-- color EEG encoder ----> 16x16 color tokens --> direct OKLab renderer
  `-- edge EEG encoder -----> 16x16 geometry tokens --> edge pyramid 16/32/64
```

## Why the task paths are independent

The previous color and edge models both started from an 8x8 DINO grid predicted
by the semantic path. Color froze its baseline luminance and edges could only add
small gated residuals. This made both outputs inherit the same coarse spatial
prototype.

The direct models remove that route:

- the color encoder has its own temporal CNN/Transformer parameters, learned
  16x16 output queries, subject fusion and direct luminance/chroma renderer;
- the edge encoder has a separate temporal CNN/Transformer, separate 16x16
  geometry queries and directly supervised heads at three resolutions;
- the semantic encoder remains a global retrieval model and produces neither
  color nor edge maps.

The task encoders may still discover genuinely shared evidence in EEG. We do not
force their raw features to be orthogonal, because color regions and object
boundaries are physically correlated. Complementarity comes from disjoint
parameters and task-specific targets. VICReg-style variance/covariance terms and
shuffled-target ranking discourage constant category-average outputs.

## Production color hybrid

The final color candidate combines two private experts inside one EEG-only model:

1. an exact copy of the proven spatial color pipeline, including its own semantic
   predictor and adapter;
2. the independently trained direct 16x16 color-token encoder;
3. a bidirectional token cross-attention block and a new OKLab proposal renderer;
4. a per-pixel, per-OKLab-channel router over legacy, direct, and proposal values.

This is a mixture of experts rather than a constant RGB average. The router can,
for example, select legacy luminance and direct chroma in one region. Parent
experts stay frozen while fusion trains, preventing a good input model from being
silently destroyed. Routing has a per-pixel/per-channel oracle target: for each
OKLab component it learns which expert is closest to the real image. A CVaR guard
penalizes the worst 10% of local errors, while the proposal expert is restricted
to a small envelope around the two validated parent predictions. This prevents a
free proposal from inventing saturated blobs merely to improve an average loss.
`best_qualified.pt` is written only after the hybrid meets explicit parent-relative
constraints for RGB/chroma error, palette correlation, between-image diversity,
salient-color localization, and chromatic-edge artifacts.

## Design sources

- EEG temporal convolutions followed by self-attention follow the motivation of
  [EEG Conformer](https://doi.org/10.1109/TNSRE.2022.3230250).
- Learned spatial output queries follow the flexible output-query formulation of
  [Perceiver IO](https://arxiv.org/abs/2107.14795).
- Categorical chroma bins and rare-color reweighting follow
  [Colorful Image Colorization](https://arxiv.org/abs/1603.08511).
- Direct multi-scale edge supervision follows
  [Holistically-Nested Edge Detection](https://arxiv.org/abs/1504.06375).
- The anti-collapse regularizer follows
  [VICReg](https://arxiv.org/abs/2105.04906).
- The separation between a representation prior and an image decoder follows
  [DALL-E 2](https://arxiv.org/abs/2204.06125).
- The generative latent objective follows
  [Flow Matching](https://arxiv.org/abs/2210.02747), with conditional/unconditional
  dropout from [classifier-free guidance](https://arxiv.org/abs/2207.12598).

## Next-stage hierarchical visual target

The first reconstruction bootstrap stage is a frozen, image-only teacher bank:

```text
image --+--> SynCLR ViT-B/16 CLS -------- 768 values
        +--> CLIP ViT-B/32 projection --- 512 values
        `--> SDXL VAE 4x16x16 latent ---- 1024 values
```

These are parallel descriptions, not a sequence of models. SynCLR supplies
visual invariances, CLIP supplies image-language semantics, and the VAE preserves
low-level information in the latent space of the eventual generator. The bank
matches the teacher family and feature locations used by Brain-HIVE while adding
pinned revisions, deterministic per-image VAE sampling, transactional resume,
and explicit train/validation/test gating.

The existing color and edge outputs are not fused into this bank: they are EEG
conditions and therefore belong after the new semantic EEG encoder, not among
the image-only supervision targets. The fusion-prior stage combines the three
predicted hierarchical features with color and geometry conditions.

That prior is a conditional rectified-flow transformer. Four global semantic
tokens remain explicit, while Perceiver-style spatial queries combine the
independent 16x16 color and geometry grids. A flow head learns the distribution
of scaled 16x16 SDXL-VAE latents rather than minimizing only pixelwise regression,
which would favor a conditional average. Classifier-free condition dropout
enables guidance at sampling time. Validation reports the increase in flow error
when semantic, color, or edge conditions are removed; a non-positive contribution
is therefore visible before the later DiT adapter is trained.

The hierarchical EEG stage uses the validated DINO EEG checkpoint as its
subject-adaptive backbone. Three residual heads predict SynCLR-, CLIP-, and
VAE-aligned tokens in a common 1024-dimensional space; a fourth token fuses the
three predictions. Image-side projections and the fusion block are learned
jointly, following Brain-HIVE. Random teacher dropout prevents dependence on
every teacher at every update, while DINO cosine distillation limits catastrophic
forgetting when the backbone is unfrozen.

The papers motivate individual mechanisms; they do not constitute evidence that
this exact EEG-to-image combination is optimal. That must be established by the
untouched test split and shuffled/zero-EEG controls.
