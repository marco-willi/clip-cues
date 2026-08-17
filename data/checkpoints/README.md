# Pre-trained Model Checkpoints

This directory contains 12 pre-trained checkpoints for synthetic image detection. All of them are
heads over the same **frozen** CLIP ViT-L/14-336 encoder, so none of them is a fine-tuned backbone.

## How these map to the paper

| checkpoints | paper object | role |
| --- | --- | --- |
| `linear_probe_*.ckpt` | **`D_h`** — the canonical detector | A single logistic direction in the 1024-d pre-projection `pooler_output` space. This is *the* detector: every headline detection number and every interpretation target in the paper refers to it. |
| `clip_orthogonal_*.ckpt` | factorized k=8 head | A **diagnostic ablation**, not the detector. Without a nonlinearity it spans the same class of decision boundaries as the linear head. Its *effective* boundary is stable across seeds (Σ-cos 0.963), but its **individual axes are not** (Hungarian-matched \|cos\| 0.289, cue-profile ρ 0.119) — so per-axis interpretations of this head are not reproducible and should not be made. |
| `cm_antonyms_*.ckpt` | concept model | A separate, intrinsically text-grounded classifier over the 168 antonym cue pairs. It is trained on labels, not fitted to reproduce the linear detector's scores, so it is a complementary model rather than an explanation of `D_h`. |

Two further objects in the paper have **no checkpoint here** because they are analysis tools rebuilt
from cached features: `D_e`, the 768-d shared-space analysis head (used only where a detector
direction must be compared with text directions), and `D_cue`, the probe restricted to the 168 named
cue scores. Both are reproduced by the F1–F7 consolidation — see
[`experiments/final_consolidation/`](../../reproduction/experiments/final_consolidation/README.md).

## Provenance

- `linear_probe_synthclic`, `linear_probe_cnnspot` and `clip_orthogonal_synthclic` were trained
  end-to-end **with** augmentation (RandomResizedCrop 0.5–1.0 → 512, horizontal flip, JPEG 65–100).
  The measured effect of augmentation on the resulting direction is small (+0.007).
- Three of these files are registered by sha256 in the frozen input snapshot
  ([`experiments/data/MANIFEST.md`](../../reproduction/experiments/data/MANIFEST.md)) and are verified on load,
  so a checkpoint cannot be silently swapped.
- The checkpoints score directly from cached embeddings in numpy, which is why **no GPU is needed**
  to reproduce any interpretation result.

## Model Types

### CLIP Orthogonal Models

Lightweight classifiers (8 hidden units) trained with an activation orthogonality loss.
Diagnostic ablation — see the mapping table above before interpreting individual axes.

| Checkpoint | Training Data | Size |
| ---------------------------------- | ------------- | ----- |
| `clip_orthogonal_synthclic.ckpt` | SynthCLIC | 35 KB |
| `clip_orthogonal_synthbuster.ckpt` | SynthBuster+ | 35 KB |
| `clip_orthogonal_cnnspot.ckpt` | CNNSpot | 35 KB |
| `clip_orthogonal_combined.ckpt` | All datasets | 35 KB |

**Usage:**

```python
from clip_cues import load_clip_classifier

model = load_clip_classifier("data/checkpoints/clip_orthogonal_synthclic.ckpt")
prob = model.predict("image.jpg")
```

### Linear Probe Models

Single logistic direction in the frozen 1024-d CLIP representation — the canonical detector `D_h`.

| Checkpoint | Training Data | Size |
| ------------------------------- | ------------- | ---- |
| `linear_probe_synthclic.ckpt` | SynthCLIC | 6 KB |
| `linear_probe_synthbuster.ckpt` | SynthBuster+ | 6 KB |
| `linear_probe_cnnspot.ckpt` | CNNSpot | 6 KB |
| `linear_probe_combined.ckpt` | All datasets | 6 KB |

**Usage:**

```python
from clip_cues import load_clip_classifier

model = load_clip_classifier("data/checkpoints/linear_probe_combined.ckpt")
prob = model.predict("image.jpg")
```

### Concept Bottleneck Models

Interpretable models using 168 visual concept pairs. Provides both predictions and concept activations.

| Checkpoint | Training Data | Size |
| ------------------------------ | ------------- | ------ |
| `cm_antonyms_synthclic.ckpt` | SynthCLIC | 1.0 MB |
| `cm_antonyms_synthbuster.ckpt` | SynthBuster+ | 1.0 MB |
| `cm_antonyms_cnnspot.ckpt` | CNNSpot | 1.0 MB |
| `cm_antonyms_combined.ckpt` | All datasets | 1.0 MB |

**Usage:**

```python
from clip_cues import load_concept_model

model, extractor = load_concept_model("data/checkpoints/cm_antonyms_combined.ckpt")
# See examples/concept_classifier_inference.ipynb for a full example
```

## Training Datasets

- **SynthCLIC**: Modern diffusion models (SD3, FLUX, Imagen3) with CLIC2020 real images
- **SynthBuster+**: Diverse generators (GLIDE, Midjourney, DALL-E, etc.) with Raise1K real images
- **CNNSpot**: ProGAN generated images
- **Combined**: All datasets merged

## Checkpoint Format

Each checkpoint is a PyTorch file containing:

- `state_dict`: Model weights
- `model_type`: One of "clip", "linear", or "concept"
