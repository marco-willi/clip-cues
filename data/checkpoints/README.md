# Pre-trained Model Checkpoints

This directory contains 12 pre-trained checkpoints for synthetic image detection.

## Quick Reference

| Model | Type | Size | Best For |
| -------------------------------- | ---------- | ------ | ----------------------- |
| `clip_orthogonal_synthclic.ckpt` | Orthogonal | 35 KB | Modern diffusion models |
| `clip_orthogonal_combined.ckpt` | Orthogonal | 35 KB | General use |
| `cm_antonyms_combined.ckpt` | Concept | 1.0 MB | Interpretability |

## Model Types

### CLIP Orthogonal Models

Lightweight classifiers (8 hidden units) trained with activation orthogonality loss.

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

Simple linear classifiers (no hidden layers) as baseline comparison.

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
# See notebooks/concept_model_inference.ipynb for full example
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
