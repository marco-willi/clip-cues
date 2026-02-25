# CLIP-Cues

**Synthetic Image Detection with CLIP: Understanding and Assessing Predictive Cues**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Detect AI-generated images using CLIP features with interpretable models. This repository provides code and **12 pre-trained models** for detecting synthetic images.

## Installation

```bash
git clone https://github.com/marco-willi/clip-cues.git
cd clip-cues
pip install -e .
```

## Quick Start

```python
from clip_cues import load_clip_classifier

# Load model
model = load_clip_classifier("data/checkpoints/clip_orthogonal_synthclic.ckpt")

# Predict on a single image
prob = model.predict("path/to/image.jpg")
print(f"Synthetic probability: {prob:.1%}")
print("Prediction:", "Synthetic" if prob > 0.5 else "Real")
```

### Batch Inference

```python
from clip_cues import load_clip_classifier

model = load_clip_classifier("data/checkpoints/clip_orthogonal_synthclic.ckpt", device="cuda")

# Process multiple images
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
probs = model.predict_batch(image_paths, batch_size=32)

for path, prob in zip(image_paths, probs):
    print(f"{path}: {prob:.1%} synthetic")
```

## Pre-trained Models

We provide 12 pre-trained checkpoints in `data/checkpoints/`:

### CLIP Orthogonal Models (Recommended)

| Checkpoint | Training Data | Description |
| ---------------------------------- | ------------- | ----------------------------------------------------- |
| `clip_orthogonal_synthclic.ckpt` | SynthCLIC | Best for modern diffusion models (SD3, FLUX, Imagen3) |
| `clip_orthogonal_synthbuster.ckpt` | SynthBuster+ | Diverse generators including older models |
| `clip_orthogonal_cnnspot.ckpt` | CNNSpot | ProGAN images |
| `clip_orthogonal_combined.ckpt` | Combined | All datasets |

### Linear Probe Models

| Checkpoint | Training Data |
| ------------------------------- | ------------- |
| `linear_probe_synthclic.ckpt` | SynthCLIC |
| `linear_probe_synthbuster.ckpt` | SynthBuster+ |
| `linear_probe_cnnspot.ckpt` | CNNSpot |
| `linear_probe_combined.ckpt` | Combined |

### Concept Bottleneck Models (Interpretable)

| Checkpoint | Training Data |
| ------------------------------ | ------------- |
| `cm_antonyms_synthclic.ckpt` | SynthCLIC |
| `cm_antonyms_synthbuster.ckpt` | SynthBuster+ |
| `cm_antonyms_cnnspot.ckpt` | CNNSpot |
| `cm_antonyms_combined.ckpt` | Combined |

### Which Model to Use?

- **Modern AI images** (Midjourney, DALL-E, Stable Diffusion): Use `clip_orthogonal_synthclic.ckpt`
- **Best generalization**: Use `clip_orthogonal_combined.ckpt`
- **Interpretability needed**: Use `cm_antonyms_*.ckpt` models

### List Available Models

```python
from clip_cues import list_available_models

for name, info in list_available_models().items():
    print(f"{name}: {info['description']}")
```

## Datasets

Our datasets are available on HuggingFace Hub:

```python
from datasets import load_dataset

# SynthCLIC: Modern diffusion models (SD3, FLUX, Imagen3)
synthclic = load_dataset("marco-willi/synthclic")

# SynthBuster+: Diverse generators
synthbuster = load_dataset("marco-willi/synthbuster-plus")

# CNNSpot: ProGAN images
cnnspot = load_dataset("marco-willi/cnnspot-small")
```

### Dataset Examples

**SynthCLIC** - High-quality photographs paired with modern diffusion models:

![SynthCLIC Dataset](examples/synthclic_paired_samples_collage.png)

**SynthBuster+** - Diverse real and synthetic image pairs:

![Synthbuster Plus](examples/synthbuster-plus_paired_samples_collage.png)

### Generation Prompts

We include the prompts used to generate synthetic images:

- [SynthCLIC Prompts](data/datasets/synthclic/synthclic_prompts.parquet)
- [SynthBuster+ Prompts](data/datasets/synthbuster-plus/synthbuster_plus_prompts.parquet)

![Prompts for SynthCLIC](docs/images/synthclic_clic2020_real_images_with_prompts.png)

## Model Architecture

### CLIP Orthogonal Model

```
Input Image (336×336)
    │
    ▼
CLIP ViT-L/14 (frozen)
    │ 1024-dim embedding
    ▼
Linear Layer (1024 → 8)
    │ with activation orthogonality loss
    ▼
Linear Layer (8 → 1)
    │
    ▼
Sigmoid → [0, 1] probability
```

### Concept Bottleneck Model

```
Input Image (336×336)
    │
    ▼
CLIP ViT-L/14 (frozen)
    │ 1024-dim embedding
    ▼
Concept Selection (learnable gates)
    │ selects relevant concepts from 168 pairs
    ▼
Image-Concept Similarity × Gates
    │
    ▼
Linear Classifier → [0, 1] probability
```

## Concept Vocabulary

We release the antonyms vocabulary used for concept bottleneck models, containing **168 concept pairs** across 5 categories.

📄 [Download: antonyms.csv](data/vocabularies/antonyms.csv)

| Category | Count | Examples |
| --------- | ----- | ------------------------------------------ |
| Others | 67 | natural lighting, sharp focus, fine detail |
| Technique | 38 | shallow depth of field, long exposure |
| Color | 25 | high contrast, warm tones, saturated |
| Lense | 20 | bokeh, vignetting, chromatic aberration |
| Camera | 18 | high ISO noise, motion blur |

## Training

For training your own models, see the [Training Guide](scripts/TRAINING_GUIDE.md).

## Citation

```bibtex
@article{willi2026synthetic,
  title={Synthetic Image Detection with CLIP: Understanding and Assessing Predictive Cues},
  author={Willi, Marco and Mathys, Melanie and Graber, Michael},
  journal={arXiv preprint arXiv:2602.12381},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenAI CLIP](https://github.com/openai/CLIP) for the vision-language model
- [HuggingFace Transformers](https://huggingface.co/transformers) for model implementations
