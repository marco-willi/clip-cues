# Extract CLIP Image Embeddings

This script extracts CLIP image embeddings for datasets and saves them in a format suitable for training concept bottleneck models.

## Quick Start

### Extract embeddings for SynthCLIC

```bash
python scripts/extract_clip_embeddings.py synthclic
```

Output: `data/embeddings/synthclic_embeddings.pkl`

### Extract embeddings for SynthBuster+

```bash
python scripts/extract_clip_embeddings.py synthbuster-plus
```

Output: `data/embeddings/synthbuster-plus_embeddings.pkl`

### Extract embeddings for CNNSpot

```bash
python scripts/extract_clip_embeddings.py cnnspot
```

Output: `data/embeddings/cnnspot_embeddings.pkl`

## Options

- `--dataset-dir`: Directory containing datasets (default: `data/datasets`)
- `--output-dir`: Directory to save embeddings (default: `data/embeddings`)
- `--device`: Device for extraction, `cuda` or `cpu` (default: `cuda`)

### Examples

```bash
# Custom output directory
python scripts/extract_clip_embeddings.py synthclic --output-dir custom/path

# Use CPU instead of GPU
python scripts/extract_clip_embeddings.py synthbuster-plus --device cpu

# Custom input directory
python scripts/extract_clip_embeddings.py synthclic \
    --dataset-dir /path/to/datasets \
    --output-dir /path/to/embeddings
```

## Output Format

The script saves embeddings as a pickle file containing:

```python
{
    "embeddings": np.ndarray([num_images, 1024]),  # CLIP image embeddings
    "df": pd.DataFrame({
        "image_id": [...],      # Image identifiers
        "label": [...],         # 0 (real) or 1 (synthetic)
        "ds_name": [...],       # Dataset name
        "split": [...],         # "train", "validation", or "test"
        "source": [...],        # Image source (e.g., "CLIC2020", "Raise1K")
    }),
    "identifier": "synthclic_embeddings",  # Dataset identifier
    "model": "CLIP ViT-L/14",              # Model used
}
```

## Using Embeddings for Training

Once you have extracted embeddings, you can use them to train a concept bottleneck model:

```bash
python -m clip_cues.concept_modeling.train \
    --vocabulary-path data/vocabulary \
    --embeddings-path data/embeddings/synthclic_embeddings.pkl \
    --output-dir outputs/concept_model \
    --vocabulary antonym_pairs_v2 \
    --epochs 100
```

## Requirements

- Dataset must be downloaded first (use `scripts/download_dataset.py`)
- CUDA GPU recommended for faster extraction (CPU is supported but slower)
- Requires torch, torchvision, transformers, and datasets libraries

## Troubleshooting

### CUDA out of memory

If you get an out-of-memory error on GPU:

1. Use `--device cpu` (slower but no memory limit)
2. Modify the script to use batch processing

### Dataset not found

If you get "Dataset not found" error:

```bash
# Download the dataset first
python scripts/download_dataset.py synthclic
```

### Slow extraction

If extraction is very slow on CPU, ensure you have a CUDA-capable GPU and CUDA installed:

```bash
# Install with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
