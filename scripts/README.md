# Scripts

This directory contains utility scripts for the CLIP-Cues project.

## Dataset Download Scripts

### download_dataset.py

General script for downloading any supported dataset:

```bash
# Download SynthCLIC dataset
python scripts/download_dataset.py synthclic

# Download synthbuster-plus+ dataset
python scripts/download_dataset.py synthbuster-plus

# Download CNNSpot dataset
python scripts/download_dataset.py cnnspot

# Download to custom location
python scripts/download_dataset.py synthclic --output-dir /path/to/datasets

# Use streaming mode (doesn't download full dataset)
python scripts/download_dataset.py synthclic --streaming
```

**Supported datasets:**

- `synthclic` - marco-willi/synthclic
- `synthbuster-plus` - marco-willi/synthbuster-plus
- `cnnspot` - marco-willi/cnnspot

### download_synthclic.py

Dedicated script for downloading the SynthCLIC dataset:

```bash
# Download to default location (data/datasets/)
python scripts/download_synthclic.py

# Download to custom location
python scripts/download_synthclic.py --output-dir /path/to/datasets

# Use custom cache directory
python scripts/download_synthclic.py --cache-dir /path/to/cache
```

## Model Validation Scripts

### validate_model.py

Validate a trained model on a dataset test split with comprehensive metrics:

```bash
# Validate model on SynthCLIC test set
python scripts/validate_model.py \
    data/checkpoints/linear_probe_combined.ckpt \
    data/datasets/synthclic

# Use validation split instead of test
python scripts/validate_model.py \
    data/checkpoints/linear_probe_combined.ckpt \
    data/datasets/synthclic \
    --split validation

# Specify output location
python scripts/validate_model.py \
    data/checkpoints/linear_probe_combined.ckpt \
    data/datasets/synthclic \
    --output results/my_validation.json

# Use CPU instead of GPU
python scripts/validate_model.py \
    data/checkpoints/linear_probe_combined.ckpt \
    data/datasets/synthclic \
    --device cpu

# Adjust batch size
python scripts/validate_model.py \
    data/checkpoints/linear_probe_combined.ckpt \
    data/datasets/synthclic \
    --batch-size 64
```

**Features:**

- Calculates overall metrics: Accuracy, AUC, AP, mAP
- Computes per-source metrics (each synthetic source vs real)
  - Example: Imagen3 vs CLIC2020 (real)
- Generates confusion matrices
- Saves detailed results to JSON
- Exports predictions for further analysis

**Output:**

- `results/<dataset>_<checkpoint>.json` - Metrics summary
- `results/<dataset>_<checkpoint>_predictions.npz` - Detailed predictions

## Requirements

Make sure you have the `datasets` library installed:

```bash
pip install datasets
```

Or install the full project dependencies:

```bash
pip install -e .
```

## Dataset Storage

By default, datasets are downloaded to `data/datasets/<dataset_name>/` and saved in Hugging Face's Arrow format for fast loading.

To load a downloaded dataset in your code:

```python
from datasets import load_from_disk

dataset = load_from_disk("data/datasets/synthclic")
```
