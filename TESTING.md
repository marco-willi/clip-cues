# Testing Guide

This document describes how to test the repository to ensure it's publication-ready.

## Quick Test

Run the comprehensive test suite:

```bash
python test_readme.py
```

This will test:
- ✓ Package can be imported
- ✓ All exported functions are available
- ✓ All 12 pre-trained models exist
- ✓ All referenced files exist (images, datasets, documentation)
- ✓ Models can be loaded successfully
- ✓ Prediction interface works (single and batch)
- ✓ All code examples from README work

Expected output: **49 tests passed, 0 failed**

## Fresh Installation Test

Test the installation process in a clean environment:

```bash
bash test_fresh_install.sh
```

This creates a temporary virtual environment, installs the package, and runs all README examples.

## Manual Testing

### 1. Test Installation

In a new terminal/environment:

```bash
# Create a virtual environment
python3 -m venv test_env
source test_env/bin/activate

# Install from local directory (or from GitHub once published)
pip install .

# Test import
python -c "from clip_cues import load_clip_classifier; print('Success!')"
```

### 2. Test Quick Start Example

```python
from clip_cues import load_clip_classifier

# Load model
model = load_clip_classifier("data/checkpoints/clip_orthogonal_synthclic.ckpt")

# Predict on a single image (use your own test image)
prob = model.predict("path/to/test_image.jpg")
print(f"Synthetic probability: {prob:.1%}")
print("Prediction:", "Synthetic" if prob > 0.5 else "Real")
```

### 3. Test Batch Inference

```python
from clip_cues import load_clip_classifier

model = load_clip_classifier("data/checkpoints/clip_orthogonal_synthclic.ckpt")

# Process multiple images
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
probs = model.predict_batch(image_paths, batch_size=32)

for path, prob in zip(image_paths, probs):
    print(f"{path}: {prob:.1%} synthetic")
```

### 4. Test Model Listing

```python
from clip_cues import list_available_models

for name, info in list_available_models().items():
    print(f"{name}: {info['description']}")
```

### 5. Test All Model Types

```python
from clip_cues import load_checkpoint

models_to_test = [
    "data/checkpoints/clip_orthogonal_synthclic.ckpt",
    "data/checkpoints/linear_probe_synthclic.ckpt",
    "data/checkpoints/cm_antonyms_synthclic.ckpt",
]

for model_path in models_to_test:
    model = load_checkpoint(model_path)
    print(f"✓ Loaded {model_path}")
```

## Testing Before Publication

### Pre-GitHub Release Checklist

1. **Run automated tests**:
   ```bash
   python test_readme.py
   ```

2. **Test fresh installation**:
   ```bash
   bash test_fresh_install.sh
   ```

3. **Verify all files exist**:
   ```bash
   # Check checkpoints
   ls -lh data/checkpoints/*.ckpt | wc -l  # Should be 12

   # Check datasets
   ls -lh data/datasets/*/

   # Check vocabulary
   cat data/vocabularies/antonyms.csv | wc -l  # Should be 169 (168 + header)

   # Check examples
   ls -lh examples/*.png
   ls -lh docs/images/*.png
   ```

4. **Test README links** (manually click through or use link checker)

5. **Review README examples** (copy-paste each code block and verify it works)

### After GitHub Release

Once the repository is public, test installation from GitHub:

```bash
# Create a fresh virtual environment
python3 -m venv publication_test
source publication_test/bin/activate

# Install from GitHub
pip install git+https://github.com/marco-willi/clip-cues.git

# Test
python -c "from clip_cues import load_clip_classifier; print('Installation successful!')"
```

### HuggingFace Dataset Testing

Test that datasets are accessible:

```python
from datasets import load_dataset

# Test each dataset
datasets_to_test = [
    "marco-willi/synthclic",
    "marco-willi/synthbuster-plus",
    "marco-willi/cnnspot-small",
]

for dataset_name in datasets_to_test:
    try:
        ds = load_dataset(dataset_name, split='train[:1]')  # Load just 1 sample
        print(f"✓ {dataset_name} is accessible")
    except Exception as e:
        print(f"✗ {dataset_name} failed: {e}")
```

## Common Issues and Solutions

### Issue: Models not found after installation

**Solution**: Models should be included in the package. Check that `data/checkpoints/` is properly packaged:

```python
import clip_cues
from pathlib import Path
package_path = Path(clip_cues.__file__).parent
checkpoints_path = package_path.parent / "data" / "checkpoints"
print(f"Checkpoints should be at: {checkpoints_path}")
print(f"Exists: {checkpoints_path.exists()}")
```

### Issue: Import errors

**Solution**: Ensure all dependencies are installed:

```bash
pip install torch torchvision transformers scikit-learn pillow numpy matplotlib seaborn
```

### Issue: CUDA/GPU errors

**Solution**: Models work on CPU by default. For GPU:

```python
model = load_clip_classifier("data/checkpoints/clip_orthogonal_synthclic.ckpt", device="cuda")
```

### Issue: Image loading errors

**Solution**: Ensure images are in a supported format (JPEG, PNG) and paths are correct:

```python
from PIL import Image
img = Image.open("path/to/image.jpg")
print(f"Image size: {img.size}, mode: {img.mode}")
```

## CI/CD Testing (Future)

For automated testing with GitHub Actions, create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest
    - name: Run tests
      run: python test_readme.py
```

## Test Coverage Summary

| Component | Test Coverage |
|-----------|--------------|
| Package installation | ✅ Automated |
| Model loading | ✅ Automated |
| Prediction interface | ✅ Automated |
| All checkpoints | ✅ Automated |
| Referenced files | ✅ Automated |
| README examples | ✅ Automated |
| Fresh install | ✅ Script available |
| HuggingFace datasets | ⚠️ Manual recommended |
| Cross-platform | ⚠️ Manual recommended |

## Reporting Issues

If tests fail:

1. Note the exact error message
2. Check Python version (`python --version`)
3. Check package version (`pip show clip-cues`)
4. List installed dependencies (`pip list`)
5. Verify checkpoint files exist
6. Check for any local modifications

For publication purposes, all automated tests should pass with 100% success rate.
