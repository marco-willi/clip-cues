# Training a Concept Bottleneck Model

Complete guide to training a concept bottleneck model on SynthCLIC dataset.

## Quick Start

Run the automated training script:

```bash
./scripts/train_concept_model_synthclic.sh
```

This script will:

1. Create text embeddings from antonyms vocabulary
2. Extract CLIP image embeddings from SynthCLIC dataset
3. Train the concept bottleneck model

## Manual Step-by-Step

If you prefer to run each step manually:

### Step 1: Create Text Embeddings

Create CLIP text embeddings for the antonyms vocabulary:

```bash
python scripts/create_text_embeddings.py \
    --vocab-csv data/vocabularies/antonyms.csv \
    --output data/vocabulary/text_embeddings_antonym_pairs_v2.pt \
    --device cuda
```

**Output:** `data/vocabulary/text_embeddings_antonym_pairs_v2.pt`

- Contains 336 concept embeddings (168 pairs × 2)
- Each embedding is 1024-dimensional CLIP text features

### Step 2: Extract Image Embeddings

Extract CLIP image embeddings for the SynthCLIC dataset:

```bash
python scripts/extract_clip_embeddings.py synthclic --device cuda
```

**Output:** `data/embeddings/synthclic_embeddings.pkl`

- Contains CLIP image features for all SynthCLIC images
- Includes metadata (labels, splits, sources)

### Step 3: Train the Model

Train the concept bottleneck model:

```bash
python -m clip_cues.concept_modeling.train \
    --text-embeddings-path data/vocabulary/text_embeddings_antonym_pairs_v2.pt \
    --image-embeddings-path data/embeddings/synthclic_embeddings.pkl \
    --output-dir outputs/concept_model_synthclic \
    --ds-names synthclic \
    --train-splits train \
    --val-splits validation \
    --epochs 100 \
    --batch-size 256 \
    --lr 1e-3 \
    --tau 0.1 \
    --beta 1e-4 \
    --alpha 1e-4 \
    --device cuda
```

**Output:** `outputs/concept_model_synthclic/best_model.pt`

## Training on Other Datasets

### SynthBuster+

```bash
# Extract embeddings
python scripts/extract_clip_embeddings.py synthbuster-plus --device cuda

# Train model
python -m clip_cues.concept_modeling.train \
    --vocabulary-path data/vocabulary \
    --embeddings-path data/embeddings/synthbuster-plus_embeddings.pkl \
    --output-dir outputs/concept_model_synthbuster \
    --vocabulary antonym_pairs_v2 \
    --ds-names synthbuster_plus \
    --epochs 100
```

### Combined Datasets

```bash
# Extract embeddings for each dataset
python scripts/extract_clip_embeddings.py synthclic --device cuda
python scripts/extract_clip_embeddings.py synthbuster-plus --device cuda
python scripts/extract_clip_embeddings.py cnnspot --device cuda

# Merge embeddings (you'll need to create a script for this)
# OR train on each separately and evaluate cross-dataset
```

## Hyperparameters

| Parameter      | Default | Description                                             |
| -------------- | ------- | ------------------------------------------------------- |
| `--tau`        | 0.1     | Temperature for concept selection (lower = more sparse) |
| `--beta`       | 1e-4    | Weight for KL divergence sparsity loss                  |
| `--alpha`      | 1e-4    | Target sparsity level for concepts                      |
| `--lr`         | 1e-3    | Learning rate                                           |
| `--epochs`     | 100     | Number of training epochs                               |
| `--batch-size` | 256     | Training batch size                                     |

## Monitoring Training

The training script will output:

- Training loss, AUROC, and Average Precision
- Validation loss, AUROC, and Average Precision
- Best model is saved based on validation AUROC

## Troubleshooting

### CUDA out of memory

Reduce batch size:

```bash
python -m clip_cues.concept_modeling.train ... --batch-size 128
```

### Slow training on CPU

Extract embeddings with GPU first (much faster), then train:

```bash
# Extract with GPU
python scripts/extract_clip_embeddings.py synthclic --device cuda

# Train on CPU if needed
python -m clip_cues.concept_modeling.train ... --device cpu
```

### Dataset not found

Download the dataset first:

```bash
python scripts/download_dataset.py synthclic
```

## Expected Results

Training on SynthCLIC for 100 epochs should yield:

- Validation AUROC: ~0.95-0.98
- Training time: ~10-20 minutes on GPU (depends on hardware)
- Model size: ~5 MB

## Next Steps

After training, you can:

1. Evaluate the model on test set
2. Visualize selected concepts
3. Interpret predictions using concept attributions
4. Test cross-dataset generalization
