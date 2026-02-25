#!/bin/bash
# Train a Concept Bottleneck Model on SynthCLIC dataset
# This script extracts embeddings and trains the model

set -e  # Exit on error

echo "================================"
echo "Training Concept Model on SynthCLIC"
echo "================================"
echo ""

# Step 1: Create text embeddings from antonyms vocabulary
echo "Step 1/3: Creating text embeddings from antonyms vocabulary..."
if [ ! -f "data/embeddings/antonyms_embeddings.pt" ]; then
    python scripts/create_text_embeddings.py \
        --vocab-csv data/vocabularies/antonyms.csv \
        --output data/embeddings/antonyms_embeddings.pt \
        --device cuda
    echo "✓ Text embeddings created"
else
    echo "✓ Text embeddings already exist"
fi
echo ""

# Step 2: Extract CLIP image embeddings for SynthCLIC
echo "Step 2/3: Extracting CLIP image embeddings for SynthCLIC..."
if [ ! -f "data/embeddings/synthclic_embeddings.pkl" ]; then
    python scripts/extract_clip_embeddings.py synthclic --device cuda
    echo "✓ Image embeddings extracted"
else
    echo "✓ Image embeddings already exist"
fi
echo ""

# Step 3: Train the concept model
echo "Step 3/3: Training concept bottleneck model..."
python -m clip_cues.concept_modeling.train \
    --text-embeddings-path data/embeddings/antonyms_embeddings.pt \
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

echo ""
echo "================================"
echo "Training complete!"
echo "Model saved to: outputs/concept_model_synthclic/best_model.pt"
echo "================================"
