#!/bin/bash
# Example validation script for CLIP-Cues models

echo "==================================================================="
echo "CLIP-Cues Model Validation Examples"
echo "==================================================================="
echo ""

# Example 1: Validate linear probe on SynthCLIC
echo "Example 1: Validate linear probe model on SynthCLIC test set"
echo "-------------------------------------------------------------------"
python scripts/validate_model.py \
    data/checkpoints/linear_probe_combined.ckpt \
    data/datasets/synthclic \
    --batch-size 32

echo ""
echo ""

# Example 2: Validate CLIP orthogonal model on SynthCLIC
echo "Example 2: Validate CLIP orthogonal model on SynthCLIC test set"
echo "-------------------------------------------------------------------"
python scripts/validate_model.py \
    data/checkpoints/clip_orthogonal_combined.ckpt \
    data/datasets/synthclic \
    --batch-size 32

echo ""
echo ""

# Example 3: Compare different checkpoints on validation set
echo "Example 3: Validate on validation split for model comparison"
echo "-------------------------------------------------------------------"
for ckpt in data/checkpoints/linear_probe_*.ckpt; do
    echo "Testing: $(basename $ckpt)"
    python scripts/validate_model.py \
        "$ckpt" \
        data/datasets/synthclic \
        --split validation \
        --batch-size 32 \
        --output "results/validation_$(basename $ckpt .ckpt).json"
    echo ""
done

echo ""
echo "==================================================================="
echo "Validation complete! Check the 'results/' directory for outputs."
echo "==================================================================="
