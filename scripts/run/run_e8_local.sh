#!/usr/bin/env bash
# E8 Tier-A local driver: ortho seed-stability (init-hypothesis) + concept beta-sensitivity.
# Runs on the local GPU against cached embeddings. Idempotent-ish: each invocation makes new run_ids.
set -euo pipefail
cd "$(dirname "$0")/../.."
EMB=data/embeddings
PY="uv run python scripts/run/run_interpretation_stability.py"

echo "=== ORTHO seed stability (10 seeds, both regimes, SynthCLIC + CNNSpot) ==="
for ds_pkl in "synthclic:$EMB/synthclic_clip_large_patch14.pkl" "cnnspot:$EMB/cnnspot_clip_large_patch14.pkl"; do
  ds=${ds_pkl%%:*}; pkl=${ds_pkl#*:}
  for regime in vary-init vary-shuffle; do
    echo ">>> ortho $ds $regime"
    $PY --mode ortho --embeddings "$pkl" --dataset "$ds" --regime "$regime" \
        --seeds 0 1 2 3 4 5 6 7 8 9 --device cuda --top-k 4
  done
done

echo "=== CONCEPT seed+beta stability (SynthCLIC, 5 betas x 5 seeds) ==="
$PY --mode concept \
    --image-embeddings "$EMB/synthclic_projected_embeddings.pkl" \
    --text-embeddings "$EMB/antonyms_diff_embeddings.pt" \
    --dataset synthclic --betas 1e-5 1e-4 3e-4 1e-3 1e-2 --seeds 0 1 2 3 4 \
    --device cuda --top-k 30

echo "=== ALL TIER-A RUNS DONE ==="
