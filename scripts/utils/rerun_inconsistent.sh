#!/usr/bin/env bash
# Re-run the inconsistent revision experiments (E3, E2, E5) + recompute E4 under the corrected
# config (docs/REVISION_CONFIG_AUDIT.md) and Convention-A metric. Idempotent: each run writes a
# fresh timestamped run_id; aggregation picks the latest. Progress is mirrored to RERUN_PROGRESS.md.
# Usage: bash scripts/utils/rerun_inconsistent.sh >> /tmp/rerun.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
EMB=data/embeddings
PROG=RERUN_PROGRESS.md
DEV=${DEV:-cuda}

mark() { echo "$(date '+%F %T') STATUS $1" | tee -a "$PROG"; }
run() { echo "----- $(date '+%F %T') RUN: $* -----"; "$@"; echo "----- exit=$? -----"; }

UV="uv run --no-sync python"

# ── E3: linear-probe cross-dataset matrix (cached embeddings, corrected config) ──
mark "E3 START"
for B in clip_base_patch16 clip_base_patch32 clip_large_patch14; do
  EVAL="synthclic=$EMB/synthclic_${B}.pkl,cnnspot=$EMB/cnnspot_${B}.pkl,synthbuster-plus=$EMB/synthbuster-plus_${B}.pkl"
  for T in synthclic cnnspot synthbuster-plus; do
    run $UV scripts/run/run_linear_probe.py --embeddings $EMB/${T}_${B}.pkl \
      --backbone $B --dataset $T --device $DEV --no-wandb --eval-embeddings "$EVAL"
  done
done
mark "E3 DONE"

# ── E2: beta sensitivity sweep (168 antonym difference-directions, corrected config) ──
mark "E2 START"
for BETA in 1e-5 1e-4 3e-4 1e-3 1e-2; do
  run $UV scripts/run/run_beta_sweep.py --beta $BETA --epochs 1000 \
    --vocabulary antonyms --device $DEV --no-wandb
done
mark "E2 DONE"

# ── E5: activation- vs weight-orthogonality ablation (corrected config) ──
mark "E5 START"
run $UV scripts/run/run_orthogonality_ablation.py \
  --embeddings $EMB/synthclic_l14_local.pkl --dataset synthclic --device $DEV
mark "E5 DONE"

# ── E4: cross-family recompute (depends on E3 L/14 cnnspot→synthclic predictions) ──
mark "E4 START"
run $UV scripts/run/run_cross_family_analysis.py
mark "E4 DONE"

mark "ALL EXPERIMENTS DONE — proceed to EXPORT rebuild (agent-side)"
