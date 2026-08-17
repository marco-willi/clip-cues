#!/usr/bin/env bash
# Runs once, after the container is created. Creates .venv via uv.
#
#   post-create.sh          CPU torch (default) - enough for every experiment and figure
#   post-create.sh --gpu    cu128 torch - only needed to extract CLIP embeddings from images
set -euo pipefail

SYNC_ARGS=(--extra all)
if [ "${1:-}" = "--gpu" ]; then
    SYNC_ARGS+=(--no-default-groups --group gpu)
    echo "post-create: syncing with the GPU (cu128) torch group"
else
    echo "post-create: syncing with the CPU torch group (default)"
fi

command -v uv >/dev/null 2>&1 || pip install uv

# Creates .venv with clip_cues + clip_cues_research and the train/dev/research extras,
# pinned by uv.lock.
uv sync "${SYNC_ARGS[@]}"

# shellcheck source=/dev/null
source .venv/bin/activate

# Jupyter kernel, registered from the synced env.
python -m ipykernel install --name=clip-cues \
    --display-name="Python (clip-cues)" 2>/dev/null || true

if [ -f ".pre-commit-config.yaml" ]; then
    pre-commit install || echo "post-create: pre-commit install failed (continuing)"
fi

# Optional: the Claude Code CLI, as the previous devcontainer setup did. Non-fatal.
if ! command -v claude >/dev/null 2>&1; then
    curl -fsSL https://claude.ai/install.sh | bash || echo "post-create: claude install skipped"
fi

# install HF CLI
curl -LsSf https://hf.co/cli/install.sh | bash

echo "post-create: done."
