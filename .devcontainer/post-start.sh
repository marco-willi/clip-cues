#!/usr/bin/env bash
# Runs on every container start.
set -euo pipefail

if [ -f .venv/bin/activate ]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
fi

echo "Python: $(command -v python) ($(python --version 2>&1))"
python - <<'PY' 2>/dev/null || true
import torch
print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} (x{torch.cuda.device_count()})")
PY

cat <<'EOF'

Container ready.
  uv run python scripts/...    run in the project env (CPU torch by default)
  uv sync --extra all          refresh dependencies
  source .venv/bin/activate    or activate the env and use plain 'python'

GPU work (embedding extraction only): reopen in the .devcontainer/gpu configuration,
or run  uv sync --extra all --no-default-groups --group gpu  on a CUDA host.
EOF
