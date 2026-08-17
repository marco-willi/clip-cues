"""Consistent on-disk results layout for the revision experiments.

Every experiment run writes its raw results under a uniform tree::

    results/<experiment>/<run>/<run_id>/metrics.json        # scalar/dict metrics
    results/<experiment>/<run>/<run_id>/predictions.npz     # optional raw arrays (preds/labels/...)

where ``run_id`` is a timestamp tag (yyyymmddHHMM) generated once per script invocation.

so results are always persisted locally (not only in W&B) and are easy to find, diff, and
re-aggregate. Use `save_run_results(...)` from every run script.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

RESULTS_ROOT = "results"


def make_run_id() -> str:
    """Return a timestamp-based run identifier (yyyymmddHHMM)."""
    return datetime.now().strftime("%Y%m%d%H%M")


def _slug(text: str) -> str:
    """Filesystem-safe run/experiment component (keeps letters, digits, ._- and our '__to__')."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def _to_native(value: Any) -> Any:
    """Recursively convert numpy/torch scalars and containers to JSON-native types."""
    if isinstance(value, dict):
        return {str(k): _to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "item") and getattr(value, "ndim", None) == 0:  # 0-d tensor/array
        return value.item()
    return value


def save_run_results(
    experiment: str,
    run: str,
    metrics: dict,
    *,
    arrays: dict[str, np.ndarray] | None = None,
    base: str | Path = RESULTS_ROOT,
    run_id: str | None = None,
) -> Path:
    """Persist one run's results under ``base/<experiment>/<run>/<run_id>/``.

    Args:
        experiment: experiment name, e.g. ``e1_forensic`` / ``e2_beta_sweep`` / ``e3_clip_variants``.
        run: sub-experiment identifier, e.g. ``synthclic__to__cnnspot`` /
            ``synthclic__beta_1e-4`` / ``synthclic__clip_base_patch16``.
        metrics: JSON-serializable metrics (numpy scalars/dicts are converted automatically;
            numpy arrays are dropped from metrics.json — pass them via ``arrays`` instead).
        arrays: optional raw arrays saved to ``predictions.npz`` (e.g. predictions/labels/sources).
        base: results root (default ``results/``).
        run_id: timestamp tag (yyyymmddHHMM); auto-generated via ``make_run_id()`` if omitted.

    Returns:
        The run directory path.
    """
    rid = run_id or make_run_id()
    run_dir = Path(base) / _slug(experiment) / _slug(run) / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    serializable = {k: _to_native(v) for k, v in metrics.items() if not isinstance(v, np.ndarray)}
    (run_dir / "metrics.json").write_text(json.dumps(serializable, indent=2))

    if arrays:
        np.savez(run_dir / "predictions.npz", **arrays)

    return run_dir
