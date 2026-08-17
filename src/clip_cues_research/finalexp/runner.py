"""Run folders and ``run_meta.json`` — which script produced which result.

The manifest (:mod:`~clip_cues_research.finalexp.snapshot`) traces *inputs*. This traces *code*.
Every F-experiment wraps itself in :class:`Run`, which creates the output folder and writes a
``run_meta.json`` recording the script, the full command line, the git commit, the snapshot
manifest version and the sha256 of every artifact the run read, package versions, the host (so a
Lambda result stays distinguishable after it is synced back), and the wall time.

A result without a ``run_meta.json`` is not a result.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from clip_cues_research.finalexp import data as D

EXPERIMENTS_ROOT = Path("reproduction/experiments/final_consolidation")


def _git() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:  # pragma: no cover
        return {"commit": "unknown", "dirty": None}


def _versions() -> dict:
    import sklearn
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
    }


def _host() -> str:
    """``local`` or a Lambda instance tag — so remote and local artifacts stay distinguishable."""
    name = socket.gethostname()
    return name if name else "unknown"


def to_native(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays to JSON-native types."""
    if isinstance(value, dict):
        return {str(k): to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(v) for v in value]
    if isinstance(value, np.ndarray):
        return [to_native(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_native(payload), indent=2) + "\n")
    return path


class Run:
    """One experiment run: an output folder plus its provenance record."""

    def __init__(self, experiment: str, subdir: str = "artifacts", inputs: list[str] | None = None):
        """
        Args:
            experiment: folder name under ``reproduction/experiments/final_consolidation/`` (e.g. ``F1-canonical-stability``).
            subdir: sub-folder for this run (``artifacts`` or ``runs/seed123``).
            inputs: snapshot artifact ids the run reads — their shas are pinned in ``run_meta.json``.
        """
        self.experiment = experiment
        self.dir = EXPERIMENTS_ROOT / experiment / subdir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.inputs = list(inputs or [])
        # Wall clock for the timestamp, monotonic for the duration: a mid-run NTP adjustment
        # otherwise yields a negative elapsed time (observed).
        self._started = time.time()
        self._t0 = time.monotonic()
        self._extra: dict[str, Any] = {}

    def note(self, **kwargs: Any) -> None:
        """Attach extra fields to this run's provenance record."""
        self._extra |= kwargs

    def path(self, name: str) -> Path:
        return self.dir / name

    def save_json(self, name: str, payload: Any) -> Path:
        return write_json(self.dir / name, payload)

    def save_csv(self, name: str, df) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        out = self.dir / name
        df.to_csv(out, index=False)
        return out

    def save_npz(self, name: str, **arrays: np.ndarray) -> Path:
        out = self.dir / name
        np.savez_compressed(out, **arrays)
        return out

    def finish(self) -> Path:
        """Write ``run_meta.json``. Called automatically by :func:`run_context`."""
        duration = time.monotonic() - self._t0
        meta = {
            "experiment": self.experiment,
            "script": sys.argv[0],
            "argv": sys.argv,
            "git": _git(),
            "snapshot_manifest_version": D.manifest_version(),
            "inputs": D.input_shas(*self.inputs),
            "package_versions": _versions(),
            "host": _host(),
            "started_at": datetime.fromtimestamp(self._started, timezone.utc).isoformat(
                timespec="seconds"
            ),
            "duration_s": round(duration, 3),
            **self._extra,
        }
        path = write_json(self.dir / "run_meta.json", meta)
        # Surfacing the wall time is how the ~20-minute Lambda threshold gets measured rather than
        # guessed (PLAN_FINAL_CONSOLIDATION.md, "Where things run").
        print(f"  [{self.experiment}/{self.dir.name}] {duration:.1f}s -> {self.dir}")
        return path


@contextmanager
def run_context(
    experiment: str, subdir: str = "artifacts", inputs: list[str] | None = None
) -> Iterator[Run]:
    """Context manager that always writes ``run_meta.json``, including on failure."""
    run = Run(experiment, subdir, inputs)
    try:
        yield run
    finally:
        run.finish()
