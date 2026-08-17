"""The frozen input snapshot for F1-F7: manifest-mediated, checksum-verified artifact access.

Why this exists (PLAN_FINAL_CONSOLIDATION.md §Context 5): `data/` currently holds artifacts that are
**silently substitutable** — three SynthCLIC pooler pickles with identical shape/splits/id-order, and
a canonical vs a *retracted* (double-projected W-squared) vocabulary that are both (168, 768) float32.
Shape assertions cannot catch a wrong path; only checksums can.

So every F-experiment input is copied into ``reproduction/experiments/data/``, recorded in ``manifest.json`` with
its sha256 and a declared embedding ``space``, and reached **only** through :func:`get` — which
verifies the checksum on load. A guard test forbids literal input paths anywhere else in the
``finalexp`` code.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SNAPSHOT = Path("reproduction/experiments/data")
MANIFEST = SNAPSHOT / "manifest.json"

# Declared embedding spaces. The `W2_LEGACY` tag exists so a retracted artifact can be *named* in the
# manifest without ever being loadable into a text-space computation (see `require_space`).
SPACE_POOLER = "pooler_1024"
SPACE_CANON = "crossmodal_768_canon"
SPACE_W2_LEGACY = "crossmodal_768_W2_LEGACY"
SPACE_NA = "n/a"


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """Full sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def fingerprint(arr: np.ndarray) -> dict[str, Any]:
    """Cheap structural digest of an array — catches a swapped-then-renamed file.

    Row norms and a few leading values are stable under a re-save (which changes the sha256) but
    differ between genuinely different artifacts, so this complements the checksum rather than
    duplicating it.
    """
    a = np.asarray(arr, dtype=np.float64).reshape(len(arr), -1)
    norms = np.linalg.norm(a, axis=1)
    return {
        "row_norm_mean": round(float(norms.mean()), 8),
        "row_norm_std": round(float(norms.std()), 8),
        "first_row_head": [round(float(v), 8) for v in a[0, :8]],
    }


@dataclass(frozen=True)
class Record:
    """One manifest entry."""

    id: str
    path: str
    kind: str
    space: str
    sha256: str
    bytes: int
    raw: dict

    @property
    def full_path(self) -> Path:
        return SNAPSHOT / self.path


def load_manifest(manifest: Path | None = None) -> dict[str, Record]:
    """Read ``manifest.json`` into ``{id: Record}``.

    The default is resolved at *call* time from the module-level ``MANIFEST`` so tests (and any
    caller pointing at an alternate snapshot) can redirect it.
    """
    manifest = manifest if manifest is not None else MANIFEST
    if not manifest.exists():
        raise FileNotFoundError(
            f"No snapshot manifest at {manifest}. Build it first:\n"
            f"    uv run python scripts/finalexp/build_data_snapshot.py"
        )
    doc = json.loads(manifest.read_text())
    return {
        r["id"]: Record(
            id=r["id"],
            path=r["path"],
            kind=r["kind"],
            space=r["space"],
            sha256=r["sha256"],
            bytes=r["bytes"],
            raw=r,
        )
        for r in doc["artifacts"]
    }


def manifest_version(manifest: Path | None = None) -> str:
    """The snapshot's version string — recorded by every run for provenance."""
    return json.loads((manifest if manifest is not None else MANIFEST).read_text())["version"]


def record(artifact_id: str) -> Record:
    """Manifest record for ``artifact_id`` (no file access)."""
    recs = load_manifest()
    if artifact_id not in recs:
        raise KeyError(f"Unknown snapshot id {artifact_id!r}. Known: {sorted(recs)}")
    return recs[artifact_id]


# Artifacts verified in this process: (path, mtime_ns, size) -> sha256 already checked. Hashing a
# 45 MB frame on every load makes repeated access dominate runtime. The cached stamp includes the
# file's mtime and size *and the manifest's expected sha256*: a file replaced mid-run, or a manifest
# whose expectation changed (a re-registered artifact, a different snapshot), both force a re-check
# instead of inheriting a stale verdict.
_VERIFIED: dict[str, tuple[int, int, str]] = {}


def resolve(artifact_id: str, *, verify: bool = True) -> Path:
    """Verified path to a snapshot artifact — checks sha256 before returning."""
    rec = record(artifact_id)
    if not rec.full_path.exists():
        raise FileNotFoundError(
            f"Snapshot artifact {artifact_id!r} missing at {rec.full_path}. "
            f"Rebuild: uv run python scripts/finalexp/build_data_snapshot.py"
        )
    if not verify:
        return rec.full_path

    st = rec.full_path.stat()
    stamp = (st.st_mtime_ns, st.st_size, rec.sha256)
    if _VERIFIED.get(artifact_id) == stamp:
        return rec.full_path

    actual = sha256(rec.full_path)
    if actual != rec.sha256:
        raise ValueError(
            f"CHECKSUM MISMATCH for {artifact_id!r} at {rec.full_path}\n"
            f"  manifest: {rec.sha256}\n  on disk:  {actual}\n"
            f"The snapshot has been modified or the wrong file is in place. Refusing to load."
        )
    _VERIFIED[artifact_id] = stamp
    return rec.full_path


RELEASE_MANIFEST = SNAPSHOT / "release_manifest.json"


def _release_record(artifact_id: str) -> dict | None:
    """Release-manifest entry for ``artifact_id``, if a release is materialised here."""
    if not RELEASE_MANIFEST.exists():
        return None
    import json

    recs = json.loads(RELEASE_MANIFEST.read_text()).get("artifacts", {})
    return recs.get(artifact_id)


def resolve_any(artifact_id: str) -> Path:
    """Verified path to an artifact in **either** the built or the released format.

    The built form (pandas pickle / torch ``.pt``) is what the snapshot builder writes locally and
    what every ``run_meta.json`` hash refers to. The released form is object-free ``.npz`` — see
    :mod:`clip_cues_research.finalexp.release` — and is verified against ``release_sha256``, its
    own hash, because a converted file cannot have the same digest as its source.
    """
    rec = record(artifact_id)
    if rec.full_path.exists():
        return resolve(artifact_id)

    rel = _release_record(artifact_id)
    if rel is None:
        raise FileNotFoundError(
            f"Snapshot artifact {artifact_id!r} missing at {rec.full_path}, and no release "
            f"manifest at {RELEASE_MANIFEST}.\n"
            f"Fetch the released snapshot:  make finalexp-fetch\n"
            f"or rebuild it locally:        make finalexp-data"
        )
    path = SNAPSHOT / rel["release_path"]
    if not path.exists():
        raise FileNotFoundError(
            f"Snapshot artifact {artifact_id!r} missing in both formats "
            f"({rec.full_path}, {path}). Fetch it: make finalexp-fetch"
        )
    actual = sha256(path)
    if actual != rel["release_sha256"]:
        raise ValueError(
            f"CHECKSUM MISMATCH for released {artifact_id!r} at {path}\n"
            f"  release manifest: {rel['release_sha256']}\n  on disk:          {actual}\n"
            f"Refusing to load. Re-fetch: make finalexp-fetch"
        )
    return path


def require_space(artifact_id: str, expected: str) -> None:
    """Assert an artifact's declared embedding space.

    The direct guard against the 2026-07-17 double-projection bug: anything entering a text-space
    computation must be tagged ``crossmodal_768_canon``.
    """
    got = record(artifact_id).space
    if got != expected:
        raise ValueError(
            f"Space mismatch for {artifact_id!r}: declared {got!r}, required {expected!r}."
            + (
                "  This artifact is in the RETRACTED double-projected (W-squared) text space and "
                "must never be used — see docs/revision_state/INTERPRETATION.md §1."
                if got == SPACE_W2_LEGACY
                else ""
            )
        )


# ── typed loaders ────────────────────────────────────────────────────────────────────────────
@dataclass
class Frame:
    """An embedding frame: per-image metadata + the embedding matrix, aligned **by position**.

    ``image_id`` is *not* unique in SynthCLIC/SynthBuster (the real image and every generator's
    synthetic share one id), so every join in this package is positional.
    """

    df: pd.DataFrame
    emb: np.ndarray
    artifact_id: str

    def split(self, name: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """``(embeddings, labels, sub-frame)`` for one split, positionally aligned."""
        m = (self.df["split"] == name).to_numpy()
        sub = self.df[m].reset_index(drop=True)
        return self.emb[m], sub["label"].to_numpy().astype(int), sub


def get_frame(artifact_id: str, expected_space: str | None = None) -> Frame:
    """Load a pickled embedding frame (``{df, embeddings}``) from the snapshot."""
    if expected_space:
        require_space(artifact_id, expected_space)
    path = resolve_any(artifact_id)
    if path.suffix == ".npz":
        from clip_cues_research.finalexp.release import npz_to_frame

        d = npz_to_frame(path)
    else:
        with open(path, "rb") as f:
            d = pickle.load(f)
    return Frame(d["df"].reset_index(drop=True), np.asarray(d["embeddings"]), artifact_id)


def get_vocab(artifact_id: str) -> tuple[np.ndarray, list[str]]:
    """Load a cue vocabulary as ``(unit-row [K, 768], names)``. Canonical text space enforced."""
    import torch

    require_space(artifact_id, SPACE_CANON)
    path = resolve_any(artifact_id)
    if path.suffix == ".npz":
        from clip_cues_research.finalexp.release import npz_to_vocab

        a = npz_to_vocab(path)
    else:
        a = torch.load(path, map_location="cpu", weights_only=False)
    emb = np.asarray(a["embeddings"], dtype=np.float64)
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    return emb, list(a["vocabulary"])


def get_array(artifact_id: str) -> np.ndarray:
    """Load a ``.npy`` array (e.g. the visual projection matrix) from the snapshot."""
    return np.load(resolve_any(artifact_id))


def get_npz(artifact_id: str) -> dict[str, np.ndarray]:
    """Load a ``.npz`` bundle (e.g. cached cue scores) from the snapshot."""
    with np.load(resolve_any(artifact_id), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def get_predictions(artifact_id: str) -> pd.DataFrame:
    """Load a per-image predictions parquet (``image_id, label, score, source, …``).

    Used by the appendix per-generator table, whose cells are re-aggregations of predictions the
    E3 cross-dataset runs already wrote. Two E3 generations share one file-name pattern but
    evaluate CNNSpot on different populations, so these must come through the checksum.
    """
    return pd.read_parquet(resolve(artifact_id))


def get_json(artifact_id: str) -> dict:
    """Load a JSON artifact (e.g. the F1 regression anchor) from the snapshot."""
    return json.loads(resolve(artifact_id).read_text())


def get_checkpoint(artifact_id: str) -> dict:
    """Load a Lightning-style checkpoint's ``state_dict`` from the snapshot."""
    import torch

    d = torch.load(resolve(artifact_id), map_location="cpu", weights_only=False)
    return d.get("state_dict", d)


def input_shas(*artifact_ids: str) -> dict[str, str]:
    """``{id: sha256}`` for the artifacts a run consumed — embedded in ``run_meta.json``."""
    recs = load_manifest()
    return {i: recs[i].sha256 for i in artifact_ids if i in recs}
