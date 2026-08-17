"""Distribution format for the input snapshot.

The snapshot is *built* with pandas pickles and torch ``.pt`` files, which are convenient locally
and wrong for distribution: a pickle written under Python 3.10 does not load under 3.12 (pandas
block-manager incompatibility), and unpickling executes arbitrary code from whoever produced the
file. Both problems disappear if the released artifacts are plain ``.npz``.

So the release carries every array-bearing artifact as ``.npz`` with no object dtypes, alongside a
``release_manifest.json`` that records **both** hashes per artifact:

``sha256``
    the hash of the *built* artifact, which every ``run_meta.json`` in
    ``reproduction/experiments/final_consolidation/`` already cites. This is the provenance anchor and must
    never be rewritten.
``release_sha256``
    the hash of the distributed file, which is what a download is verified against.

Formats that are already safe and portable — ``.npz``, ``.npy``, ``.csv``, ``.json``, ``.parquet``
and our own small ``.ckpt`` state dicts — are released unchanged, and their two hashes are equal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Artifact kinds stored as pandas pickles: ``{"df": DataFrame, "embeddings": ndarray}``.
FRAME_KINDS = {"pooler_embeddings", "projected_embeddings", "projected_embeddings_reference"}
#: Artifact kinds stored as torch files: ``{"embeddings": Tensor, "vocabulary": list[str]}``.
VOCAB_KINDS = {"cue_vocabulary"}

#: Column prefix for the metadata columns packed alongside the embedding matrix.
_COL = "col__"


def needs_conversion(kind: str, path: str) -> bool:
    """True if this artifact is in a format that must not be distributed as-is."""
    return kind in FRAME_KINDS or kind in VOCAB_KINDS or path.endswith((".pkl", ".pt"))


def release_name(path: str) -> str:
    """Path of an artifact inside the release, converting the extension where needed."""
    p = Path(path)
    return str(p.with_suffix(".npz")) if p.suffix in {".pkl", ".pt"} else str(p)


# ── frames ───────────────────────────────────────────────────────────────────────────────────
def frame_to_npz(obj: dict[str, Any], dest: Path) -> None:
    """Write ``{"df", "embeddings"}`` as an object-free ``.npz``.

    Metadata columns are stored as fixed-width unicode or numeric arrays under a ``col__`` prefix,
    so the file loads with ``allow_pickle=False`` and carries no executable payload.
    """
    df, emb = obj["df"].reset_index(drop=True), np.asarray(obj["embeddings"])
    arrays: dict[str, np.ndarray] = {
        "embeddings": emb,
        "columns": np.asarray(list(df.columns), dtype=str),
    }
    for c in df.columns:
        s = df[c]
        # `Series.astype(str).to_numpy()` yields dtype=object, which np.savez silently pickles and
        # np.load(allow_pickle=False) then refuses. Force a fixed-width unicode dtype instead.
        arrays[f"{_COL}{c}"] = (
            s.to_numpy() if pd.api.types.is_numeric_dtype(s) else s.astype(str).to_numpy(dtype=str)
        )
    bad = [k for k, v in arrays.items() if v.dtype == object]
    if bad:
        raise TypeError(f"object dtype would be pickled into the release: {bad}")
    np.savez_compressed(dest, **arrays)


def npz_to_frame(path: Path) -> dict[str, Any]:
    """Inverse of :func:`frame_to_npz`, preserving column order."""
    with np.load(path, allow_pickle=False) as z:
        cols = [str(c) for c in z["columns"]]
        df = pd.DataFrame({c: z[f"{_COL}{c}"] for c in cols})[cols]
        return {"df": df, "embeddings": z["embeddings"]}


# ── vocabularies ─────────────────────────────────────────────────────────────────────────────
def vocab_to_npz(obj: dict[str, Any], dest: Path) -> None:
    """Write ``{"embeddings", "vocabulary"}`` as an object-free ``.npz``."""
    emb = np.asarray(obj["embeddings"], dtype=np.float32)
    np.savez_compressed(
        dest, embeddings=emb, vocabulary=np.asarray(list(obj["vocabulary"]), dtype=str)
    )


def npz_to_vocab(path: Path) -> dict[str, Any]:
    """Inverse of :func:`vocab_to_npz`."""
    with np.load(path, allow_pickle=False) as z:
        return {"embeddings": z["embeddings"], "vocabulary": [str(v) for v in z["vocabulary"]]}
