"""Snapshot maintenance: describe, register and render ``reproduction/experiments/data/`` artifacts.

The *write* side of the snapshot (:mod:`~clip_cues_research.finalexp.data` is the read side).
Used by ``scripts/finalexp/build_data_snapshot.py`` (initial copy),
``scripts/finalexp/prepare_features.py`` (registers derived features) and
``scripts/finalexp/verify_data_snapshot.py``.
"""

from __future__ import annotations

import json
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from clip_cues_research.finalexp.data import MANIFEST, SNAPSHOT, fingerprint, sha256


def git_commit() -> dict:
    """Current commit + dirty flag, for provenance."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return {"commit": sha, "dirty": dirty}
    except Exception:  # pragma: no cover - git may be unavailable
        return {"commit": "unknown", "dirty": None}


def describe(path: Path, kind: str) -> dict[str, Any]:
    """Structural description of an artifact: shape, dtype, splits, vocab size, fingerprint.

    Recomputed by the verifier and compared against the manifest, so a file that was swapped for a
    different artifact of the same size is caught even before the checksum is considered.
    """
    import torch

    out: dict[str, Any] = {}
    if kind in ("pooler_embeddings", "projected_embeddings", "projected_embeddings_reference"):
        # A fetched snapshot holds the released .npz form of this artifact; describe either.
        if path.suffix == ".npz":
            from clip_cues_research.finalexp.release import npz_to_frame

            d = npz_to_frame(path)
        else:
            with open(path, "rb") as f:
                d = pickle.load(f)
        df, emb = d["df"], np.asarray(d["embeddings"])
        out |= {
            "shape": list(emb.shape),
            "dtype": str(emb.dtype),
            "n_rows": int(len(df)),
            "columns": [str(c) for c in df.columns],
            "fingerprint": fingerprint(emb),
        }
        # Eval-only frames (CF-Eval) carry no `split` column — they are one evaluation population.
        if "split" in df.columns:
            out["split_counts"] = {str(k): int(v) for k, v in df["split"].value_counts().items()}
    elif kind == "cue_vocabulary":
        if path.suffix == ".npz":
            from clip_cues_research.finalexp.release import npz_to_vocab

            a = npz_to_vocab(path)
        else:
            a = torch.load(path, map_location="cpu", weights_only=False)
        emb = np.asarray(a["embeddings"])
        out |= {
            "shape": list(emb.shape),
            "dtype": str(emb.dtype),
            "n_vocab": int(len(a["vocabulary"])),
            "fingerprint": fingerprint(emb),
        }
    elif kind == "cue_scores":
        with np.load(path, allow_pickle=False) as z:
            scores = z["scores"]
            out |= {
                "shape": list(scores.shape),
                "dtype": str(scores.dtype),
                "n_rows": int(scores.shape[0]),
                "fingerprint": fingerprint(scores),
            }
    elif kind == "projection_matrix":
        arr = np.load(path)
        out |= {"shape": list(arr.shape), "dtype": str(arr.dtype), "fingerprint": fingerprint(arr)}
    elif kind == "checkpoint":
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        out |= {"state_dict_keys": sorted(str(k) for k in sd)}
    elif kind == "reference_metrics":
        out |= {"content": json.loads(path.read_text())}
    elif kind == "vocabulary_terms":
        import pandas as pd

        df = pd.read_csv(path)
        out |= {"n_rows": int(len(df)), "columns": [str(c) for c in df.columns]}
    elif kind == "ranking":
        import pandas as pd

        df = pd.read_csv(path)
        out |= {"n_rows": int(len(df)), "columns": [str(c) for c in df.columns]}
    elif kind == "predictions":
        import pandas as pd

        df = pd.read_parquet(path)
        # `n_rows` and `n_generators` are the two fields that distinguish the *two generations* of
        # E3 CNNSpot evaluations — 4,000 rows / 20 generators (cnnspot-small) vs 108,310 / 21 (the
        # full CNNSpot benchmark test set). They carry the same file-name pattern, so without this
        # the wrong evaluation population is a rename away.
        fakes = df[df["label"] == 1]
        out |= {
            "n_rows": int(len(df)),
            "columns": [str(c) for c in df.columns],
            "n_generators": int(fakes["source"].nunique()),
            "n_fake": int(len(fakes)),
            "n_real": int(len(df) - len(fakes)),
            "fingerprint": fingerprint(df["score"].to_numpy().reshape(-1, 1)),
        }
    return out


def read_doc() -> dict:
    return json.loads(MANIFEST.read_text())


def write_doc(doc: dict) -> None:
    """Persist manifest.json and re-render MANIFEST.md."""
    MANIFEST.write_text(json.dumps(doc, indent=2) + "\n")
    (SNAPSHOT / "MANIFEST.md").write_text(render_markdown(doc))


def register_artifact(
    *,
    artifact_id: str,
    path: str,
    kind: str,
    space: str,
    used_by: list[str],
    provenance: str,
    derived_from: dict[str, str] | None = None,
) -> dict:
    """Add (or replace) a manifest record for an artifact already written into the snapshot.

    ``derived_from`` is the ``{input_id: sha256}`` map of everything that fed this artifact, so a
    derived feature carries the identity of its inputs — the provenance chain stays closed.
    """
    full = SNAPSHOT / path
    if not full.exists():
        raise FileNotFoundError(f"Cannot register {artifact_id!r}: {full} does not exist")
    rec = {
        "id": artifact_id,
        "path": path,
        "source_path": None,
        "source_sha256": None,
        "sha256": sha256(full),
        "bytes": full.stat().st_size,
        "kind": kind,
        "space": space,
        "used_by": used_by,
        "provenance": provenance,
        "derived_from": derived_from or {},
        "copied_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **describe(full, kind),
    }
    doc = read_doc()
    doc["artifacts"] = [a for a in doc["artifacts"] if a["id"] != artifact_id] + [rec]
    write_doc(doc)
    return rec


def render_markdown(doc: dict) -> str:
    """Render ``MANIFEST.md`` from the manifest document."""
    dirty = " (dirty)" if doc["git"].get("dirty") else ""
    lines = [
        "# `reproduction/experiments/data/` — input snapshot manifest",
        "",
        f"> **Version {doc['version']}** · built {doc['built_at']} · "
        f"git `{doc['git']['commit'][:12]}`{dirty}",
        f"> Builder: `{doc['builder']}` · Plan: `{doc['plan']}`",
        "",
        "Generated file — **do not edit by hand**; rebuild with `make finalexp-data`.",
        "",
        "Every F1–F7 input lives here and is reached only through",
        "`clip_cues_research.finalexp.data.get_*`, which verifies the sha256 on load and asserts the",
        "declared embedding `space`. See [EXCLUDED.md](EXCLUDED.md) for what was deliberately left",
        "out and why.",
        "",
        "**Spaces:** `pooler_1024` = frozen CLIP ViT-L/14-336 `pooler_output` · "
        "`crossmodal_768_canon` = shared image–text space, canonical (post-2026-07-17 fix) · "
        "`n/a` = not an embedding.",
        "",
        "## Artifacts",
        "",
        "| id | path | kind | space | shape | sha256 (short) | MB | used by |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in sorted(doc["artifacts"], key=lambda r: r["id"]):
        shape = "x".join(str(s) for s in a["shape"]) if a.get("shape") else "—"
        lines.append(
            f"| `{a['id']}` | `{a['path']}` | {a['kind']} | `{a['space']}` | {shape} | "
            f"`{a['sha256'][:16]}` | {a['bytes'] / 1e6:.2f} | {', '.join(a['used_by'])} |"
        )
    lines += ["", "## Provenance", ""]
    for a in sorted(doc["artifacts"], key=lambda r: r["id"]):
        lines += [
            f"### `{a['id']}`",
            "",
            f"- **File:** `{a['path']}` ({a['bytes'] / 1e6:.2f} MB)",
            f"- **sha256:** `{a['sha256']}`",
            f"- **Space:** `{a['space']}`",
        ]
        if a.get("source_path"):
            lines.append(f"- **Copied from:** `{a['source_path']}`")
        if a.get("derived_from"):
            inputs = ", ".join(f"`{k}` (`{v[:12]}`)" for k, v in a["derived_from"].items())
            lines.append(f"- **Derived from:** {inputs}")
        lines.append(f"- **Origin:** {a['provenance']}")
        if a.get("split_counts"):
            counts = ", ".join(f"{k} {v}" for k, v in sorted(a["split_counts"].items()))
            lines.append(f"- **Splits:** {counts}")
        if a.get("n_vocab"):
            lines.append(f"- **Vocabulary size:** {a['n_vocab']}")
        if a.get("content"):
            lines.append(f"- **Content:** `{json.dumps(a['content'])}`")
        lines.append("")
    return "\n".join(lines) + "\n"
