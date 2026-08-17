#!/usr/bin/env python
"""T0/E7: apply CLIP heads to cached CF-Eval embeddings -> E7-protocol predictions parquet.

Computes the canonical *linear-probe* (k=1/logistic) CommunityForensics-Eval number with the SAME
pipeline as E7: it writes a predictions parquet that ``scripts/export/export_community_eval_tables.py``
consumes (identical per-source==per-generator mAP pairing). It first VALIDATES the embedding path by
reproducing the persisted E7 orthogonal-head scores per-image from the same embeddings — if that
matches, the linear-probe number is produced by a provably identical protocol and is strictly
comparable to the existing CF-Eval rows.

Embeddings come from ``scripts/extract/extract_cf_eval_embeddings.py`` (run on the GPU box, synced to data/).

Run (local, CPU):
    python scripts/utils/eval_heads_on_cf_embeddings.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clip_cues.model import load_clip_classifier
from clip_cues_research.results import make_run_id

EMB = Path("data/embeddings/communityforensics_l14_eval.pkl")
PRED = Path("results/e7_community_eval/predictions")
OUT_COLS = [
    "image_id",
    "label",
    "source",
    "architecture",
    "real_source",
    "subset",
    "score",
    "detector",
]
HEADS = {
    "linear_probe_synthclic": "data/checkpoints/linear_probe_synthclic.ckpt",
    "clip_orthogonal_synthclic": "data/checkpoints/clip_orthogonal_synthclic.ckpt",
}
# Canonical CLIP linear-probe detectors trained on each dataset — scored on CF-Eval for the publication
# detector table (all operate on the same cached 1024-d pooler embeddings).
PROBES = {
    "linear_probe_synthclic": "data/checkpoints/linear_probe_synthclic.ckpt",
    "linear_probe_synthbuster": "data/checkpoints/linear_probe_synthbuster.ckpt",
    "linear_probe_cnnspot": "data/checkpoints/linear_probe_cnnspot.ckpt",
    "linear_probe_combined": "data/checkpoints/linear_probe_combined.ckpt",
}


def head_scores(ckpt: str, emb: np.ndarray) -> np.ndarray:
    """P(fake) for cached pooler embeddings: sigmoid(classification_head(emb)['logits']).

    Reuses ``load_clip_classifier`` (builds the exact head; backbone loaded but unused) so the head
    construction matches ``predict_batch`` byte-for-byte."""
    model = load_clip_classifier(ckpt, cache_dir="data/hf_cache", device="cpu").eval()
    with torch.no_grad():
        logits = model.classification_head(torch.from_numpy(emb).float())["logits"].reshape(-1)
        return torch.sigmoid(logits).numpy()


def validate_ortho(meta: pd.DataFrame, ortho_emb: np.ndarray) -> dict:
    """Reproduce the persisted E7 ortho parquet scores from the embeddings (per-image).

    Alignment is **positional**, not by ``image_id`` (image_id is NOT unique in CF-Eval — reals are
    paired into each generator's group, so a join would explode). Both this extraction and E7's
    ``score_cf_split`` iterate the same ``CompEval`` split in order, so row i corresponds; we assert the
    metadata columns line up positionally before trusting the score comparison."""
    cands = sorted(PRED.glob("clip_orthogonal_synthclic__*.parquet"))
    if not cands:
        return {"validated": False, "reason": "no existing ortho parquet to validate against"}
    ref = pd.read_parquet(cands[-1]).reset_index(drop=True)
    if len(ref) != len(meta):
        return {"validated": False, "reason": f"length mismatch emb={len(meta)} ref={len(ref)}"}
    meta_aligned = all(
        bool((meta[c].astype(str).to_numpy() == ref[c].astype(str).to_numpy()).all())
        for c in ["image_id", "label", "source", "real_source"]
        if c in ref.columns
    )
    diff = np.abs(ortho_emb - ref["score"].to_numpy())
    return {
        "validated": bool(meta_aligned and diff.max() < 1e-3),
        "meta_positionally_aligned": meta_aligned,
        "n": int(len(meta)),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "corr": float(np.corrcoef(ortho_emb, ref["score"].to_numpy())[0, 1]),
    }


def main() -> None:
    d = pickle.load(open(EMB, "rb"))
    meta = d["df"].reset_index(drop=True)
    emb = np.asarray(d["embeddings"], dtype=np.float32)
    assert len(meta) == len(emb)
    print(f"loaded {len(meta)} CF-Eval embeddings; labels={meta.label.value_counts().to_dict()}")

    # 1) validation: ortho head on embeddings must reproduce the persisted E7 ortho scores
    ortho = head_scores(HEADS["clip_orthogonal_synthclic"], emb)
    val = validate_ortho(meta, ortho)
    print("ortho-path validation:", val)
    if not val.get("validated"):
        print(
            "WARNING: embedding path did NOT reproduce E7 ortho scores — investigate before trusting "
            "the linear-probe number (preprocessing/ordering mismatch)."
        )

    # 2) one predictions parquet per CLIP linear probe (consumed by export_community_eval_tables.py)
    PRED.mkdir(parents=True, exist_ok=True)
    rid = make_run_id()
    for det, ckpt in PROBES.items():
        if not Path(ckpt).exists():
            print(f"  skip {det}: checkpoint missing ({ckpt})")
            continue
        lp = head_scores(ckpt, emb)
        df = meta.copy()
        df["score"] = lp
        df["detector"] = det
        df = df[OUT_COLS]
        out = PRED / f"{det}__{rid}.parquet"
        df.to_parquet(out, index=False)
        print(
            f"wrote {out}  (P(fake) mean fake={lp[meta.label == 1].mean():.3f} "
            f"real={lp[meta.label == 0].mean():.3f})"
        )
    print("next: uv run python scripts/export/export_community_eval_tables.py")


if __name__ == "__main__":
    main()
