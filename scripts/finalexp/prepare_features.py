#!/usr/bin/env python
"""Step 2: derive the 768-d projected features and 168/128-d cue scores into the snapshot.

Reads only snapshot ids (`pooler/*`, `projection/wp_l14_336`, `vocab/*`), writes the derived
features back **into** ``reproduction/experiments/data/`` and registers each in the manifest with the sha256 of
every input that fed it — so a run is reproducible from ``reproduction/experiments/data/`` alone and the
provenance chain stays closed.

Also runs the derived-vs-cached cross-check against `reference/projected_cached_*` (which are
marked cross-check-only in the manifest and are never fitted on): per-image cosine plus a
downstream AUROC delta, compared to the <= 0.003 benchmark recorded in
EXTERNAL_VALIDATION_PROTOCOL.md.

    uv run python scripts/finalexp/prepare_features.py
"""

from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from clip_cues_research.finalexp import data as D
from clip_cues_research.finalexp.features import cue_scores, per_row_cosine, project, unit
from clip_cues_research.finalexp.snapshot import register_artifact

DATASETS = ["synthclic", "cnnspot", "synthbuster-plus"]
# Only the antonym vocabulary is published: it is the one the manuscript uses. The E9
# optimized set has no manuscript section, so it is not part of the released snapshot.
VOCABS = ["antonyms"]
CROSSCHECK = {
    "synthclic": "reference/projected_cached_synthclic",
    "cnnspot": "reference/projected_cached_cnnspot",
}


def build_projected(ds: str) -> dict:
    """Derive and register the 768-d projected frame for one dataset."""
    frame = D.get_frame(f"pooler/{ds}", expected_space=D.SPACE_POOLER)
    wp = D.get_array("projection/wp_l14_336")
    emb = project(frame.emb, wp)

    rel = f"embeddings/projected_derived/{ds}.pkl"
    out = D.SNAPSHOT / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"df": frame.df, "embeddings": emb.astype(np.float32)}, f)

    rec = register_artifact(
        artifact_id=f"projected/{ds}",
        path=rel,
        kind="projected_embeddings",
        space=D.SPACE_CANON,
        used_by=["F3", "F4", "F6"],
        provenance=(
            "scripts/finalexp/prepare_features.py — derived as e = Wp h from the cached pooler "
            "frame (both-sides-derived rule, EXTERNAL_VALIDATION_PROTOCOL.md). NOT a separate "
            "extraction: D_h and D_e therefore see the same image representation and differ only "
            "by the projection."
        ),
        derived_from=D.input_shas(f"pooler/{ds}", "projection/wp_l14_336"),
    )
    print(f"  projected/{ds:18s} {tuple(emb.shape)}  -> {rel}  ({rec['bytes'] / 1e6:.1f} MB)")
    return rec


def build_cue_scores(ds: str, vocab: str) -> dict:
    """Derive and register the cue-score features for one (dataset, vocabulary)."""
    frame = D.get_frame(f"projected/{ds}", expected_space=D.SPACE_CANON)
    V, names = D.get_vocab(f"vocab/{vocab}")  # asserts canonical text space
    C = cue_scores(frame.emb, V)

    rel = f"embeddings/cue_scores/{ds}__{vocab}.npz"
    out = D.SNAPSHOT / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, scores=C.astype(np.float32), cues=np.array(names, dtype=object).astype(str)
    )

    rec = register_artifact(
        artifact_id=f"cue_scores/{ds}__{vocab}",
        path=rel,
        kind="cue_scores",
        space=D.SPACE_NA,
        used_by=["F4"] + (["F1", "F2", "F3", "F6", "F7"] if vocab == "antonyms" else ["F6", "F7"]),
        provenance=(
            "scripts/finalexp/prepare_features.py — c_j = <e/||e||, v_j> on the derived projected "
            "frame against the CANONICAL cue basis (never the retracted W-squared vocabularies)."
        ),
        derived_from=D.input_shas(f"projected/{ds}", f"vocab/{vocab}"),
    )
    print(f"  cue_scores/{ds}__{vocab:20s} {tuple(C.shape)} -> {rel}")
    return rec


def probe_auroc(emb: np.ndarray, df, eval_split: str) -> float:
    """Deterministic standardized logistic probe AUROC (train -> eval_split), cross-check only.

    Matches the P768t recipe (standardized features, C=0.01) so the derived-vs-cached delta is
    measured on the same kind of model the interpretation record uses.
    """
    tr = (df["split"] == "train").to_numpy()
    ev = (df["split"] == eval_split).to_numpy()
    y = df["label"].to_numpy().astype(int)
    X = unit(emb)
    sc = StandardScaler().fit(X[tr])
    lr = LogisticRegression(C=0.01, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
    return float(roc_auc_score(y[ev], lr.decision_function(sc.transform(X[ev]))))


def crosscheck(ds: str, ref_id: str) -> dict:
    """Derived vs separately-extracted projected embeddings: cosine + downstream AUROC delta.

    The cached references do not all cover the same splits — `cnnspot_projected_embeddings.pkl` is
    an E8 helper covering train+val only — so the comparison is restricted to the splits the
    reference actually has, and the eval split is the largest held-out one available in both.
    """
    derived = D.get_frame(f"projected/{ds}", expected_space=D.SPACE_CANON)
    cached = D.get_frame(ref_id, expected_space=D.SPACE_CANON)

    shared = sorted(set(cached.df["split"]) & set(derived.df["split"]))
    dsel = derived.df["split"].isin(shared).to_numpy()
    csel = cached.df["split"].isin(shared).to_numpy()
    ddf, cdf = derived.df[dsel].reset_index(drop=True), cached.df[csel].reset_index(drop=True)
    demb, cemb = derived.emb[dsel], cached.emb[csel]

    if len(ddf) != len(cdf) or not (
        (ddf["image_id"].astype(str).values == cdf["image_id"].astype(str).values).all()
        and (ddf["split"].values == cdf["split"].values).all()
    ):
        raise AssertionError(
            f"{ds}: derived/cached row order differs on shared splits {shared} — cannot align by position"
        )

    eval_split = "test" if "test" in shared else "validation"
    cos = per_row_cosine(demb, cemb)
    auroc_d = probe_auroc(demb, ddf, eval_split)
    auroc_c = probe_auroc(cemb, cdf, eval_split)
    delta = abs(auroc_d - auroc_c)
    out = {
        "dataset": ds,
        "reference_id": ref_id,
        "shared_splits": shared,
        "eval_split": eval_split,
        "n": int(len(cos)),
        "cosine_median": round(float(np.median(cos)), 6),
        "cosine_min": round(float(cos.min()), 6),
        "cosine_p01": round(float(np.percentile(cos, 1)), 6),
        "auroc_derived": round(auroc_d, 6),
        "auroc_cached": round(auroc_c, 6),
        "auroc_delta": round(delta, 6),
        "benchmark_auroc_delta": 0.003,
        "cosine_ok": bool(np.median(cos) > 0.99),
        "within_benchmark": bool(delta <= 0.003),
    }
    status = "PASS" if out["cosine_ok"] and out["within_benchmark"] else "NOTE"
    print(
        f"  [{status}] {ds:16s} splits={','.join(shared)} eval={eval_split} n={out['n']}\n"
        f"         median cos {out['cosine_median']:.6f} (min {out['cosine_min']:.4f})  "
        f"AUROC derived {auroc_d:.4f} vs cached {auroc_c:.4f}  delta {delta:.4f}"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--skip-crosscheck", action="store_true")
    args = ap.parse_args()

    print("Deriving 768-d projected features (e = Wp h):")
    for ds in args.datasets:
        build_projected(ds)

    print("\nDeriving cue-score features:")
    for ds in args.datasets:
        for v in VOCABS:
            build_cue_scores(ds, v)

    if not args.skip_crosscheck:
        print("\nDerived-vs-cached cross-check (benchmark: AUROC delta <= 0.003):")
        checks = [crosscheck(ds, rid) for ds, rid in CROSSCHECK.items() if ds in args.datasets]
        out = D.SNAPSHOT / "reference" / "derived_vs_cached_crosscheck.json"
        out.write_text(json.dumps({"checks": checks}, indent=2) + "\n")
        print(f"\n  -> {out}")

    print("\nDone. Re-verify with: uv run python scripts/finalexp/verify_data_snapshot.py")


if __name__ == "__main__":
    main()
