#!/usr/bin/env python
"""E11 — sparse signed decomposition of detector boundary normals onto canonical cue axes.

Two modes:

  e11a   SC-internal targets: the tuned cross-modal probe P768t (cached projected space) and
         the P1024 pooler probe ridge-distilled into the shared space (reports preserved
         logit-variance R^2). Decomposed with ant168 and v2-128 canonical signed axes via a
         data-weighted lasso path; knee selected on SC val score-R^2 (tol 0.01); bootstrap
         support-selection frequencies at the knee.

  cross  E11b cross-dataset boundaries in DERIVED shared space (pooler @ Wp^T, both-sides-
         derived): P768t-recipe probes (C by 5-fold train-CV) per dataset (synthclic, cnnspot,
         synthbuster), pairwise boundary cosines, and decompositions of each unit normal plus
         the difference directions (cnnspot - synthclic, synthbuster - synthclic; data-weighted
         on the union of the two train sets).

Outputs: outputs/e11_boundary/{path_<mode>.csv, axes_<mode>.csv, summary_<mode>.json}.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from clip_cues_research.vocab_opt.boundary import (
    distill_to_shared,
    fit_probe,
    knee_row,
    lasso_path_decompose,
    raw_normal,
    support_stability,
    unitv,
)
from clip_cues_research.vocab_opt.data import ensure_dir, load_frame, load_vocab

WP = "data/embeddings/clip_l14_336_visual_projection.npy"
POOLER = {
    "synthclic": "data/embeddings/synthclic_clip_large_patch14.pkl",
    "cnnspot": "data/embeddings/cnnspot_clip_large_patch14.pkl",
    "synthbuster": "data/embeddings/synthbuster-plus_clip_large_patch14.pkl",
}
PROJECTED_SC = "data/embeddings/synthclic_projected_embeddings.pkl"
VOCABS = {
    "ant168": "data/embeddings/vocab_canon/antonyms.pt",
    "v2_128": "data/embeddings/vocab_canon/optimized_v2_canon.pt",
}
OUT = ensure_dir("outputs/e11_boundary")
C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def alpha_path(Vtr, w, T, n=25, decades=4):
    """alpha_max (all-zero point) scaled logspace path for the data-weighted lasso."""
    Z, z = Vtr @ T.T, Vtr @ unitv(w)
    zc = z - z.mean()
    a_max = np.abs(Z.T @ zc).max() / len(z)
    return a_max * np.logspace(0, -decades, n)


def cv_pick_C(Xtr, ytr, folds=5, seed=0):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    best = None
    for C in C_GRID:
        aucs = []
        for tr, te in skf.split(Xtr, ytr):
            sc, lr = fit_probe(Xtr[tr], ytr[tr], C)
            s = lr.decision_function(sc.transform(Xtr[te]))
            aucs.append(roc_auc_score(ytr[te], s))
        m = float(np.mean(aucs))
        if best is None or m > best[1]:
            best = (C, m)
    return best


def decompose_target(tag, w, Vtr, Vval, yval, ids_tr, vocabs, path_rows, axis_rows, summary):
    for vname, (T, names) in vocabs.items():
        alphas = alpha_path(Vtr, w, T)
        rows = lasso_path_decompose(Vtr, Vval, yval, w, T, alphas)
        for r in rows:
            path_rows.append(
                {"target": tag, "vocab": vname, **{k: v for k, v in r.items() if k != "coef"}}
            )
        knee = knee_row(rows)
        freq = support_stability(Vtr, ids_tr, w, T, knee["alpha"])
        coef = knee["coef"]
        for j in np.argsort(-np.abs(coef)):
            if coef[j] == 0 and freq[j] == 0:
                continue
            axis_rows.append(
                {
                    "target": tag,
                    "vocab": vname,
                    "axis": names[j],
                    "coef": float(coef[j]),
                    "abs_rank": int((np.abs(coef) > abs(coef[j])).sum()) + 1,
                    "selection_freq": float(freq[j]),
                }
            )
        summary[f"{tag}/{vname}"] = {
            "knee_alpha": knee["alpha"],
            "nnz": knee["nnz"],
            "cos_coverage": knee["cos_coverage"],
            "val_score_r2": knee["val_score_r2"],
            "val_auroc": knee["val_auroc"],
            "residual_norm_frac": knee["residual_norm_frac"],
            "path_max_val_r2": max(r["val_score_r2"] for r in rows),
        }


def run_e11a():
    pooler = load_frame(POOLER["synthclic"])
    projected = load_frame(PROJECTED_SC)
    Ptr, pytr, ptr_df = pooler.split("train")
    Qtr, ytr, qtr_df = projected.split("train")
    Qva, yva, _ = projected.split("validation")
    ids_tr = qtr_df["image_id"].to_numpy()
    Wp = np.load(WP)
    vocabs = {k: load_vocab(v) for k, v in VOCABS.items()}

    path_rows, axis_rows, summary = [], [], {}

    # target 1: P768t (tuned probe, cached projected space)
    w768, _ = raw_normal(*fit_probe(Qtr, ytr, C=0.01))
    decompose_target("P768t", w768, Qtr, Qva, yva, ids_tr, vocabs, path_rows, axis_rows, summary)

    # target 2: P1024 distilled into shared space (data-aware ridge on SC train)
    w1024, _ = raw_normal(*fit_probe(Ptr, pytr, C=1.0))
    u, rep = distill_to_shared(Ptr, w1024, Wp)
    summary["P1024_distillation"] = rep
    # decompose in the derived shared space (u lives there)
    Dtr, Dva = Ptr @ Wp.T, pooler.split("validation")[0] @ Wp.T
    pyva = pooler.split("validation")[1]
    decompose_target(
        "P1024_distilled",
        u,
        Dtr,
        Dva,
        pyva,
        ptr_df["image_id"].to_numpy(),
        vocabs,
        path_rows,
        axis_rows,
        summary,
    )
    # alignment of the two targets (in their common 768 space; different data frames noted)
    summary["cos_P768t_vs_P1024_distilled"] = float(unitv(w768) @ unitv(u))
    return path_rows, axis_rows, summary


def run_cross():
    Wp = np.load(WP)
    vocabs = {k: load_vocab(v) for k, v in VOCABS.items()}
    frames = {k: load_frame(v) for k, v in POOLER.items()}
    tr, va = {}, {}
    for k, f in frames.items():
        X, y, df = f.split("train")
        tr[k] = (X @ Wp.T, y, df["image_id"].to_numpy())
        X, y, df = f.split("validation")
        va[k] = (X @ Wp.T, y)

    path_rows, axis_rows, summary = [], [], {}
    normals = {}
    for k in POOLER:
        X, y, ids = tr[k]
        C, cv_auc = cv_pick_C(X, y)
        w, _ = raw_normal(*fit_probe(X, y, C))
        normals[k] = unitv(w)
        Xv, yv = va[k]
        sc, lr = fit_probe(X, y, C)
        val_auc = float(roc_auc_score(yv, lr.decision_function(sc.transform(Xv))))
        summary[f"probe/{k}"] = {"C_traincv": C, "traincv_auroc": cv_auc, "val_auroc": val_auc}
        decompose_target(k, w, X, Xv, yv, ids, vocabs, path_rows, axis_rows, summary)

    summary["boundary_cosines"] = {
        f"{a}~{b}": float(normals[a] @ normals[b])
        for i, a in enumerate(POOLER)
        for b in list(POOLER)[i + 1 :]
    }

    # difference directions, data-weighted on the union of the two train sets
    for other in ("cnnspot", "synthbuster"):
        dw = normals[other] - normals["synthclic"]
        Vtr = np.vstack([tr[other][0], tr["synthclic"][0]])
        ids = np.concatenate(
            [
                np.char.add(f"{other}:", tr[other][2].astype(str)),
                np.char.add("sc:", tr["synthclic"][2].astype(str)),
            ]
        )
        Vval = np.vstack([va[other][0], va["synthclic"][0]])
        decompose_target(
            f"delta_{other}_minus_synthclic",
            dw,
            Vtr,
            Vval,
            None,
            ids,
            vocabs,
            path_rows,
            axis_rows,
            summary,
        )
    return path_rows, axis_rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("e11a", "cross"), required=True)
    args = ap.parse_args()
    path_rows, axis_rows, summary = run_e11a() if args.mode == "e11a" else run_cross()
    pd.DataFrame(path_rows).to_csv(OUT / f"path_{args.mode}.csv", index=False)
    pd.DataFrame(axis_rows).to_csv(OUT / f"axes_{args.mode}.csv", index=False)
    (OUT / f"summary_{args.mode}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    ax = pd.DataFrame(axis_rows)
    for (t, v), g in ax.groupby(["target", "vocab"]):
        top = g[g.coef != 0].nlargest(8, "coef", keep="all").head(8)
        bot = g[g.coef != 0].nsmallest(8, "coef", keep="all").head(8)
        print(f"\n== {t} / {v} (top +/- by signed coef; freq = bootstrap selection) ==")
        for _, r in pd.concat([top, bot]).iterrows():
            print(f"  {r.coef:+.4f}  freq {r.selection_freq:.2f}  {r.axis}")
    print(f"\nwrote {OUT}/")


if __name__ == "__main__":
    main()
