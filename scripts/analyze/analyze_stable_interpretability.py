#!/usr/bin/env python
"""E8: how to extract STABLE interpretability from CLIP SID. Experiments A, C, D (analysis-only).

Given that learned k=8 ortho axes are seed-unstable, test reproducible alternatives:

  A. DETERMINISTIC DIRECTION stability — bootstrap the training set B times, refit logistic regression
     and LDA on the frozen 1024-d embeddings; report pairwise cosine of the resulting directions and the
     logreg<->LDA agreement. High ⇒ a convex probe gives a stable interpretability anchor (no init).
  C. STABLE CUE PROFILE — bootstrap the deterministic cue-basis model (768-d projected onto the 168 antonym
     diff-cues); report per-cue importance stability (rank correlation across bootstraps) and the consensus
     cue set (top cues by mean |coef| with selection bands). Cross-dataset: are the same cues predictive on
     SynthCLIC vs CNNSpot? Also the single-cue-AUROC ranking stability.
  D. CONSENSUS of the seed ortho directions — sign-align the per-seed most-relevant directions, average,
     and check (i) does the consensus align with the deterministic logreg direction? (ii) is the consensus
     stable across disjoint seed halves? If yes, averaging-over-seeds ≈ the deterministic direction.

Run: python scripts/analyze/analyze_stable_interpretability.py   (writes outputs/e8/stable_interp/)
"""

from __future__ import annotations

import glob
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

POOLER = {
    "synthclic": "data/embeddings/synthclic_clip_large_patch14.pkl",
    "cnnspot": "data/embeddings/cnnspot_clip_large_patch14.pkl",
}
PROJ = {
    "synthclic": "data/embeddings/synthclic_projected_embeddings.pkl",
    "cnnspot": "data/embeddings/cnnspot_projected_embeddings.pkl",
}
OUT = Path("outputs/e8/stable_interp")
OUT.mkdir(parents=True, exist_ok=True)


def load(pkl):
    d = pickle.load(open(pkl, "rb"))
    return d["df"].reset_index(drop=True), d["embeddings"].astype(np.float64)


def xy(df, emb, sp):
    m = (df["split"] == sp).to_numpy()
    return emb[m], df.loc[m, "label"].to_numpy().astype(int)


def unit(v):
    return v / np.clip(np.linalg.norm(v), 1e-12, None)


def logreg_dir(X, y, scaler):
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(scaler.transform(X), y)
    return unit(lr.coef_.ravel() / scaler.scale_)  # back to raw embedding space


def expA(ds, B=30):
    df, emb = load(POOLER[ds])
    Xtr, ytr = xy(df, emb, "train")
    sc = StandardScaler().fit(Xtr)
    rng = np.random.default_rng(0)
    lr_dirs = []
    lda_dirs = []
    for _ in range(B):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        lr_dirs.append(logreg_dir(Xtr[idx], ytr[idx], sc))
        lda = LinearDiscriminantAnalysis().fit(Xtr[idx], ytr[idx])
        lda_dirs.append(unit(lda.coef_.ravel()))

    def mean_cos(D):
        return float(
            np.mean([abs(D[i] @ D[j]) for i, j in itertools.combinations(range(len(D)), 2)])
        )

    lr_full = logreg_dir(Xtr, ytr, sc)
    return {
        "dataset": ds,
        "logreg_bootstrap_cos": mean_cos(lr_dirs),
        "lda_bootstrap_cos": mean_cos(lda_dirs),
        "logreg_vs_lda_cos": float(
            abs(lr_full @ unit(LinearDiscriminantAnalysis().fit(Xtr, ytr).coef_.ravel()))
        ),
        "n_bootstrap": B,
    }, lr_full


def expC(ds, A, names, B=30):
    df, emb = load(PROJ[ds])
    Xtr, ytr = xy(df, emb, "train")
    F = Xtr @ A.T
    sc = StandardScaler().fit(F)
    rng = np.random.default_rng(0)
    coefs = []
    aucs = np.zeros(len(names))
    ev = "test" if (df["split"] == "test").any() else "validation"
    Xe, ye = xy(df, emb, ev)
    Fe = Xe @ A.T
    for c in range(len(names)):
        aucs[c] = roc_auc_score(ye, Fe[:, c])
    for _ in range(B):
        idx = rng.integers(0, len(F), len(F))
        lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(F)[idx], ytr[idx])
        coefs.append(lr.coef_.ravel())
    coefs = np.array(coefs)
    rho = np.mean(
        [
            spearmanr(np.abs(coefs[i]), np.abs(coefs[j])).statistic
            for i, j in itertools.combinations(range(B), 2)
        ]
    )
    mean_imp = np.abs(coefs).mean(0)
    order = np.argsort(-mean_imp)
    prof = pd.DataFrame(
        {
            "cue": [names[i] for i in order],
            "mean_abs_coef": mean_imp[order],
            "coef_cv": (coefs.std(0) / np.clip(np.abs(coefs.mean(0)), 1e-9, None))[order],
            "single_cue_auroc": aucs[order],
        }
    )
    prof.to_csv(OUT / f"{ds}_cue_profile.csv", index=False)
    return {
        "dataset": ds,
        "cue_importance_rank_corr_across_bootstraps": float(rho),
        "top_cues": [names[i] for i in order[:10]],
    }, mean_imp


def expD(ds, det_dir):
    runs = [
        s.rsplit("/", 1)[0]
        for s in glob.glob("results/e8_interpretability_stability/ortho/*/stability.json")
        if (json.load(open(s))["dataset"], json.load(open(s))["regime"]) == (ds, "vary-init")
    ]
    if not runs:
        return {"dataset": ds, "note": "no vary-init k=8 run found"}
    fits = sorted(
        glob.glob(runs[0] + "/fit_*.npz"), key=lambda p: int(p.split("fit_")[1].split(".")[0])
    )
    dirs = []
    for f in fits:
        z = np.load(f)
        imp = z["importance"]
        j = int(np.argmax(np.abs(imp)))
        dirs.append(unit(z["W_L1"][j] * np.sign(imp[j])))
    dirs = np.array(dirs)
    # sign-align all to the first, then average
    aligned = np.array([d * np.sign(d @ dirs[0]) for d in dirs])
    consensus = unit(aligned.mean(0))
    # disjoint halves
    c1 = unit(aligned[:5].mean(0))
    c2 = unit(aligned[5:].mean(0))
    return {
        "dataset": ds,
        "consensus_vs_deterministic_cos": float(abs(consensus @ det_dir)),
        "consensus_half1_vs_half2_cos": float(abs(c1 @ c2)),
        "mean_single_seed_vs_deterministic_cos": float(np.mean([abs(d @ det_dir) for d in dirs])),
    }


def main():
    av = torch.load("data/embeddings/antonyms_diff_embeddings.pt")
    A = av["embeddings"].numpy().astype(np.float64)
    names = list(av["vocabulary"])
    A = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    report = {
        "A_deterministic_stability": [],
        "C_stable_cue_profile": [],
        "D_consensus_direction": [],
    }
    detdirs = {}
    cueimp = {}
    for ds in ["synthclic", "cnnspot"]:
        a, detdir = expA(ds)
        report["A_deterministic_stability"].append(a)
        detdirs[ds] = detdir
        c, imp = expC(ds, A, names)
        report["C_stable_cue_profile"].append(c)
        cueimp[ds] = imp
        report["D_consensus_direction"].append(expD(ds, detdir))
    # cross-dataset cue agreement
    report["C_cross_dataset_cue_rank_corr"] = float(
        spearmanr(cueimp["synthclic"], cueimp["cnnspot"]).statistic
    )
    (OUT / "stable_interp_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
