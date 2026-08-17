#!/usr/bin/env python
"""E8 follow-up: is CLIP's real-vs-synthetic signal a few NAMED photographic cues, or diffuse?

Replaces the seed-dependent learned ortho axes with analyses that are reproducible by construction:

  A. DETERMINISTIC discriminative direction — logistic regression (and LDA) on the frozen 1024-d CLIP
     pooler embeddings. Convex ⇒ a UNIQUE, seed-independent real-vs-synthetic direction (no init problem).
     Report its detection AUROC and project it (CLIP visual_projection) onto the antonym cues for its own
     top cues.
  B. CUE-BASIS PREDICTIVE GROUNDING (fixed named basis, not learned axes), on 768-d projected embeddings:
     - sufficiency: L2-logreg using ONLY the 168 cue-similarity features vs the full 1024-d model. If the
       cue-basis AUROC ~ full AUROC, named cues SUFFICE (nameable); if << , the signal is DIFFUSE.
     - per-cue single-feature AUROC (which named cues alone separate real/synthetic).
     - L1 selection stability across 25 bootstraps (cues selected consistently = the stable predictive core).

Analysis-only, CPU, minutes. Persists to outputs/e8/cue_grounding/. Conclusions in [[e8-interpretability-stability]].
Run: python scripts/analyze/analyze_cue_grounding.py --dataset synthclic   (and cnnspot)
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import CLIPVisionModelWithProjection

POOLER = {
    "synthclic": "data/embeddings/synthclic_clip_large_patch14.pkl",
    "cnnspot": "data/embeddings/cnnspot_clip_large_patch14.pkl",
}
PROJ = {
    "synthclic": "data/embeddings/synthclic_projected_embeddings.pkl",
    "cnnspot": "data/embeddings/cnnspot_projected_embeddings.pkl",
}


def load(pkl):
    d = pickle.load(open(pkl, "rb"))
    return d["df"].reset_index(drop=True), d["embeddings"].astype(np.float64)


def split_xy(df, emb, split):
    m = (df["split"] == split).to_numpy()
    return emb[m], df.loc[m, "label"].to_numpy().astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(POOLER))
    ap.add_argument("--eval-split", default=None, help="default: test if present else validation")
    ap.add_argument(
        "--vocab",
        default="antonyms",
        choices=["antonyms", "antonyms_poles", "textspan"],
        help="named-cue basis: 168 antonym DIFF-directions (photographic, common-mode removed), "
        "336 raw antonym POLES (control: keeps common-mode), or 3497 TextSpan (GENERAL semantic, not photographic)",
    )
    a = ap.parse_args()
    ds = a.dataset
    out = Path("outputs/e8/cue_grounding")
    out.mkdir(parents=True, exist_ok=True)
    res = {"dataset": ds}

    # ---- data ----
    pdf, pemb = load(POOLER[ds])  # 1024-d pooler (detector space)
    qdf, qemb = load(PROJ[ds])  # 768-d projected (cue space)
    VOCAB = {
        "antonyms": "data/embeddings/antonyms_diff_embeddings.pt",
        "antonyms_poles": "data/embeddings/antonyms_embeddings.pt",
        "textspan": "data/embeddings/textspan_embeddings.pt",
    }
    av = torch.load(VOCAB[a.vocab])
    res["vocab"] = a.vocab
    A = np.asarray(av["embeddings"], dtype=np.float64)
    names = list(av["vocabulary"])
    A = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    # eval split must exist in BOTH pooler and projected frames (cnnspot projected has no test split)
    has_test = (pdf["split"] == "test").any() and (qdf["split"] == "test").any()
    ev = a.eval_split or ("test" if has_test else "validation")
    res["eval_split"] = ev
    chance = 1 / np.sqrt(768)

    # ---- A: deterministic direction ----
    Xtr, ytr = split_xy(pdf, pemb, "train")
    Xev, yev = split_xy(pdf, pemb, ev)
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(Xtr), ytr)
    lda = LinearDiscriminantAnalysis().fit(Xtr, ytr)
    auc_full = roc_auc_score(yev, lr.decision_function(sc.transform(Xev)))
    # determinism: refit, identical coef
    lr2 = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(Xtr), ytr)
    res["A_full_logreg_auroc"] = float(auc_full)
    res["A_lda_auroc"] = float(roc_auc_score(yev, lda.decision_function(Xev)))
    res["A_logreg_deterministic"] = bool(np.allclose(lr.coef_, lr2.coef_))
    # interpret canonical direction (un-standardize coef back to embedding space): w_emb = coef/scale
    w = lr.coef_.ravel() / sc.scale_
    w = w / np.linalg.norm(w)
    Wp = (
        CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14-336", cache_dir="data/hf_cache"
        )
        .visual_projection.weight.detach()
        .numpy()
        .astype(np.float64)
    )
    p = w @ Wp.T
    p /= np.linalg.norm(p)
    cos = A @ p
    top = np.argsort(-np.abs(cos))[:15]
    tag = f"{ds}_{a.vocab}"
    pd.DataFrame(
        {"cue": [names[i] for i in top], "signed_cos": [float(cos[i]) for i in top]}
    ).to_csv(out / f"{tag}_canonical_direction_cues.csv", index=False)
    res["A_canonical_top_cue"] = names[int(top[0])]
    res["A_canonical_top_cos"] = float(cos[top[0]])
    res["chance_cos"] = float(chance)
    res["n_cues"] = len(names)

    # ---- B: cue-basis predictive grounding (768-d projected) ----
    Str, bytr = split_xy(qdf, qemb, "train")
    Sev, byev = split_xy(qdf, qemb, ev)
    Ftr = Str @ A.T
    Fev = Sev @ A.T  # (N, n_cues) cosine-to-cue features
    fs = StandardScaler().fit(Ftr)
    cue_lr = LogisticRegression(C=1.0, max_iter=5000).fit(fs.transform(Ftr), bytr)
    res["B_cuebasis_auroc"] = float(
        roc_auc_score(byev, cue_lr.decision_function(fs.transform(Fev)))
    )
    res["B_full768_auroc"] = float(
        roc_auc_score(
            byev,
            LogisticRegression(C=1.0, max_iter=5000)
            .fit(StandardScaler().fit_transform(Str), bytr)
            .decision_function(StandardScaler().fit(Str).transform(Sev)),
        )
    )
    # per-cue single-feature AUROC
    per = []
    for c in range(len(names)):
        au = roc_auc_score(byev, Fev[:, c])
        per.append((names[c], au, abs(au - 0.5)))
    per = pd.DataFrame(per, columns=["cue", "auroc", "discriminativeness"]).sort_values(
        "discriminativeness", ascending=False
    )
    per.to_csv(out / f"{tag}_per_cue_auroc.csv", index=False)
    res["B_top_singlecue"] = per.head(8)[["cue", "auroc"]].to_dict("records")
    # L1 selection stability over 25 bootstraps (skip for the large TextSpan basis — too slow / overfits)
    if len(names) <= 500:
        rng = np.random.default_rng(0)
        sel = np.zeros(len(names))
        coef_acc = np.zeros(len(names))
        B = 25
        for _ in range(B):
            idx = rng.integers(0, len(Ftr), len(Ftr))
            l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.2, max_iter=5000).fit(
                fs.transform(Ftr)[idx], bytr[idx]
            )
            c = l1.coef_.ravel()
            sel += np.abs(c) > 1e-6
            coef_acc += c
        stab = pd.DataFrame(
            {"cue": names, "select_freq": sel / B, "mean_coef": coef_acc / B}
        ).sort_values("select_freq", ascending=False)
        stab.to_csv(out / f"{tag}_cue_l1_stability.csv", index=False)
        res["B_stable_cues_ge0.8"] = int((stab.select_freq >= 0.8).sum())
        res["B_top_stable"] = stab.head(8)[["cue", "select_freq", "mean_coef"]].to_dict("records")
    else:
        res["B_stable_cues_ge0.8"] = None
        res["B_top_stable"] = None

    (out / f"{tag}_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=2))
    gap = res["A_full_logreg_auroc"] - res["B_cuebasis_auroc"]
    print(
        f"\n[{tag}] full-1024 AUROC={res['A_full_logreg_auroc']:.3f}  cue-basis({len(names)}) AUROC={res['B_cuebasis_auroc']:.3f}  "
        f"gap={gap:+.3f}  -> {'NAMEABLE' if gap < 0.03 else 'DIFFUSE'}"
    )
    if res["B_top_stable"]:
        print(
            "top stable predictive cues:",
            ", ".join(f"{r['cue']}({r['select_freq']:.2f})" for r in res["B_top_stable"]),
        )
    print(
        "top single-cue AUROC:",
        ", ".join(f"{r['cue']}({r['auroc']:.2f})" for r in res["B_top_singlecue"]),
    )
    print(f"wrote outputs/e8/cue_grounding/{tag}_*.csv/json")


if __name__ == "__main__":
    main()
