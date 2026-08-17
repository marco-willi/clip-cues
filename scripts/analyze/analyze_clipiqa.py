#!/usr/bin/env python
"""E8: how CLIP-IQA fits. CLIP-IQA scores perceptual attributes via CLIP antonym PROMPTS (semantic),
not pixels. This tests: (A) do real/synth shift on each CLIP-IQA axis (reproduce paper)? (B) does the
detector direction align with CLIP-IQA axes? (C) does CLIP-IQA 'sharpness'/'noisiness' equal the PIXEL
sharpness/noise (Laplacian/residual)? (D) how much detection do the ~8 CLIP-IQA axes alone recover?

Run: python scripts/analyze/analyze_clipiqa.py --dataset synthclic
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from analyze_residual_directions import attributes  # reuse pixel attributes()
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoProcessor, CLIPModel

POOLER = {
    "synthclic": "data/embeddings/synthclic_clip_large_patch14.pkl",
    "cnnspot": "data/embeddings/cnnspot_clip_large_patch14.pkl",
}
PROJ = {
    "synthclic": "data/embeddings/synthclic_projected_embeddings.pkl",
    "cnnspot": "data/embeddings/cnnspot_projected_embeddings.pkl",
}
HF = {"synthclic": "marco-willi/synthclic", "cnnspot": "marco-willi/cnnspot-small"}
OUT = Path("outputs/e8/clipiqa")
OUT.mkdir(parents=True, exist_ok=True)
# standard CLIP-IQA antonym prompt pairs (Wang et al. 2022) — positive pole first
PAIRS = [
    ("Good photo.", "Bad photo.", "quality"),
    ("Sharp photo.", "Blurry photo.", "sharpness"),
    ("Noisy photo.", "Clean photo.", "noisiness"),
    ("Colorful photo.", "Dull photo.", "colorfulness"),
    ("Bright photo.", "Dark photo.", "brightness"),
    ("High contrast photo.", "Low contrast photo.", "contrast"),
    ("Natural photo.", "Synthetic photo.", "naturalness"),
    ("A real photo.", "A computer-generated image.", "realness"),
]


def unit(v):
    return v / np.clip(np.linalg.norm(v), 1e-12, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(POOLER))
    a = ap.parse_args()
    ds = a.dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clip = (
        CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", cache_dir="data/hf_cache")
        .to(dev)
        .eval()
    )
    proc = AutoProcessor.from_pretrained(
        "openai/clip-vit-large-patch14-336", cache_dir="data/hf_cache"
    )

    def emb_text(t):
        tok = proc(text=t, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            f = clip.get_text_features(**tok)
            if not torch.is_tensor(f):
                f = clip.text_projection(f.pooler_output)
        return f.cpu().numpy().astype(np.float64)

    pos = emb_text([p for p, _, _ in PAIRS])
    neg = emb_text([n for _, n, _ in PAIRS])
    iqa_dirs = unit(
        (pos / np.linalg.norm(pos, axis=1, keepdims=True))
        - (neg / np.linalg.norm(neg, axis=1, keepdims=True))
    )  # (8,768) diff dirs
    iqa_dirs = iqa_dirs / np.linalg.norm(iqa_dirs, axis=1, keepdims=True)
    axes = [n for *_, n in PAIRS]

    # projected (768) eval embeddings -> CLIP-IQA scores; pooler (1024) -> detector direction
    qd = pickle.load(open(PROJ[ds], "rb"))
    qdf = qd["df"]
    qemb = qd["embeddings"].astype(np.float64)
    pd_ = pickle.load(open(POOLER[ds], "rb"))
    pdf = pd_["df"]
    pemb = pd_["embeddings"].astype(np.float64)
    ev = "test" if (qdf["split"] == "test").any() else "validation"
    qm = (qdf["split"] == ev).to_numpy()
    pm_tr = (pdf["split"] == "train").to_numpy()
    pm_ev = (pdf["split"] == ev).to_numpy()
    y = qdf.loc[qm, "label"].to_numpy().astype(int)
    IQA = (qemb[qm] / np.linalg.norm(qemb[qm], axis=1, keepdims=True)) @ iqa_dirs.T  # (Nev,8)
    # detector direction (deterministic logreg on pooler train)
    sc = StandardScaler().fit(pemb[pm_tr])
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(
        sc.transform(pemb[pm_tr]), pdf.loc[pm_tr, "label"].to_numpy()
    )
    det = pemb[pm_ev] @ unit(lr.coef_.ravel() / sc.scale_)  # detector activation on eval

    # pixel attributes on eval images
    from datasets import load_dataset

    dsimg = load_dataset(HF[ds])[ev]
    pix = pd.DataFrame([attributes(dsimg[i]["image"]) for i in range(int(qm.sum()))])

    rows = []
    for j, name in enumerate(axes):
        rows.append(
            {
                "clipiqa_axis": name,
                "real_synth_AUROC": roc_auc_score(y, IQA[:, j]),
                "corr_with_detector": float(spearmanr(IQA[:, j], det).statistic),
            }
        )
    tab = pd.DataFrame(rows)
    tab["abs_AUROC_dev"] = (tab.real_synth_AUROC - 0.5).abs()
    tab = tab.sort_values("abs_AUROC_dev", ascending=False)
    tab.to_csv(OUT / f"{ds}_clipiqa.csv", index=False)
    # (C) semantic vs pixel: CLIP-IQA sharpness vs pixel sharpness; CLIP-IQA noisiness vs pixel residual noise
    sem_vs_pix = {
        "sharpness": float(spearmanr(IQA[:, axes.index("sharpness")], pix["sharpness"]).statistic),
        "noisiness": float(
            spearmanr(IQA[:, axes.index("noisiness")], pix["residual_noise_std"]).statistic
        ),
        "colorfulness": float(
            spearmanr(IQA[:, axes.index("colorfulness")], pix["colorfulness"]).statistic
        ),
        "contrast": float(spearmanr(IQA[:, axes.index("contrast")], pix["contrast"]).statistic),
    }
    # (D) detection from the 8 CLIP-IQA axes alone (train/eval split inside eval would leak; fit on train projected)
    qtr = (qdf["split"] == "train").to_numpy()
    IQAtr = (qemb[qtr] / np.linalg.norm(qemb[qtr], axis=1, keepdims=True)) @ iqa_dirs.T
    lr8 = LogisticRegression(C=1.0, max_iter=5000).fit(
        StandardScaler().fit_transform(IQAtr), qdf.loc[qtr, "label"].to_numpy()
    )
    iqa_auroc = roc_auc_score(y, lr8.decision_function(StandardScaler().fit(IQAtr).transform(IQA)))
    print(f"### {ds} (eval={ev}) ###")
    print(tab.round(3).to_string(index=False))
    print(
        "\n(C) CLIP-IQA-semantic vs PIXEL Spearman: "
        + ", ".join(f"{k}={v:+.2f}" for k, v in sem_vs_pix.items())
    )
    print(
        f"(D) detection AUROC from the 8 CLIP-IQA axes alone: {iqa_auroc:.3f}  (full detector ~0.888 sc / 0.96 cnn)"
    )
    pd.Series(sem_vs_pix).to_csv(OUT / f"{ds}_semantic_vs_pixel.csv")
    (OUT / f"{ds}_iqa_detection.txt").write_text(f"clipiqa_8axis_auroc={iqa_auroc:.4f}\n")


if __name__ == "__main__":
    main()
