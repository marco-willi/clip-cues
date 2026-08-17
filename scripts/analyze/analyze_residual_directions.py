#!/usr/bin/env python
"""E8: effective dimensionality + naming of the SID discriminative direction.

Addresses three questions:
  (1) k=1 saturates detection, yet the k=8 head shows ~2 substantial logit contributors — how many
      dimensions are REALLY discriminative?  -> DEFLATION: fit the deterministic direction d1, project it
      OUT of the embeddings, refit on the residual (d2), repeat. Report how detection AUROC depletes as
      each direction is removed (residual AUROC) + each direction's top antonym cues. If AUROC collapses to
      ~chance after removing 1-2 dirs, the signal is effectively rank-1/2 (reconciles k=1).
  (2) interpret d1, then the residual d2, d3 via text matching (top antonyms of each, projected to CLIP text space).
  (3) NAME d1/d2 by correlating their activation with MEASURABLE photographic attributes computed on pixels
      (colorfulness, saturation, contrast, brightness, sharpness, warmth) — far more concrete than weak
      single-term cosines.

Run: python scripts/analyze/analyze_residual_directions.py --dataset synthclic [--no-images]
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import CLIPVisionModelWithProjection

POOLER = {
    "synthclic": "data/embeddings/synthclic_clip_large_patch14.pkl",
    "cnnspot": "data/embeddings/cnnspot_clip_large_patch14.pkl",
}
CKPT = {
    "synthclic": "data/checkpoints/clip_orthogonal_synthclic.ckpt",
    "cnnspot": "data/checkpoints/clip_orthogonal_cnnspot.ckpt",
}
HF = {"synthclic": "marco-willi/synthclic", "cnnspot": "marco-willi/cnnspot-small"}
OUT = Path("outputs/e8/residual")
OUT.mkdir(parents=True, exist_ok=True)


def unit(v):
    return v / np.clip(np.linalg.norm(v), 1e-12, None)


def logreg_dir(X, y):
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X), y)
    w = unit(lr.coef_.ravel() / sc.scale_)
    return w, sc, lr


def auroc_of_dir(Xtr, ytr, Xev, yev):
    # AUROC of a 1-D logreg on the projection onto the (residual) logreg direction
    w, sc, lr = logreg_dir(Xtr, ytr)
    return float(roc_auc_score(yev, lr.decision_function(sc.transform(Xev)))), w


def antonym_terms(direction, A, names, k=8):
    Wp = antonym_terms.Wp
    p = unit(direction @ Wp.T)
    cos = A @ p
    return [(names[i], float(cos[i])) for i in np.argsort(-np.abs(cos))[:k]]


def colorfulness(rgb):  # Hasler-Süsstrunk
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    return float(
        np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )


def _hf_energy_ratio(gray):
    """Fraction of FFT power above a mid radius — high = more high-frequency content (forensic signature)."""
    F = np.fft.fftshift(np.abs(np.fft.fft2(gray - gray.mean())) ** 2)
    h, w = gray.shape
    cy, cx = h / 2, w / 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt(((Y - cy) / cy) ** 2 + ((X - cx) / cx) ** 2)
    tot = F.sum()
    return float(F[r > 0.5].sum() / tot) if tot > 0 else 0.0


def attributes(img):
    img = img.convert("RGB")
    a = np.asarray(img, dtype=np.float64)
    gray = a.mean(2)
    hsv = np.asarray(img.convert("HSV"), dtype=np.float64)
    from scipy.ndimage import gaussian_filter, laplace

    noise = gray - gaussian_filter(gray, 2.0)
    return {
        "colorfulness": colorfulness(a),
        "saturation": hsv[..., 1].mean(),
        "brightness": gray.mean(),
        "contrast": gray.std(),
        "sharpness": float(laplace(gray).var()),
        "warmth_R_minus_B": float(a[..., 0].mean() - a[..., 2].mean()),
        "hf_energy_ratio": _hf_energy_ratio(gray),
        "residual_noise_std": float(noise.std()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(POOLER))
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--no-images", action="store_true")
    a = ap.parse_args()
    ds = a.dataset
    d = pickle.load(open(POOLER[ds], "rb"))
    df = d["df"]
    emb = d["embeddings"].astype(np.float64)
    y = df["label"].to_numpy()
    m = (df["split"] == "train").to_numpy()
    ev = "test" if (df["split"] == "test").any() else "validation"
    me = (df["split"] == ev).to_numpy()
    Xtr, ytr, Xev, yev = emb[m], y[m], emb[me], y[me]
    av = torch.load("data/embeddings/antonyms_diff_embeddings.pt")
    A = av["embeddings"].numpy().astype(np.float64)
    names = list(av["vocabulary"])
    A = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    antonym_terms.Wp = (
        CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14-336", cache_dir="data/hf_cache"
        )
        .visual_projection.weight.detach()
        .numpy()
        .astype(np.float64)
    )

    # (0) confirm the k=8 checkpoint really has ~2 substantial logit contributors
    sd = torch.load(CKPT[ds], map_location="cpu", weights_only=False)["state_dict"]
    W = sd["model.classification_head.layers.0.weight"].numpy()
    wl = sd["model.classification_head.to_logits.weight"].numpy().ravel()
    contrib = (emb @ W.T) * wl
    imp = contrib[y == 1].mean(0) - contrib[y == 0].mean(0)
    print(
        f"[{ds}] k=8 checkpoint per-direction |logit contribution| (sorted): "
        + ", ".join(f"d{i}:{abs(imp[i]):.2f}" for i in np.argsort(-np.abs(imp)))
    )

    # (1)+(2) DEFLATION
    Rtr, Rev = Xtr.copy(), Xev.copy()
    dirs = []
    rows = []
    full_auroc, _ = auroc_of_dir(Xtr, ytr, Xev, yev)
    for s in range(a.steps):
        au, w = auroc_of_dir(Rtr, ytr, Rev, yev)  # AUROC achievable from the CURRENT residual
        terms = antonym_terms(w, A, names, 6)
        rows.append(
            {
                "step": s + 1,
                "residual_auroc": au,
                "top_cues": ", ".join(f"{n}{c:+.2f}" for n, c in terms),
            }
        )
        dirs.append(w)
        Rtr = Rtr - np.outer(Rtr @ w, w)
        Rev = Rev - np.outer(Rev @ w, w)  # deflate this direction
    defl = pd.DataFrame(rows)
    defl.to_csv(OUT / f"{ds}_deflation.csv", index=False)
    print(f"[{ds}] full-space AUROC={full_auroc:.3f}")
    print(defl.to_string(index=False))

    # (3) measurable-attribute grounding for d1, d2
    if not a.no_images:
        from datasets import load_dataset

        dsimg = load_dataset(HF[ds])[ev]
        pos = np.flatnonzero(me)
        # d1,d2 activations on eval
        act1 = Xev @ dirs[0]
        act2 = Xev @ dirs[1] if len(dirs) > 1 else None
        attr = [attributes(dsimg[i]["image"]) for i in range(len(pos))]
        adf = pd.DataFrame(attr)
        cors = {"d1": {k: float(spearmanr(adf[k], act1).statistic) for k in adf.columns}}
        if act2 is not None:
            cors["d2"] = {k: float(spearmanr(adf[k], act2).statistic) for k in adf.columns}
        cdf = pd.DataFrame(cors)
        cdf.to_csv(OUT / f"{ds}_attribute_correlations.csv")
        print(f"\n[{ds}] Spearman(direction activation, measurable attribute):")
        print(cdf.round(3).to_string())


if __name__ == "__main__":
    main()
