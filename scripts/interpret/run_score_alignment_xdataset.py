#!/usr/bin/env python
"""E12/S3: does the score-space cue profile travel across datasets?

Follow-up to run_score_alignment.py. Two questions the in-domain run cannot answer:

  Q1 "same detector, different data" — score the SynthCLIC-trained deployed detector on CNNSpot and
     SynthBuster+ images. Does its cue profile survive the distribution shift?
  Q2 "different detector, own data"  — score each dataset's OWN deployed detector on its own images.
     Do detectors trained on other corpora key on the same named cues?

The specific hypothesis under test: the real-side **capture-provenance markers** found in-domain
(watermark, instant_camera_cues, documentary_look, provenance_scan, stock_photo_look) are
SynthCLIC **corpus shortcuts** rather than properties of authenticity. If so they should rank high
on SynthCLIC and drop elsewhere.

Split discipline: SynthCLIC uses **test** (exploratory, matches the E12 headline); CNNSpot and
SynthBuster+ use **validation**. SynthBuster+ **test is never touched** (frozen protocol, one read
executed) — asserted in code.

Cue scores use derived projected features (pooler @ Wp^T, the both-sides-derived rule); E12's S4
showed cached vs derived cue profiles agree at rho 0.995.

Outputs: outputs/interpretation/score_alignment/{xdataset_profile.csv, xdataset_summary.json}.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from clip_cues_research.vocab_opt.data import ensure_dir, load_frame, load_vocab

OUT = ensure_dir("outputs/interpretation/score_alignment")
WP = "data/embeddings/clip_l14_336_visual_projection.npy"
CKPT = "data/checkpoints"
N_BOOT = 1000

# dataset -> (pooler pkl, split, own deployed checkpoint stem)
DATASETS = {
    "synthclic": ("data/embeddings/synthclic_clip_large_patch14.pkl", "test", "synthclic"),
    "cnnspot": ("data/embeddings/cnnspot_clip_large_patch14.pkl", "validation", "cnnspot"),
    "synthbuster": (
        "data/embeddings/synthbuster-plus_clip_large_patch14.pkl",
        "validation",
        "synthbuster",
    ),
}
VOCABS = {
    "antonyms": "data/embeddings/vocab_canon/antonyms.pt",
    "optimized_v2_canon": "data/embeddings/vocab_canon/optimized_v2_canon.pt",
}
PROVENANCE = [
    "watermark",
    "instant_camera_cues",
    "documentary_look",
    "provenance_scan",
    "stock_photo_look",
    "provenance_press",
]


def linear_probe(stem: str) -> tuple[np.ndarray, float]:
    s = torch.load(f"{CKPT}/linear_probe_{stem}.ckpt", map_location="cpu", weights_only=False)
    s = s.get("state_dict", s)
    w = s["model.classification_head.fc.weight"].numpy().astype(np.float64).ravel()
    b = float(s["model.classification_head.fc.bias"].numpy().ravel()[0])
    return w, b


def col_corr(z: np.ndarray, C: np.ndarray) -> np.ndarray:
    zc = z - z.mean()
    Cc = C - C.mean(0)
    den = np.linalg.norm(zc) * np.linalg.norm(Cc, axis=0)
    return (zc @ Cc) / np.clip(den, 1e-12, None)


def boot_p(z: np.ndarray, C: np.ndarray, ids: np.ndarray, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    uids = np.unique(ids)
    idx_of = {u: np.where(ids == u)[0] for u in uids}
    draws = np.empty((n, C.shape[1]))
    for i in range(n):
        rows = np.concatenate([idx_of[u] for u in rng.choice(uids, len(uids), replace=True)])
        draws[i] = col_corr(z[rows], C[rows])
    frac_neg = (draws <= 0).mean(0)
    p = 2 * np.minimum(frac_neg, 1 - frac_neg)
    return (
        np.percentile(draws, 2.5, axis=0),
        np.percentile(draws, 97.5, axis=0),
        np.clip(p, 1.0 / n, 1.0),
    )


def bh_fdr(p: np.ndarray) -> np.ndarray:
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    q[order] = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def main() -> None:
    Wp = np.load(WP)
    sc_w, sc_b = linear_probe("synthclic")
    summary: dict = {
        "note": "E12/S3 cross-dataset score-space cue profiles; SB+ TEST never read",
        "splits": {k: v[1] for k, v in DATASETS.items()},
    }
    rows = []

    for ds, (pkl, split, own) in DATASETS.items():
        if ds == "synthbuster":
            assert split != "test", "SB+ test is frozen — one read already executed"
        frame = load_frame(pkl)
        P, y, df = frame.split(split)
        ids = df["image_id"].to_numpy()
        Q = P @ Wp.T  # derived projected features
        own_w, own_b = linear_probe(own)

        scorers = {"sc_trained": (sc_w, sc_b)}
        if own != "synthclic":
            scorers["own_trained"] = (own_w, own_b)

        for sname, (w, b) in scorers.items():
            z = P @ w + b
            auroc = float(roc_auc_score(y, z))
            summary[f"{ds}/{sname}/auroc"] = auroc
            summary[f"{ds}/{sname}/n"] = int(len(y))
            summary[f"{ds}/{sname}/n_clusters"] = int(len(np.unique(ids)))
            for vname, vpath in VOCABS.items():
                T, names = load_vocab(vpath)
                C = Q @ T.T
                r_all = col_corr(z, C)
                r_real = col_corr(z[y == 0], C[y == 0])
                r_syn = col_corr(z[y == 1], C[y == 1])
                lo, hi, p = boot_p(z, C, ids)
                rows.append(
                    pd.DataFrame(
                        {
                            "dataset": ds,
                            "split": split,
                            "scorer": sname,
                            "vocab": vname,
                            "cue_idx": np.arange(len(names)),
                            "cue": names,
                            "r_pooled": r_all,
                            "r_macro_within": 0.5 * (r_real + r_syn),
                            "ci_lo": lo,
                            "ci_hi": hi,
                            "q_fdr": bh_fdr(p),
                        }
                    )
                )
                # random-direction null on this dataset
                rng = np.random.default_rng(0)
                R = rng.normal(size=T.shape)
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                nn = np.abs(col_corr(z, Q @ R.T))
                summary[f"{ds}/{sname}/{vname}/null_p95"] = float(np.percentile(nn, 95))

    prof = pd.concat(rows, ignore_index=True)
    prof.to_csv(OUT / "xdataset_profile.csv", index=False)

    # ---- cross-condition profile agreement (per vocab) ----
    for vname in VOCABS:
        sub = prof[prof.vocab == vname].copy()
        sub["cond"] = sub.dataset + ":" + sub.scorer
        for field in ("r_pooled", "r_macro_within"):
            piv = sub.pivot(index="cue_idx", columns="cond", values=field)
            ks = list(piv.columns)
            summary[f"S3_profile_spearman/{vname}/{field}"] = {
                f"{a}~{b}": round(float(spearmanr(piv[a], piv[b]).statistic), 3)
                for i, a in enumerate(ks)
                for b in ks[i + 1 :]
            }

    # ---- the shortcut test: provenance-marker r + rank per condition ----
    track: dict = {}
    for (ds, sname, vname), g in prof.groupby(["dataset", "scorer", "vocab"], sort=False):
        g = g.copy()
        g["rank_abs"] = g.r_macro_within.abs().rank(ascending=False).astype(int)
        n95 = summary[f"{ds}/{sname}/{vname}/null_p95"]
        for cue in PROVENANCE:
            hit = g[g.cue == cue]
            if len(hit):
                r = hit.iloc[0]
                track.setdefault(cue, {})[f"{ds}:{sname}:{vname}"] = {
                    "r_within": round(float(r.r_macro_within), 3),
                    "r_pooled": round(float(r.r_pooled), 3),
                    "rank_of": f"{int(r.rank_abs)}/{len(g)}",
                    "above_null": bool(abs(r.r_pooled) > n95),
                    "q": round(float(r.q_fdr), 4),
                }
    summary["S3_provenance_marker_tracking"] = track

    # ---- top cues per condition (published vocabulary) ----
    for (ds, sname), g in prof[prof.vocab == "antonyms"].groupby(["dataset", "scorer"], sort=False):
        top = g.reindex(g.r_macro_within.abs().sort_values(ascending=False).index).head(10)
        summary[f"top10/{ds}:{sname}"] = [
            f"{r.cue} {r.r_macro_within:+.3f}" for r in top.itertuples()
        ]

    (OUT / "xdataset_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float)[:3000])
    print(f"\nwrote {OUT}/xdataset_*")


if __name__ == "__main__":
    main()
