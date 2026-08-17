#!/usr/bin/env python
"""E12: score-space cue alignment against the DEPLOYED detectors (no re-training, no weight mapping).

Every earlier interpretation analysis either (a) mapped a decision direction between the 1024-d
pooler space and the 768-d shared space, or (b) targeted a re-fitted analysis probe (P768t/P1024t).
This experiment does neither. It needs only each detector's *scores*, so it can interpret the
**published checkpoints** directly — including the paper's k=8 orthogonal head (scored through its
final logit, which removes the "which of the 8 directions?" choice) and the nonlinear concept model.

  S1  Per-cue alignment profile: for scorer logit z and cue score c_j = <x, t_j> on the same images,
      report pooled r, WITHIN-CLASS r (real / synthetic / macro — the identifying estimand, since
      pooled r is dominated by class separation), and partial r given the rest of the cue basis.
      Cluster-bootstrap CIs by source photo + BH-FDR; random-direction null for the |r| floor.
  S2  Cross-scorer agreement: Spearman between the per-cue profiles of every scorer. If the deployed
      detector's profile matches the analysis proxies', proxy-based conclusions transfer.
  S4  Metric consistency: for linear scorers, corr(z, c_j) is EXACTLY the data-metric (Sigma) cosine
      between w and Wp^T t_j. Verified numerically; also contrasts the naive RAW cosine ranking.

All CPU, on cached embeddings. SC-test is exploratory (already multi-read); SB+ test is NOT touched.

Outputs: outputs/interpretation/score_alignment/{profile_*.csv, summary.json}.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from clip_cues_research.finalexp.profiles import bh_fdr, boot_ci, col_corr, partial_corr
from clip_cues_research.vocab_opt.data import (
    POOLER_PKL,
    PROJECTED_PKL,
    ensure_dir,
    load_frame,
    load_vocab,
)

OUT = ensure_dir("outputs/interpretation/score_alignment")
WP = "data/embeddings/clip_l14_336_visual_projection.npy"
CKPT = "data/checkpoints"
N_BOOT = 1000
RIDGE = 1e-3  # on the correlation matrix diagonal, for the partial-correlation inverse

VOCABS = {
    "optimized_v2_canon": "data/embeddings/vocab_canon/optimized_v2_canon.pt",
    "antonyms": "data/embeddings/vocab_canon/antonyms.pt",
}


def sd(name: str) -> dict:
    d = torch.load(f"{CKPT}/{name}.ckpt", map_location="cpu", weights_only=False)
    return d.get("state_dict", d)


def deployed_scorers() -> dict:
    """Effective scoring functions of the published SynthCLIC checkpoints.

    The k=8 head has non_linear=False, so it composes to a single effective 1024-d direction
    w_eff = W0^T w_logit — exactly the object score-space analysis needs, obtained without the
    "most relevant direction" heuristic used by the montage/stability work.
    """
    out = {}
    s = sd("linear_probe_synthclic")
    w = s["model.classification_head.fc.weight"].numpy().astype(np.float64).ravel()
    b = float(s["model.classification_head.fc.bias"].numpy().ravel()[0])
    out["deployed_linear_k1"] = {"space": "pooler", "w": w, "b": b, "linear": True}

    s = sd("clip_orthogonal_synthclic")
    W0 = s["model.classification_head.layers.0.weight"].numpy().astype(np.float64)  # (8,1024)
    b0 = s["model.classification_head.layers.0.bias"].numpy().astype(np.float64)
    wl = s["model.classification_head.to_logits.weight"].numpy().astype(np.float64).ravel()
    bl = float(s["model.classification_head.to_logits.bias"].numpy().ravel()[0])
    out["deployed_ortho_k8"] = {
        "space": "pooler",
        "w": W0.T @ wl,
        "b": float(b0 @ wl + bl),
        "linear": True,
    }

    s = sd("cm_antonyms_synthclic")
    out["deployed_concept"] = {
        "space": "projected_nonlinear",
        "T": s["model.text_embeddings"].numpy().astype(np.float64),
        "Wc": s["model.W_concepts.weight"].numpy().astype(np.float64),
        "bc": s["model.W_concepts.bias"].numpy().astype(np.float64),
        "Wk": s["model.W_classifier.weight"].numpy().astype(np.float64).ravel(),
        "bk": float(s["model.W_classifier.bias"].numpy().ravel()[0]),
        "linear": False,
    }
    return out


def score(spec: dict, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Logits of one scorer on a split (P = pooler 1024-d, Q = cached projected 768-d)."""
    if spec["space"] == "pooler":
        return P @ spec["w"] + spec["b"]
    x = Q / np.clip(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12, None)
    gates = 1.0 / (1.0 + np.exp(-(x @ spec["Wc"].T + spec["bc"])))
    return (x @ spec["T"].T * gates) @ spec["Wk"] + spec["bk"]


def main() -> None:
    pooler, projected = load_frame(POOLER_PKL), load_frame(PROJECTED_PKL)
    Wp = np.load(WP)
    Ptr, pytr, _ = pooler.split("train")
    Qtr, ytr, _ = projected.split("train")

    scorers = deployed_scorers()
    # analysis proxies, for the S2 comparison (fit on SC train only, tuned C from review6)
    for nm, (X, y, C_) in {
        "proxy_P1024t": (Ptr, pytr, 0.03),
        "proxy_P768t": (Qtr, ytr, 0.01),
    }.items():
        sc = StandardScaler().fit(X)
        lr = LogisticRegression(C=C_, max_iter=5000).fit(sc.transform(X), y)
        scorers[nm] = {
            "space": "pooler" if nm.endswith("1024t") else "projected_linear",
            "w": lr.coef_.ravel() / sc.scale_,
            "b": float(lr.intercept_[0] - (lr.coef_.ravel() * sc.mean_ / sc.scale_).sum()),
            "linear": True,
        }

    summary: dict = {"note": "E12 score-space alignment; deployed checkpoints + analysis proxies"}
    profiles: dict = {}

    for split in ("validation", "test"):
        P, py, pdf = pooler.split(split)
        Q, y, qdf = projected.split(split)
        ids = qdf["image_id"].to_numpy()
        Qd = P @ Wp.T  # derived projected (for the exact S4 identity)

        # --- scorer logits + sanity gate ---
        Z, aur = {}, {}
        for nm, spec in scorers.items():
            z = P @ spec["w"] + spec["b"] if spec["space"] == "pooler" else None
            if z is None:
                z = Q @ spec["w"] + spec["b"] if spec["linear"] else score(spec, P, Q)
            Z[nm] = z
            aur[nm] = float(roc_auc_score(y, z))
        summary[f"{split}/scorer_auroc"] = aur
        if min(aur.values()) < 0.75:
            raise SystemExit(f"sanity gate: a scorer AUROC looks wrong on {split}: {aur}")

        for vname, vpath in VOCABS.items():
            T, names = load_vocab(vpath)
            Ccue = Q @ T.T  # canonical cue scores (cached projected space)
            rows = []
            for nm in scorers:
                z = Z[nm]
                r_all = col_corr(z, Ccue)
                r_real = col_corr(z[y == 0], Ccue[y == 0])
                r_syn = col_corr(z[y == 1], Ccue[y == 1])
                pr = partial_corr(z, Ccue)
                rec = {
                    "scorer": nm,
                    "cue_idx": np.arange(len(names)),  # names are not unique (v2 has a dup)
                    "cue": names,
                    "r_pooled": r_all,
                    "r_real": r_real,
                    "r_synth": r_syn,
                    "r_macro_within": 0.5 * (r_real + r_syn),
                    "partial_r": pr,
                }
                if split == "test":  # CIs/FDR for BOTH vocabs (antonyms168 = published set)
                    lo, hi, p = boot_ci(z, Ccue, ids)
                    rec |= {"ci_lo": lo, "ci_hi": hi, "p_boot": p, "q_fdr": bh_fdr(p)}
                rows.append(pd.DataFrame(rec))
            df = pd.concat(rows, ignore_index=True)
            df.to_csv(OUT / f"profile_{vname}_{split}.csv", index=False)
            profiles[(vname, split)] = df

            # --- S2: cross-scorer agreement of the per-cue profiles ---
            for field in ("r_pooled", "r_macro_within"):
                piv = df.pivot(index="cue_idx", columns="scorer", values=field)
                ks = list(piv.columns)
                summary[f"{split}/{vname}/S2_profile_spearman/{field}"] = {
                    f"{a}~{b}": float(spearmanr(piv[a], piv[b]).statistic)
                    for i, a in enumerate(ks)
                    for b in ks[i + 1 :]
                }

            # --- null: random unit directions in text space ---
            rng = np.random.default_rng(0)
            R = rng.normal(size=(T.shape[0], T.shape[1]))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            rn = np.abs(col_corr(Z["deployed_linear_k1"], Q @ R.T))
            summary[f"{split}/{vname}/null_random_dirs_absr"] = {
                "p50": float(np.percentile(rn, 50)),
                "p95": float(np.percentile(rn, 95)),
                "max": float(rn.max()),
            }

        # --- S4: metric consistency (exact identity for linear pooler scorers) ---
        if split == "test":
            T, _ = load_vocab(VOCABS["optimized_v2_canon"])
            U = T @ Wp  # (k,1024): pulled-back cue directions Wp^T t_j
            Sig = np.cov(P, rowvar=False)
            checks = {}
            for nm in ("deployed_linear_k1", "deployed_ortho_k8", "proxy_P1024t"):
                w = scorers[nm]["w"]
                sw = Sig @ w
                sig_cos = (U @ sw) / np.sqrt((w @ sw) * np.einsum("ij,jk,ik->i", U, Sig, U))
                emp = col_corr(P @ w, Qd @ T.T)  # derived cue scores => exact identity
                raw_cos = (U @ w) / (np.linalg.norm(U, axis=1) * np.linalg.norm(w))
                checks[nm] = {
                    "max_abs_diff_corr_vs_sigma_cosine": float(np.max(np.abs(emp - sig_cos))),
                    "spearman_rawcos_vs_scorecorr": float(spearmanr(raw_cos, emp).statistic),
                    "max_abs_raw_cos": float(np.abs(raw_cos).max()),
                    "max_abs_score_corr": float(np.abs(emp).max()),
                }
            # cached vs derived cue-score agreement (construction sanity)
            cc = col_corr(Z["deployed_linear_k1"], Q @ T.T)
            checks["cached_vs_derived_cue_profile_spearman"] = float(
                spearmanr(cc, col_corr(Z["deployed_linear_k1"], Qd @ T.T)).statistic
            )
            summary["S4_metric_consistency"] = checks

    # --- headline: top aligned cues for the deployed detector (test), per vocabulary ---
    for vname in VOCABS:
        dd = profiles[(vname, "test")].query("scorer == 'deployed_linear_k1'").copy()
        null95 = summary[f"test/{vname}/null_random_dirs_absr"]["p95"]
        top = dd.reindex(dd.r_macro_within.abs().sort_values(ascending=False).index).head(14)
        summary[f"headline/{vname}"] = {
            "top_cues_by_within_class": [
                {
                    "cue": r.cue,
                    "r_macro_within": round(float(r.r_macro_within), 3),
                    "r_pooled": round(float(r.r_pooled), 3),
                    "partial_r": round(float(r.partial_r), 3),
                    "q_fdr": round(float(r.q_fdr), 4),
                }
                for r in top.itertuples()
            ],
            "n_cues_total": int(len(dd)),
            "n_fdr_sig_pooled": int((dd.q_fdr < 0.05).sum()),
            "n_above_null_p95_pooled": int((dd.r_pooled.abs() > null95).sum()),
            "n_above_null_p95_within": int((dd.r_macro_within.abs() > null95).sum()),
            "median_abs_r_pooled": round(float(dd.r_pooled.abs().median()), 3),
            "max_abs_partial_r": round(float(dd.partial_r.abs().max()), 3),
        }

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float)[:4000])
    print(f"\nwrote {OUT}/")


if __name__ == "__main__":
    main()
