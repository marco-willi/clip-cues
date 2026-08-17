#!/usr/bin/env python
"""Explainability follow-up Step 2 — name the exposure gap in cue space (PLAN_EXPLAINABILITY_FOLLOWUPS).

The linear probes live in the 1024-d pooler space; the cue vocabularies (168 antonym
diff-directions, 3,498 TextSpan terms) live in the 768-d projected space. HF CLIP's
``image_embeds = visual_projection(pooler_output)`` is an exact linear map z = P x, so a probe
direction w maps to cue space as the least-squares preimage v = (P P^T)^{-1} P w (the component of
w readable through the projection; the captured norm fraction is reported). Validated: P@pooler
matches the legacy ``*_projected_embeddings.pkl`` per-row cosine ~0.995 (residual = preprocessing
jitter of the older extraction), per-cue activation corr ~0.99.

Analyses:
  1. Cue profile per probe (synthclic / synthbuster / cnnspot / combined): loadings of the
     cue-space unit direction v_hat on the antonym cue basis (+ TextSpan top terms).
  2. Delta directions (cnnspot - synthclic, combined - synthclic): which named cues does GAN
     exposure ADD?
  3. Bootstrap validation: B=20 logreg refits per training set -> cue-profile rank stability and
     agreement with the checkpoint probe's profile (guards against single-fit artifacts).
  4. Per-architecture CF-Eval cue gaps: per-generator Cohen's d (fake vs matched reals) of each
     antonym cue activation, averaged by architecture; then Spearman(|gap|, |probe loading|) per
     architecture = does the probe READ the cues that actually shift for that family?

Append-only: writes outputs/explain/exposure_cues/ only.

Run (local, CPU):
    uv run python scripts/analyze/analyze_exposure_cues.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CKPTS = {
    "synthclic": "data/checkpoints/linear_probe_synthclic.ckpt",
    "synthbuster": "data/checkpoints/linear_probe_synthbuster.ckpt",
    "cnnspot": "data/checkpoints/linear_probe_cnnspot.ckpt",
    "combined": "data/checkpoints/linear_probe_combined.ckpt",
}
POOLER = {
    "synthclic": "data/embeddings/synthclic_clip_large_patch14.pkl",
    "cnnspot": "data/embeddings/cnnspot_clip_large_patch14.pkl",
}
LEGACY_PROJ = "data/embeddings/synthclic_projected_embeddings.pkl"
CF_EMB = "data/embeddings/communityforensics_l14_eval.pkl"
OUT = Path("outputs/explain/exposure_cues")
SEED = 123
B_BOOT = 20


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v), 1e-12, None)


def unit_rows(M: np.ndarray) -> np.ndarray:
    return M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-12, None)


def probe_direction(ckpt: str) -> np.ndarray:
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
    return sd["model.classification_head.fc.weight"].numpy().astype(np.float64).ravel()


def load_pkl(path: str) -> tuple[pd.DataFrame, np.ndarray]:
    d = pickle.load(open(path, "rb"))
    return d["df"].reset_index(drop=True), np.asarray(d["embeddings"], dtype=np.float64)


def visual_projection() -> np.ndarray:
    from transformers import CLIPVisionModelWithProjection

    m = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-large-patch14-336", cache_dir="data/hf_cache"
    )
    return m.visual_projection.weight.detach().numpy().astype(np.float64)  # (768, 1024)


def cue_preimage(w: np.ndarray, P: np.ndarray, PPt_inv: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares cue-space representation v of a pooler-space direction w (z = P x)."""
    v = PPt_inv @ (P @ w)
    captured = float(np.linalg.norm(P.T @ v) / np.clip(np.linalg.norm(w), 1e-12, None))
    return v, captured


def validate_projection(P: np.ndarray) -> dict:
    dfp, x = load_pkl(POOLER["synthclic"])
    dfz, z = load_pkl(LEGACY_PROJ)
    assert (dfp["image_id"].to_numpy() == dfz["image_id"].to_numpy()).all()
    zh = x @ P.T
    c = (unit_rows(zh) * unit_rows(z)).sum(1)
    return {
        "per_row_cosine_mean": float(c.mean()),
        "per_row_cosine_min": float(c.min()),
        "note": "residual vs legacy projected pkl = preprocessing jitter of the older extraction",
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Column-wise Cohen's d between two sample groups."""
    pooled = np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2)
    return (a.mean(0) - b.mean(0)) / np.clip(pooled, 1e-12, None)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    P = visual_projection()
    PPt_inv = np.linalg.inv(P @ P.T)
    val = validate_projection(P)
    print("projection validation:", val)

    av = torch.load("data/embeddings/antonyms_diff_embeddings.pt")
    A = unit_rows(av["embeddings"].numpy().astype(np.float64))
    cue_names = list(av["vocabulary"])
    ts = torch.load("data/embeddings/textspan_embeddings.pt")
    T = unit_rows(ts["embeddings"].numpy().astype(np.float64))
    ts_names = list(ts["vocabulary"])

    w = {k: probe_direction(p) for k, p in CKPTS.items()}
    dirs = dict(w)
    dirs["delta_cnnspot-synthclic"] = unit(w["cnnspot"]) - unit(w["synthclic"])
    dirs["delta_combined-synthclic"] = unit(w["combined"]) - unit(w["synthclic"])

    # 1+2) cue profiles (antonyms full table; TextSpan top +/- 25 per direction)
    prof = pd.DataFrame({"cue": cue_names})
    captured = {}
    ts_rows = []
    for name, d in dirs.items():
        v, cap = cue_preimage(d, P, PPt_inv)
        captured[name] = cap
        vh = unit(v)
        prof[name] = A @ vh
        tl = T @ vh
        order = np.argsort(-tl)
        for rank, i in enumerate(list(order[:25]) + list(order[-25:])):
            ts_rows.append(
                {"direction": name, "rank": rank, "term": ts_names[i], "loading": float(tl[i])}
            )
    prof.to_csv(OUT / "antonym_cue_profiles.csv", index=False)
    pd.DataFrame(ts_rows).to_csv(OUT / "textspan_top_terms.csv", index=False)
    for name in dirs:
        top = prof.reindex(prof[name].abs().sort_values(ascending=False).index)[["cue", name]]
        print(f"\n{name} (captured {captured[name]:.2f}) top cues:")
        print(top.head(8).to_string(index=False))

    # 3) bootstrap validation of the profiles (synthclic + cnnspot training sets)
    boot = {}
    rng = np.random.default_rng(SEED)
    for ds, pkl in POOLER.items():
        df, emb = load_pkl(pkl)
        m = (df["split"] == "train").to_numpy()
        X, y = emb[m], df.loc[m, "label"].to_numpy().astype(int)
        sc = StandardScaler().fit(X)
        profs, bdirs = [], []
        for _ in range(B_BOOT):
            idx = rng.integers(0, len(X), len(X))
            lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X[idx]), y[idx])
            bw = unit(lr.coef_.ravel() / sc.scale_)
            bdirs.append(bw)
            v, _ = cue_preimage(bw, P, PPt_inv)
            profs.append(A @ unit(v))
        profs = np.array(profs)
        pairs = [
            spearmanr(np.abs(profs[i]), np.abs(profs[j])).statistic
            for i in range(B_BOOT)
            for j in range(i + 1, B_BOOT)
        ]
        ck_prof = prof[ds].to_numpy()
        boot[ds] = {
            "bootstrap_profile_rank_corr": float(np.mean(pairs)),
            "ckpt_vs_mean_bootstrap_profile_rank_corr": float(
                spearmanr(np.abs(ck_prof), np.abs(profs.mean(0))).statistic
            ),
            "ckpt_vs_logreg_direction_cos": float(np.mean([abs(unit(w[ds]) @ b) for b in bdirs])),
        }
        print(ds, "bootstrap:", boot[ds])

    # 4) per-architecture CF-Eval cue gaps (per-generator Cohen's d, averaged by architecture)
    cf = pickle.load(open(CF_EMB, "rb"))
    meta, X = cf["df"].reset_index(drop=True), np.asarray(cf["embeddings"], dtype=np.float64)
    acts = unit_rows(X @ P.T) @ A.T  # (n, 168) cue activations
    gen_rows = []
    for (src, arch), g in meta.groupby(["source", "architecture"]):
        i = g.index.to_numpy()
        fake = acts[i[meta.loc[i, "label"].to_numpy() == 1]]
        real = acts[i[meta.loc[i, "label"].to_numpy() == 0]]
        gen_rows.append(pd.Series(cohens_d(fake, real), index=cue_names, name=(src, arch)))
    gen_d = pd.DataFrame(gen_rows)
    gen_d.index = pd.MultiIndex.from_tuples(gen_d.index, names=["generator", "architecture"])
    arch_d = gen_d.groupby("architecture").mean().T  # cues x arch
    arch_d.insert(0, "cue", cue_names)
    arch_d.to_csv(OUT / "cf_arch_cue_gaps.csv", index=False)

    # does each probe read the cues that shift, per architecture?
    align_rows = []
    for arch in arch_d.columns[1:]:
        gaps = arch_d[arch].to_numpy()
        row = {"architecture": arch, "mean_abs_gap_d": float(np.abs(gaps).mean())}
        for pname in CKPTS:
            row[f"gap_vs_{pname}_loading_rank_corr"] = float(
                spearmanr(np.abs(gaps), np.abs(prof[pname].to_numpy())).statistic
            )
        # signed agreement: does the probe's signed loading point the same way the cue shifts?
        for pname in CKPTS:
            row[f"gap_signed_corr_{pname}"] = float(
                spearmanr(gaps, prof[pname].to_numpy()).statistic
            )
        align_rows.append(row)
    align = pd.DataFrame(align_rows)
    align.to_csv(OUT / "cf_gap_probe_alignment.csv", index=False)
    print("\nper-architecture cue-gap vs probe-loading alignment:")
    print(align.round(3).to_string(index=False))

    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "projection_validation": val,
                "captured_norm_fraction": captured,
                "bootstrap_validation": boot,
                "seed": SEED,
                "B_bootstrap": B_BOOT,
            },
            indent=2,
        )
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
