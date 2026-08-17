#!/usr/bin/env python
"""E8: exploit SynthCLIC's REAL↔SYNTHETIC pairing (content held fixed) for clean cue interpretation.

Each real photo has 4 synthetic counterparts (imagen3/SD3/FLUX-dev/FLUX-schnell) from its caption, sharing
image_id. The paired difference d = embed(synthetic) - embed(real) isolates the real->synthetic shift with
SCENE/CONTENT CONTROLLED — removing the content confound that made the unpaired analyses diffuse/semantic.

  A. Shift direction + consistency: mean shift d̄ (pooler 1024-d); per-pair cosine to d̄; PCA variance along
     PC1 (is the shift a single consistent direction?). Per-generator d̄_g + cross-generator cosines (shared signature?).
  B. Detection: AUROC of test images projected on d̄ (content-controlled cue) vs the logreg detector; cos(d̄, logreg).
  C. CONTENT-CONTROLLED per-cue aggregation (the headline): in projected 768-d, Δq = q(syn)-q(real); for each
     antonym cue, mean±std change across pairs -> effect size (mean/std). Ranks the cues that CONSISTENTLY shift
     real->synthetic with content fixed. Compare to the unpaired single-cue AUROC.

Embeddings-only, CPU. Persists outputs/e8/paired/. Run: python scripts/analyze/analyze_paired_shift.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

OUT = Path("outputs/e8/paired")
OUT.mkdir(parents=True, exist_ok=True)


def unit(v):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def load(p):
    d = pickle.load(open(p, "rb"))
    return d["df"].reset_index(drop=True), d["embeddings"].astype(np.float64)


def pairs(df, emb, split):
    """Return per-(image_id,generator) difference S-R and the real/synth embeddings, for a split."""
    m = (df["split"] == split).to_numpy()
    sub = df[m].reset_index()
    E = emb[m]
    diffs = []
    gens = []
    reals = []
    syns = []
    for iid, grp in sub.groupby("image_id"):
        r = grp[grp.label == 0]
        if len(r) != 1:
            continue
        R = E[r.index[0]]
        for _, row in grp[grp.label == 1].iterrows():
            S = E[row["index"]] if "index" in row else E[row.name]
            diffs.append(S - R)
            gens.append(row["source"])
            reals.append(R)
            syns.append(S)
    return np.array(diffs), np.array(gens), np.array(reals), np.array(syns)


def center_basis(A, k):
    """P-Q2.a: remove the common-mode of a cue basis by projecting out its top-k right singular vectors,
    then re-unit-normalize the rows. k=0 is a no-op. For TextSpan the shared common-mode (location/style +
    a large mean component) dominates a *difference*-vector projection; stripping it tests whether the
    'strange' TextSpan paired result is entirely common-mode/capacity confound."""
    if k <= 0:
        return A
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    Vk = Vt[:k]
    A = A - (A @ Vk.T) @ Vk
    return unit(A)


def load_vocab(which, center_pcs=0):
    """Load the cue basis. 'combined' stacks antonyms (photographic) + textspan (semantic) with name prefixes."""
    a = torch.load("data/embeddings/antonyms_diff_embeddings.pt")
    An = ["ant:" + n for n in a["vocabulary"]]
    Ae = a["embeddings"].numpy().astype(np.float64)
    if which == "antonyms":
        return center_basis(unit(Ae), center_pcs), An
    t = torch.load("data/embeddings/textspan_embeddings.pt")
    Tn = ["ts:" + n for n in t["vocabulary"]]
    Te = np.asarray(t["embeddings"], dtype=np.float64)
    if which == "textspan":
        return center_basis(unit(Te), center_pcs), Tn
    return center_basis(unit(np.vstack([Ae, Te])), center_pcs), An + Tn  # combined


def effect_size(shift):
    """Per-cue effect size = mean/std across pairs (Cohen's-d-like consistency). shift: (n_pairs, n_cues)."""
    m = shift.mean(0)
    s = shift.std(0)
    return m / np.clip(s, 1e-9, None), m, s


def paired_sign_permutation_null(shift, B, seed=0):
    """P-Q2.b: paired permutation null. Swapping real<->synth within a pair flips that pair's diff sign,
    so the null is a random +/-1 sign-flip per pair. Vectorized: under sign-flips E[x^2] is invariant, only
    the mean changes. Returns per-cue empirical p (two-sided) and the max-|effect| null (for FWER)."""
    rng = np.random.default_rng(seed)
    n, _ = shift.shape
    Ex2 = (shift**2).mean(0)  # invariant under sign flips
    signs = rng.choice([-1.0, 1.0], size=(B, n))
    means = (signs @ shift) / n  # (B, n_cues)
    std = np.sqrt(np.clip(Ex2[None] - means**2, 1e-18, None))
    eff_null = np.abs(means / np.clip(std, 1e-9, None))  # (B, n_cues)
    eff_obs = np.abs(effect_size(shift)[0])
    p = (1.0 + (eff_null >= eff_obs[None]).sum(0)) / (B + 1.0)
    max_null = eff_null.max(1)  # (B,) for FWER / max-stat
    return p, max_null


def bh_fdr(p, alpha=0.05):
    """Benjamini-Hochberg: return boolean reject mask at FDR alpha."""
    p = np.asarray(p)
    order = np.argsort(p)
    m = len(p)
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        kmax = np.where(passed)[0].max()
        reject[order[: kmax + 1]] = True
    return reject


def bootstrap_ci(shift, B, seed=0, alpha=0.05):
    """P-Q2.b: percentile bootstrap CI on per-cue effect size by resampling pairs with replacement.
    Vectorized via multinomial counts: weighted mean and E[x^2] give the resampled mean/std."""
    rng = np.random.default_rng(seed + 1)
    n, _ = shift.shape
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float64)  # (B, n)
    mean_b = (counts @ shift) / n
    ex2_b = (counts @ (shift**2)) / n
    std_b = np.sqrt(np.clip(ex2_b - mean_b**2, 1e-18, None))
    eff_b = mean_b / np.clip(std_b, 1e-9, None)  # (B, n_cues)
    lo = np.percentile(eff_b, 100 * alpha / 2, axis=0)
    hi = np.percentile(eff_b, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="antonyms", choices=["antonyms", "textspan", "combined"])
    ap.add_argument(
        "--center-vocab",
        type=int,
        default=0,
        help="P-Q2.a: project out the top-k common-mode singular vectors of the cue basis (0=off).",
    )
    ap.add_argument(
        "--permute",
        type=int,
        default=0,
        help="P-Q2.b: paired sign-flip permutations for the cue-shift null (e.g. 2000; 0=off).",
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        help="P-Q2.b: bootstrap resamples for per-cue effect-size CIs (e.g. 2000; 0=off).",
    )
    ap.add_argument(
        "--per-generator",
        action="store_true",
        help="P-Q2.c: also export the per-cue shift table separately per generator.",
    )
    a = ap.parse_args()
    pdf, pemb = load("data/embeddings/synthclic_clip_large_patch14.pkl")  # pooler 1024
    qdf, qemb = load("data/embeddings/synthclic_projected_embeddings.pkl")  # projected 768
    A, names = load_vocab(a.vocab, a.center_vocab)
    tag = a.vocab + (f"_c{a.center_vocab}" if a.center_vocab else "")
    res = {"vocab": a.vocab, "center_vocab_pcs": a.center_vocab}

    # ---- A: shift direction + consistency (pooler, train) ----
    Dtr, gtr, _, _ = pairs(pdf, pemb, "train")
    dbar = unit(Dtr.mean(0))
    cos_to_mean = unit(Dtr) @ dbar
    # PCA of centered diffs: variance along PC1 / total
    Dc = Dtr - Dtr.mean(0)
    _, sv, _ = np.linalg.svd(Dc, full_matrices=False)
    var = sv**2
    pc1 = float(var[0] / var.sum())
    res["n_pairs_train"] = int(len(Dtr))
    res["mean_cos_to_dbar"] = float(cos_to_mean.mean())
    res["frac_pairs_aligned_cos>0.3"] = float((cos_to_mean > 0.3).mean())
    res["pca_pc1_var_frac"] = pc1
    # per-generator
    gen_dirs = {g: unit(Dtr[gtr == g].mean(0)) for g in np.unique(gtr)}
    gcos = {
        f"{a}|{b}": float(gen_dirs[a] @ gen_dirs[b])
        for i, a in enumerate(gen_dirs)
        for b in list(gen_dirs)[i + 1 :]
    }
    res["per_generator_dbar_cosines"] = gcos
    res["each_gen_cos_to_global_dbar"] = {g: float(gen_dirs[g] @ dbar) for g in gen_dirs}

    # ---- B: detection from the content-controlled direction ----
    mtr = (pdf["split"] == "train").to_numpy()
    mev = (pdf["split"] == "test").to_numpy()
    ytr = pdf.loc[mtr, "label"].to_numpy()
    yev = pdf.loc[mev, "label"].to_numpy()
    score_dbar = pemb[mev] @ dbar
    res["AUROC_dbar_direction_test"] = float(roc_auc_score(yev, score_dbar))
    sc = StandardScaler().fit(pemb[mtr])
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(pemb[mtr]), ytr)
    res["AUROC_logreg_test"] = float(
        roc_auc_score(yev, lr.decision_function(sc.transform(pemb[mev])))
    )
    wlog = unit(lr.coef_.ravel() / sc.scale_)
    res["cos_dbar_logreg"] = float(dbar @ wlog)

    # ---- C: content-controlled per-cue aggregation (projected, all pairs) ----
    Dq, gq, Rq, Sq = pairs(qdf, qemb, "train")
    dq = unit(Sq) - unit(Rq)  # change in (unit) projected embedding per pair
    shift = dq @ A.T  # (n_pairs, n_cues) change in each cue
    eff, mean_c, std_c = effect_size(shift)
    cue_cols = {
        "cue": names,
        "mean_shift": mean_c,
        "std_shift": std_c,
        "effect_size": eff,
        "frac_pairs_same_sign": np.maximum((shift > 0).mean(0), (shift < 0).mean(0)),
    }
    # ---- P-Q2.b: paired permutation null (FDR + FWER) and bootstrap CIs on the effect sizes ----
    if a.permute:
        pvals, max_null = paired_sign_permutation_null(shift, a.permute)
        reject_fdr = bh_fdr(pvals, 0.05)
        fwer_thresh = float(np.percentile(max_null, 95))  # max-stat 5% FWER threshold
        cue_cols["perm_p"] = pvals
        cue_cols["sig_fdr05"] = reject_fdr
        cue_cols["sig_fwer05"] = np.abs(eff) > fwer_thresh
        res["permute"] = a.permute
        res["n_cues_sig_fdr05"] = int(reject_fdr.sum())
        res["n_cues_sig_fwer05"] = int((np.abs(eff) > fwer_thresh).sum())
        res["fwer05_abs_effect_threshold"] = fwer_thresh
    if a.bootstrap:
        lo, hi = bootstrap_ci(shift, a.bootstrap)
        cue_cols["eff_ci_lo"] = lo
        cue_cols["eff_ci_hi"] = hi
        cue_cols["ci_excludes_zero"] = (lo > 0) | (hi < 0)
        res["bootstrap"] = a.bootstrap
        res["n_cues_ci_excludes_zero"] = int(((lo > 0) | (hi < 0)).sum())
    cue_tab = pd.DataFrame(cue_cols).sort_values("effect_size", key=np.abs, ascending=False)
    cue_tab.to_csv(OUT / f"paired_cue_shifts_{tag}.csv", index=False)
    # ---- P-Q2.c: per-generator per-cue shift table (does each diffusion family shift the same cues?) ----
    if a.per_generator:
        per_gen = {}
        for g in np.unique(gq):
            sg = shift[gq == g]
            eg, mg, sdg = effect_size(sg)
            gt = (
                pd.DataFrame(
                    {
                        "cue": names,
                        "mean_shift": mg,
                        "effect_size": eg,
                        "frac_pairs_same_sign": np.maximum((sg > 0).mean(0), (sg < 0).mean(0)),
                    }
                )
                .sort_values("effect_size", key=np.abs, ascending=False)
                .reset_index(drop=True)
            )
            gt.to_csv(OUT / f"paired_cue_shifts_{tag}_{g}.csv", index=False)
            per_gen[g] = {
                "n_pairs": int((gq == g).sum()),
                "top10_cues": gt.head(10)["cue"].tolist(),
                "n_cues_abs_eff>0.5": int((np.abs(eg) > 0.5).sum()),
            }
        # rank-correlation of the per-cue effect-size vectors across generators (do they agree?)
        from scipy.stats import spearmanr

        gens = list(np.unique(gq))
        effs = {g: effect_size(shift[gq == g])[0] for g in gens}
        gen_rho = {
            f"{gi}|{gj}": float(spearmanr(effs[gi], effs[gj]).statistic)
            for i, gi in enumerate(gens)
            for gj in gens[i + 1 :]
        }
        res["per_generator"] = per_gen
        res["per_generator_effect_rank_corr"] = gen_rho
    # effect-size distribution (content-control test: photographic cues should shift, scene cues should NOT)
    ae = np.abs(eff)
    res["effectsize_max"] = float(ae.max())
    res["effectsize_p99"] = float(np.percentile(ae, 99))
    res["n_cues_abs_eff>0.5"] = int((ae > 0.5).sum())
    res["n_cues"] = len(names)
    res["frac_cues_abs_eff>0.5"] = float((ae > 0.5).mean())
    # in combined mode, split the top-50 by source prefix to see if photographic (ant:) dominates
    if a.vocab == "combined":
        top50 = cue_tab.head(50)["cue"].tolist()
        res["top50_antonym_frac"] = float(sum(c.startswith("ant:") for c in top50) / 50)
    show = [
        c
        for c in ["cue", "mean_shift", "effect_size", "frac_pairs_same_sign", "perm_p", "sig_fdr05"]
        if c in cue_tab
    ]
    res["top_paired_cues_by_effect"] = cue_tab.head(15)[show].to_dict("records")

    (OUT / f"paired_summary_{tag}.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, (list, dict))}, indent=2))
    print("\nper-generator d̄ cross-cosines:", {k: round(v, 3) for k, v in gcos.items()})
    print(
        "each generator vs global d̄:",
        {k: round(v, 3) for k, v in res["each_gen_cos_to_global_dbar"].items()},
    )
    print(
        f"\nAUROC: content-controlled d̄ direction={res['AUROC_dbar_direction_test']:.3f}  vs logreg={res['AUROC_logreg_test']:.3f}  cos(d̄,logreg)={res['cos_dbar_logreg']:.3f}"
    )
    print(
        "\nTOP content-controlled per-cue shifts (effect size = consistency; +=synthetic higher):"
    )
    print(cue_tab.head(15)[show].to_string(index=False))
    if a.permute:
        print(
            f"\nP-Q2.b null: {res['n_cues_sig_fdr05']}/{res['n_cues']} cues sig at FDR 0.05; "
            f"{res['n_cues_sig_fwer05']} at FWER 0.05 (|eff|>{res['fwer05_abs_effect_threshold']:.3f})"
        )
    if a.bootstrap:
        print(
            f"P-Q2.b bootstrap: {res['n_cues_ci_excludes_zero']}/{res['n_cues']} cues' 95% CI excludes 0"
        )
    if a.per_generator:
        print(
            "\nP-Q2.c per-generator effect-size rank-corr:",
            {k: round(v, 3) for k, v in res["per_generator_effect_rank_corr"].items()},
        )


if __name__ == "__main__":
    main()
