#!/usr/bin/env python
"""F2 (spec E2): matched k=1 vs k=8 — the boundary is stable, its decomposition is not.

The k=1 probe and the k=8 orthogonal head were previously trained under *different* protocols (the
appendix says so), while the Results nevertheless compared their detection performance. F2 removes
that confound: identical cached features, identical recipe, identical seeds. The only changes are
the factorized parameterization and the activation-orthogonality penalty.

The measurement that matters is the distinction between

  * the **effective direction** ``w_eff = w2 @ W1`` (exact — the head is linear when
    ``non_linear=False``; this is E12's ``W0^T w_logit``), and
  * the **individual factorized axes** ``W1[j]``.

Individual axes are compared under **Hungarian matching on |cosine|**: without it, arbitrary axis
ordering between seeds manufactures "instability" and the headline claim would be an artifact of
permutation rather than a property of the factorization.

k=1 runs are reused from F1 by reference — same recipe, same features, which is the whole point.

    uv run python scripts/finalexp/run_f2_matched_k8.py
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations

import numpy as np
import pandas as pd

from clip_cues_research.finalexp import data as D
from clip_cues_research.finalexp import profiles, stability
from clip_cues_research.finalexp.evaluation import evaluate_head
from clip_cues_research.finalexp.runner import EXPERIMENTS_ROOT, Run, run_context
from clip_cues_research.finalexp.trainer import RECIPE, make_ortho_head, train_head

EXPERIMENT = "F2-matched-k8"
F1_EXPERIMENT = "F1-canonical-stability"
SEEDS = [123, 124, 125, 126, 127]
K = 8
LAMBDA_ORTHO = 0.33


def load_f1_runs(seeds: list[int]) -> dict[int, dict]:
    """Re-read F1's k=1 runs (weights + test logits) rather than retraining them."""
    out: dict[int, dict] = {}
    for seed in seeds:
        d = EXPERIMENTS_ROOT / F1_EXPERIMENT / f"runs/seed{seed}"
        if not (d / "weights.npz").exists():
            raise FileNotFoundError(
                f"F1 run missing at {d}. Run scripts/finalexp/run_f1_canonical_stability.py first."
            )
        w = np.load(d / "weights.npz")
        logits = pd.read_csv(d / "logits_test.csv")
        out[seed] = {
            "weight": w["weight"],
            "bias": float(w["bias"][0]),
            "z": logits["logit"].to_numpy(),
            "metrics": json.loads((d / "metrics.json").read_text()),
        }
    return out


def train_k8(dataset: str, seeds: list[int], k: int, inputs: list[str]) -> dict[int, dict]:
    frame = D.get_frame(f"pooler/{dataset}", expected_space=D.SPACE_POOLER)
    xtr, ytr, _ = frame.split("train")
    xva, yva, _ = frame.split("validation")
    xte, _, dte = frame.split("test")

    C = D.get_npz(f"cue_scores/{dataset}__antonyms")
    cues_te = C["scores"][(frame.df["split"] == "test").to_numpy()]
    cue_names = [str(c) for c in C["cues"]]
    y_te = dte["label"].to_numpy().astype(int)

    out: dict[int, dict] = {}
    for seed in seeds:
        with run_context(EXPERIMENT, f"runs/k8_seed{seed}", inputs) as run:
            head = train_head(
                xtr,
                ytr,
                xva,
                yva,
                seed=seed,
                head_factory=lambda d, _k=k: make_ortho_head(d, k=_k, lam=LAMBDA_ORTHO),
                head_type="ortho_k8",
            )
            metrics, z = evaluate_head(head, xte, dte, dataset)
            prof = profiles.cue_profile(z, cues_te, y_te)

            run.note(
                seed=seed,
                dataset=dataset,
                head=f"ActivationOrthogonalityHead[k={k}]",
                k=k,
                lambda_ortho=LAMBDA_ORTHO,
                non_linear=False,
                recipe=RECIPE.as_dict(),
                metrics=metrics,
                best_val_ce=head.best_val_ce,
                best_epoch=head.best_epoch,
                epochs_run=head.epochs_run,
            )
            run.save_json(
                "config.json",
                {
                    "seed": seed,
                    "dataset": dataset,
                    "k": k,
                    "lambda_ortho": LAMBDA_ORTHO,
                    **RECIPE.as_dict(),
                },
            )
            run.save_json("metrics.json", metrics)
            run.save_npz(
                "weights.npz", w_eff=head.weight, bias=np.array([head.bias]), axes=head.axes
            )
            run.save_csv(
                "logits_test.csv",
                pd.DataFrame(
                    {
                        "image_id": dte["image_id"].astype(str).values,
                        "source": dte["source"].values,
                        "label": y_te,
                        "logit": z,
                    }
                ),
            )
            run.save_csv(
                "cue_profile.csv",
                pd.DataFrame(
                    {
                        "cue": cue_names,
                        "pooled_r": prof["pooled"],
                        "within_macro_r": prof["within_macro"],
                    }
                ),
            )
            print(
                f"    k=8 seed {seed}: mAP {metrics['mAP']:.4f}  AUROC {metrics['auroc']:.4f}  "
                f"val-CE {head.best_val_ce:.4f}"
            )
            out[seed] = {
                "weight": head.weight,
                "bias": head.bias,
                "axes": head.axes,
                "z": z,
                "metrics": metrics,
                "profile": prof["pooled"],
            }
    return out


def axis_profiles(axes: np.ndarray, x: np.ndarray, cues: np.ndarray) -> np.ndarray:
    """Per-axis cue profile: correlation of each factorized axis's projection with every cue."""
    return np.vstack([profiles.col_corr(x @ a, cues) for a in axes])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="synthclic")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("-k", type=int, default=K)
    args = ap.parse_args()

    inputs = [f"pooler/{args.dataset}", f"cue_scores/{args.dataset}__antonyms", "vocab/antonyms"]
    print(f"F2: matched k=1 vs k={args.k} on {args.dataset}, seeds {args.seeds}")

    k1 = load_f1_runs(args.seeds)
    print(f"  reusing {len(k1)} k=1 runs from {F1_EXPERIMENT} (identical recipe and features)")
    k8 = train_k8(args.dataset, args.seeds, args.k, inputs)

    frame = D.get_frame(f"pooler/{args.dataset}", expected_space=D.SPACE_POOLER)
    xte, _, dte = frame.split("test")
    te = (frame.df["split"] == "test").to_numpy()
    cues_te = D.get_npz(f"cue_scores/{args.dataset}__antonyms")["scores"][te]

    agg = Run(EXPERIMENT, "artifacts", inputs)
    rows = []

    # ── within-family stability across seed pairs ────────────────────────────────────────────
    for family, runs, use_axes in (
        ("k1_refits", k1, False),
        ("k8_effective", k8, False),
        ("k8_individual_axes", k8, True),
    ):
        for a, b in combinations(sorted(runs), 2):
            ra, rb = runs[a], runs[b]
            if use_axes:
                m = stability.matched_axis_cosines(ra["axes"], rb["axes"])
                pa, pb = (axis_profiles(r["axes"], xte, cues_te) for r in (ra, rb))
                # rank-correlate each Hungarian-matched axis pair's cue profile
                pair_rho = [
                    stability.profile_agreement(pa[i], pb[j])
                    for i, j in zip(*_hungarian_pairs(ra["axes"], rb["axes"]))
                ]
                rows.append(
                    {
                        "family": family,
                        "seed_a": a,
                        "seed_b": b,
                        "raw_cosine": m["mean"],
                        "sigma_cosine": np.nan,
                        "cue_profile_spearman": float(np.mean(pair_rho)),
                        "logit_spearman": np.nan,
                        "top50_jaccard": np.nan,
                        "axis_cos_min": m["min"],
                        "axis_cos_max": m["max"],
                    }
                )
            else:
                rows.append(
                    {
                        "family": family,
                        "seed_a": a,
                        "seed_b": b,
                        **stability.direction_agreement(ra["weight"], rb["weight"], xte),
                        **stability.score_agreement(ra["z"], rb["z"]),
                        **{
                            k: v
                            for k, v in stability.extreme_overlap(ra["z"], rb["z"]).items()
                            if k != "n"
                        },
                        "cue_profile_spearman": stability.profile_agreement(
                            profiles.col_corr(ra["z"], cues_te), profiles.col_corr(rb["z"], cues_te)
                        ),
                    }
                )
    pairs = pd.DataFrame(rows)
    agg.save_csv("stability.csv", pairs)

    # ── cross-family: does the k=8 effective direction equal the k=1 boundary? ───────────────
    cross = []
    for seed in sorted(set(k1) & set(k8)):
        cross.append(
            {
                "seed": seed,
                **stability.direction_agreement(k1[seed]["weight"], k8[seed]["weight"], xte),
                **stability.score_agreement(k1[seed]["z"], k8[seed]["z"]),
                "cue_profile_spearman": stability.profile_agreement(
                    profiles.col_corr(k1[seed]["z"], cues_te),
                    profiles.col_corr(k8[seed]["z"], cues_te),
                ),
            }
        )
    cross_df = pd.DataFrame(cross)
    agg.save_csv("k1_vs_k8_same_seed.csv", cross_df)

    def fam(name: str, col: str) -> dict:
        v = pairs.loc[pairs["family"] == name, col].dropna()
        return stability.summarize_pairs(v.tolist()) if len(v) else {"mean": None}

    k1_m = pd.DataFrame([k1[s]["metrics"] for s in sorted(k1)])
    k8_m = pd.DataFrame([k8[s]["metrics"] for s in sorted(k8)])

    summary = {
        "experiment": EXPERIMENT,
        "spec_id": "E2",
        "dataset": args.dataset,
        "seeds": args.seeds,
        "k": args.k,
        "lambda_ortho": LAMBDA_ORTHO,
        "recipe": RECIPE.as_dict(),
        "note": (
            "k=1 runs reused from F1 (identical recipe/features). Individual axes are compared "
            "under Hungarian matching on |cosine|, so the result is not an artifact of axis "
            "ordering. Sigma-cosine is the primary direction metric for the effective directions; "
            "individual axes are compared by raw |cosine| because each axis alone has no calibrated "
            "score scale."
        ),
        "detection": {
            "k1": {
                "mAP_mean": float(k1_m["mAP"].mean()),
                "auroc_mean": float(k1_m["auroc"].mean()),
                "auroc_min": float(k1_m["auroc"].min()),
                "auroc_max": float(k1_m["auroc"].max()),
            },
            "k8": {
                "mAP_mean": float(k8_m["mAP"].mean()),
                "auroc_mean": float(k8_m["auroc"].mean()),
                "auroc_min": float(k8_m["auroc"].min()),
                "auroc_max": float(k8_m["auroc"].max()),
            },
        },
        "stability_table": {
            "k1_refits": {
                "sigma_cosine": fam("k1_refits", "sigma_cosine"),
                "raw_cosine": fam("k1_refits", "raw_cosine"),
                "cue_profile_spearman": fam("k1_refits", "cue_profile_spearman"),
                "top50_jaccard": fam("k1_refits", "top50_jaccard"),
            },
            "k8_effective_direction": {
                "sigma_cosine": fam("k8_effective", "sigma_cosine"),
                "raw_cosine": fam("k8_effective", "raw_cosine"),
                "cue_profile_spearman": fam("k8_effective", "cue_profile_spearman"),
                "top50_jaccard": fam("k8_effective", "top50_jaccard"),
            },
            "k8_individual_axes": {
                "matched_abs_cosine": fam("k8_individual_axes", "raw_cosine"),
                "cue_profile_spearman": fam("k8_individual_axes", "cue_profile_spearman"),
            },
        },
        "k1_vs_k8_same_seed": {
            "sigma_cosine_mean": float(cross_df["sigma_cosine"].mean()),
            "raw_cosine_mean": float(cross_df["raw_cosine"].mean()),
            "logit_spearman_mean": float(cross_df["logit_spearman"].mean()),
            "cue_profile_spearman_mean": float(cross_df["cue_profile_spearman"].mean()),
        },
    }
    agg.note(summary=summary)
    agg.save_json("summary.json", summary)
    agg.finish()

    t = summary["stability_table"]
    print("\n  quantity                     k=1 refits   k=8 effective   k=8 individual axes")
    print(
        f"  AUROC (mean)                 {k1_m['auroc'].mean():.4f}       "
        f"{k8_m['auroc'].mean():.4f}          —"
    )
    print(
        f"  direction cosine (Sigma)     {t['k1_refits']['sigma_cosine']['mean']:.4f}       "
        f"{t['k8_effective_direction']['sigma_cosine']['mean']:.4f}          "
        f"{t['k8_individual_axes']['matched_abs_cosine']['mean']:.4f} (raw, matched)"
    )
    print(
        f"  cue-profile Spearman         {t['k1_refits']['cue_profile_spearman']['mean']:.4f}       "
        f"{t['k8_effective_direction']['cue_profile_spearman']['mean']:.4f}          "
        f"{t['k8_individual_axes']['cue_profile_spearman']['mean']:.4f}"
    )
    print(
        f"\n  cos(k=8 w_eff, k=1 w) same seed: Sigma "
        f"{summary['k1_vs_k8_same_seed']['sigma_cosine_mean']:.4f}, "
        f"raw {summary['k1_vs_k8_same_seed']['raw_cosine_mean']:.4f}"
    )


def _hungarian_pairs(axes_a: np.ndarray, axes_b: np.ndarray):
    """Row/col indices of the optimal |cosine| matching between two axis sets."""
    from scipy.optimize import linear_sum_assignment

    a = axes_a / np.clip(np.linalg.norm(axes_a, axis=1, keepdims=True), 1e-12, None)
    b = axes_b / np.clip(np.linalg.norm(axes_b, axis=1, keepdims=True), 1e-12, None)
    return linear_sum_assignment(-np.abs(a @ b.T))


if __name__ == "__main__":
    main()
