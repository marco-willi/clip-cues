#!/usr/bin/env python
"""F3 (spec E3): the projected 768-d analysis head, trained under the matched recipe.

The one auxiliary full-space classifier with a mathematical reason to exist: text directions live
in the 768-d shared space, so any quantity like ``cos(w, v_q)`` needs a classifier direction there.
Previously that role was played by a *standardized, CV-tuned scikit-learn* probe (P768t), a third
training recipe requiring coefficient back-transformation machinery in the appendix. F3 replaces it
with the same head, optimizer, loss and schedule as F1 — so ``D_h`` and ``D_e`` differ *only* by the
projection, which is the claim under test.

Input is ``e = Wp h``, derived from the very same cached pooler frame F1 uses and rescaled by one
global scalar to the pooler train mean row norm. No standardization, no per-model tuning.

**Why the rescale** (measured 2026-08-08): the recipe fixes ``weight_decay = 0.01``, which is only
"the same regularization" if the spaces have comparable scale. They do not (mean row norms: pooler
32.95, projected 18.83, unit-normalized 1.00), and on unit-normalized features the identical recipe
reaches AUROC 0.725 vs 0.888 raw — with more epochs *not* closing the gap. One global scalar per
space preserves the geometry exactly, adds no per-dimension statistics, and makes the D_h/D_e
comparison about the projection rather than about optimization. See
``clip_cues_research.finalexp.features.match_scale``.

Reports AUROC for both, the **paired cluster-bootstrap ΔAUROC** (clusters = source photo), test-logit
correlation, and cue-profile agreement. Comparison target: N2c's matched-tuning projection cost
+0.021 [+0.013, +0.030].

    uv run python scripts/finalexp/run_f3_projected_head.py
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd

from clip_cues_research.finalexp import data as D
from clip_cues_research.finalexp import profiles, spaces, stability
from clip_cues_research.finalexp.evaluation import cluster_bootstrap_auroc_delta, evaluate_head
from clip_cues_research.finalexp.runner import EXPERIMENTS_ROOT, Run, run_context
from clip_cues_research.finalexp.trainer import RECIPE, train_head

EXPERIMENT = "F3-projected-head"
SEEDS = [123, 124, 125, 126, 127]


def load_f1_logits(seed: int) -> np.ndarray:
    p = EXPERIMENTS_ROOT / "F1-canonical-stability" / f"runs/seed{seed}" / "logits_test.csv"
    if not p.exists():
        raise FileNotFoundError(f"F1 run missing at {p}; run run_f1_canonical_stability.py first.")
    return pd.read_csv(p)["logit"].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="synthclic")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    inputs = [
        f"projected/{args.dataset}",
        f"pooler/{args.dataset}",
        f"cue_scores/{args.dataset}__antonyms",
        "vocab/antonyms",
        "projection/wp_l14_336",
    ]
    print(f"F3: projected 768-d head on {args.dataset}, seeds {args.seeds}")

    space = spaces.load(args.dataset, "projected")
    xtr, ytr, _ = space.split("train")
    xva, yva, _ = space.split("validation")
    xte, yte, dte = space.split("test")
    print(f"  space: {space.as_dict()}")

    C = D.get_npz(f"cue_scores/{args.dataset}__antonyms")
    cues_te = C["scores"][(space.df["split"] == "test").to_numpy()]
    cue_names = [str(c) for c in C["cues"]]

    runs: dict[int, dict] = {}
    for seed in args.seeds:
        with run_context(EXPERIMENT, f"runs/seed{seed}", inputs) as run:
            head = train_head(xtr, ytr, xva, yva, seed=seed)
            metrics, z = evaluate_head(head, xte, dte, args.dataset)
            prof = profiles.cue_profile(z, cues_te, yte)
            run.note(
                seed=seed,
                dataset=args.dataset,
                head="LinearHead",
                input_dim=head.input_dim,
                space=space.as_dict(),
                recipe=RECIPE.as_dict(),
                metrics=metrics,
                best_val_ce=head.best_val_ce,
                best_epoch=head.best_epoch,
            )
            run.save_json(
                "config.json", {"seed": seed, "dataset": args.dataset, **RECIPE.as_dict()}
            )
            run.save_json("metrics.json", metrics)
            run.save_npz("weights.npz", weight=head.weight, bias=np.array([head.bias]))
            run.save_csv(
                "logits_test.csv",
                pd.DataFrame(
                    {
                        "image_id": dte["image_id"].astype(str).values,
                        "source": dte["source"].values,
                        "label": yte,
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
            print(f"    seed {seed}: mAP {metrics['mAP']:.4f}  AUROC {metrics['auroc']:.4f}")
            runs[seed] = {"head": head, "metrics": metrics, "z": z, "profile": prof["pooled"]}

    agg = Run(EXPERIMENT, "artifacts", inputs)

    # ── seed stability of D_e itself ─────────────────────────────────────────────────────────
    pair_rows = []
    for a, b in combinations(sorted(runs), 2):
        pair_rows.append(
            {
                "seed_a": a,
                "seed_b": b,
                **stability.direction_agreement(
                    runs[a]["head"].weight, runs[b]["head"].weight, xte
                ),
                **stability.score_agreement(runs[a]["z"], runs[b]["z"]),
                "cue_profile_spearman": stability.profile_agreement(
                    runs[a]["profile"], runs[b]["profile"]
                ),
            }
        )
    pairs = pd.DataFrame(pair_rows)
    agg.save_csv("stability.csv", pairs)

    # ── D_h vs D_e, the projection cost ──────────────────────────────────────────────────────
    clusters = dte["image_id"].astype(str).to_numpy()
    per_seed = []
    for seed in sorted(runs):
        z_e = runs[seed]["z"]
        z_h = load_f1_logits(seed)
        boot = cluster_bootstrap_auroc_delta(z_h, z_e, yte, clusters, n_boot=args.n_boot)
        prof_h = profiles.col_corr(z_h, cues_te)
        per_seed.append(
            {
                "seed": seed,
                "auroc_Dh": boot["auroc_a"],
                "auroc_De": boot["auroc_b"],
                "delta_Dh_minus_De": boot["delta"],
                "ci_lo": boot["ci_lo"],
                "ci_hi": boot["ci_hi"],
                **{f"logit_{k}": v for k, v in stability.score_agreement(z_h, z_e).items()},
                "cue_profile_spearman_Dh_De": stability.profile_agreement(
                    prof_h, runs[seed]["profile"]
                ),
            }
        )
    proj = pd.DataFrame(per_seed)
    agg.save_csv("projection_cost.csv", proj)

    summary = {
        "experiment": EXPERIMENT,
        "spec_id": "E3",
        "dataset": args.dataset,
        "seeds": args.seeds,
        "recipe": RECIPE.as_dict(),
        "feature_construction": (
            "e = Wp h from the SAME cached pooler frame F1 uses (derived, not separately extracted "
            "- the both-sides-derived rule of EXTERNAL_VALIDATION_PROTOCOL.md), rescaled by ONE "
            "global scalar to the pooler train mean row norm so the recipe's fixed weight decay is "
            "the same amount of regularization in both spaces. Not unit-normalized and not "
            "standardized: D_h and D_e must differ only by the projection."
        ),
        "space": space.as_dict(),
        "De_detection": {
            "auroc_mean": float(proj["auroc_De"].mean()),
            "auroc_min": float(proj["auroc_De"].min()),
            "auroc_max": float(proj["auroc_De"].max()),
            "mAP_mean": float(np.mean([runs[s]["metrics"]["mAP"] for s in runs])),
        },
        "Dh_detection": {
            "auroc_mean": float(proj["auroc_Dh"].mean()),
        },
        "projection_cost_Dh_minus_De": {
            "per_seed_delta": proj["delta_Dh_minus_De"].round(6).tolist(),
            "mean_delta": round(float(proj["delta_Dh_minus_De"].mean()), 6),
            "ci_lo_mean": round(float(proj["ci_lo"].mean()), 6),
            "ci_hi_mean": round(float(proj["ci_hi"].mean()), 6),
            "reference_N2c": "+0.021 [+0.013, +0.030] (matched-tuning proxies)",
            "n_boot": args.n_boot,
            "clusters": "source photo (image_id)",
        },
        "Dh_De_agreement": {
            "logit_spearman_mean": round(float(proj["logit_logit_spearman"].mean()), 6),
            "cue_profile_spearman_mean": round(float(proj["cue_profile_spearman_Dh_De"].mean()), 6),
            "reference_E12": "proxy cue-profile agreement rho ~0.95 pooled / 0.86 within-class",
        },
        "De_seed_stability": {
            col: stability.summarize_pairs(pairs[col].tolist())
            for col in ("sigma_cosine", "raw_cosine", "logit_spearman", "cue_profile_spearman")
        },
    }
    agg.note(summary=summary)
    agg.save_json("summary.json", summary)
    agg.finish()

    pc = summary["projection_cost_Dh_minus_De"]
    print(f"\n  AUROC  D_h {proj['auroc_Dh'].mean():.4f}   D_e {proj['auroc_De'].mean():.4f}")
    print(
        f"  projection cost (D_h - D_e): {pc['mean_delta']:+.4f} "
        f"[{pc['ci_lo_mean']:+.4f}, {pc['ci_hi_mean']:+.4f}]   (N2c: +0.021 [+0.013, +0.030])"
    )
    print(
        f"  D_h~D_e logit Spearman {summary['Dh_De_agreement']['logit_spearman_mean']:.4f}, "
        f"cue-profile rho {summary['Dh_De_agreement']['cue_profile_spearman_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
