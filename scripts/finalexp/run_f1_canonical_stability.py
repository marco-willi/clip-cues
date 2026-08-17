#!/usr/bin/env python
"""F1 (spec E1): stability of the canonical 1024-d detector across refits.

Trains the canonical `LinearHead(1024)` on SynthCLIC pooler features for 5 seeds under the matched
recipe, and measures how much the decision direction, the scores, the cue profile and the extreme
images move. This is what licenses interpreting the *actual* detector rather than an auxiliary
convex fit.

**Metric discipline:** the primary direction metric is the data-metric (Sigma) cosine, not the raw
weight cosine — N21 found raw cos 0.07 between two detectors whose scores correlate 0.938, so a raw
seed-cosine could call a perfectly stable detector unstable. Both are reported; the README quotes
Sigma first.

**Regression anchor:** seed 123 must reproduce `reference/e3_seed123` (mAP 0.9239, AUROC 0.9227) —
the run behind the manuscript's Table A row. If it does not, the shared trainer differs from
`scripts/run/run_linear_probe.py` and nothing downstream is trustworthy.

    uv run python scripts/finalexp/run_f1_canonical_stability.py
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd

from clip_cues_research.analysis.metrics import detection_metrics, pairing_for_dataset
from clip_cues_research.finalexp import data as D
from clip_cues_research.finalexp import profiles, stability
from clip_cues_research.finalexp.runner import Run, run_context
from clip_cues_research.finalexp.trainer import RECIPE, TrainedHead, train_head

EXPERIMENT = "F1-canonical-stability"
SEEDS = [123, 124, 125, 126, 127]
ANCHOR_TOL = 5e-3


def evaluate(head: TrainedHead, x: np.ndarray, df: pd.DataFrame, dataset: str) -> dict:
    """Convention-A mAP + pooled AUROC for one head on one split."""
    from sklearn.metrics import roc_auc_score

    z = head.logits(x)
    pred = pd.DataFrame(
        {
            "image_id": df["image_id"].astype(str).values,
            "label": df["label"].to_numpy().astype(int),
            "score": 1.0 / (1.0 + np.exp(-z)),
            "source": df["source"].values,  # attached BY POSITION (image_id is non-unique)
        }
    )
    bundle = detection_metrics(pred, real_pairing=pairing_for_dataset(dataset))
    return {
        "mAP": float(bundle["mAP"]),
        "pooled_ap": float(bundle["pooled_ap"]),
        "auroc": float(roc_auc_score(pred["label"], pred["score"])),
        "n_generators": int(bundle["n_generators"]),
        "real_pairing": pairing_for_dataset(dataset),
    }


def train_seeds(dataset: str, seeds: list[int], inputs: list[str]) -> dict[int, dict]:
    """Train one head per seed, persisting each run folder with its own provenance."""
    frame = D.get_frame(f"pooler/{dataset}", expected_space=D.SPACE_POOLER)
    xtr, ytr, _ = frame.split("train")
    xva, yva, _ = frame.split("validation")
    xte, _, dte = frame.split("test")

    C = D.get_npz(f"cue_scores/{dataset}__antonyms")
    cues, cue_names = C["scores"], [str(c) for c in C["cues"]]
    te_mask = (frame.df["split"] == "test").to_numpy()
    cues_te = cues[te_mask]

    out: dict[int, dict] = {}
    for seed in seeds:
        # SynthCLIC is the default target and its runs are consumed by F2/F3/F5 under this exact
        # name; other datasets (F5's CNNSpot panel) are namespaced so they cannot collide.
        run_name = f"runs/seed{seed}" if dataset == "synthclic" else f"runs/{dataset}_seed{seed}"
        with run_context(EXPERIMENT, run_name, inputs) as run:
            head = train_head(xtr, ytr, xva, yva, seed=seed)
            metrics = evaluate(head, xte, dte, dataset)
            z = head.logits(xte)
            prof = profiles.cue_profile(z, cues_te, dte["label"].to_numpy().astype(int))

            run.note(
                seed=seed,
                dataset=dataset,
                head="LinearHead",
                input_dim=head.input_dim,
                recipe=RECIPE.as_dict(),
                metrics=metrics,
                best_val_ce=head.best_val_ce,
                best_epoch=head.best_epoch,
                epochs_run=head.epochs_run,
            )
            run.save_json("config.json", {"seed": seed, "dataset": dataset, **RECIPE.as_dict()})
            run.save_json("metrics.json", metrics)
            run.save_npz("weights.npz", weight=head.weight, bias=np.array([head.bias]))
            run.save_csv(
                "logits_test.csv",
                pd.DataFrame(
                    {
                        "image_id": dte["image_id"].astype(str).values,
                        "source": dte["source"].values,
                        "label": dte["label"].to_numpy().astype(int),
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
                f"    seed {seed}: mAP {metrics['mAP']:.4f}  AUROC {metrics['auroc']:.4f}  "
                f"val-CE {head.best_val_ce:.4f} @ epoch {head.best_epoch} ({head.epochs_run} run)"
            )
            out[seed] = {"head": head, "metrics": metrics, "z": z, "profile": prof["pooled"]}
    return out


def pairwise_stability(runs: dict[int, dict], x_test: np.ndarray) -> pd.DataFrame:
    """Every seed pair, on every stability metric."""
    rows = []
    for a, b in combinations(sorted(runs), 2):
        ha, hb = runs[a]["head"], runs[b]["head"]
        rows.append(
            {
                "seed_a": a,
                "seed_b": b,
                **stability.direction_agreement(ha.weight, hb.weight, x_test),
                **stability.score_agreement(runs[a]["z"], runs[b]["z"]),
                **{
                    k: v
                    for k, v in stability.extreme_overlap(runs[a]["z"], runs[b]["z"]).items()
                    if k != "n"
                },
                "cue_profile_spearman": stability.profile_agreement(
                    runs[a]["profile"], runs[b]["profile"]
                ),
            }
        )
    return pd.DataFrame(rows)


def check_anchor(seed123: dict) -> dict:
    """Compare the seed-123 run against the persisted `run_linear_probe.py` result."""
    anchor = D.get_json("reference/e3_seed123")
    got = seed123["metrics"]
    d_map, d_auroc = abs(got["mAP"] - anchor["mAP"]), abs(got["auroc"] - anchor["auroc"])
    out = {
        "anchor_mAP": anchor["mAP"],
        "anchor_auroc": anchor["auroc"],
        "f1_seed123_mAP": got["mAP"],
        "f1_seed123_auroc": got["auroc"],
        "delta_mAP": round(d_map, 6),
        "delta_auroc": round(d_auroc, 6),
        "tolerance": ANCHOR_TOL,
        "passes": bool(d_map <= ANCHOR_TOL and d_auroc <= ANCHOR_TOL),
    }
    flag = "PASS" if out["passes"] else "FAIL"
    print(
        f"\n  [{flag}] regression anchor: mAP {got['mAP']:.4f} vs {anchor['mAP']:.4f} "
        f"(d={d_map:.4f}), AUROC {got['auroc']:.4f} vs {anchor['auroc']:.4f} (d={d_auroc:.4f})"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="synthclic")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()

    inputs = [f"pooler/{args.dataset}", f"cue_scores/{args.dataset}__antonyms", "vocab/antonyms"]
    if args.dataset == "synthclic":
        inputs.append("reference/e3_seed123")

    print(f"F1: canonical 1024-d stability on {args.dataset}, seeds {args.seeds}")
    runs = train_seeds(args.dataset, args.seeds, inputs)

    frame = D.get_frame(f"pooler/{args.dataset}", expected_space=D.SPACE_POOLER)
    xte, _, _ = frame.split("test")

    suffix = "" if args.dataset == "synthclic" else f"_{args.dataset}"
    agg = Run(EXPERIMENT, f"artifacts{suffix}", inputs)
    pairs = pairwise_stability(runs, xte)
    agg.save_csv("stability.csv", pairs)

    per_seed = pd.DataFrame(
        [
            {"seed": s, **runs[s]["metrics"], "best_val_ce": runs[s]["head"].best_val_ce}
            for s in sorted(runs)
        ]
    )
    agg.save_csv("per_seed_metrics.csv", per_seed)

    # The persisted anchor is the SynthCLIC seed-123 run_linear_probe.py result; there is no
    # equivalent for other datasets, so the check only applies there.
    anchor = check_anchor(runs[123]) if (args.dataset == "synthclic" and 123 in runs) else None
    summary = {
        "experiment": EXPERIMENT,
        "spec_id": "E1",
        "dataset": args.dataset,
        "seeds": args.seeds,
        "recipe": RECIPE.as_dict(),
        "regression_anchor": anchor,
        "detection": {
            "mAP": {
                "mean": float(per_seed["mAP"].mean()),
                "min": float(per_seed["mAP"].min()),
                "max": float(per_seed["mAP"].max()),
            },
            "auroc": {
                "mean": float(per_seed["auroc"].mean()),
                "min": float(per_seed["auroc"].min()),
                "max": float(per_seed["auroc"].max()),
            },
        },
        "stability_across_seed_pairs": {
            col: stability.summarize_pairs(pairs[col].tolist())
            for col in (
                "sigma_cosine",
                # "whitened_cosine" is not a separate column: stability.whitened_cosine is a
                # documented alias of sigma_cosine, and direction_agreement stopped emitting it
                # so one piece of evidence is not reported twice. The pre-fix artifact carried
                # both keys with identical values (0.989125).
                "raw_cosine",
                "logit_spearman",
                "decision_agreement",
                "cue_profile_spearman",
                "top50_jaccard",
                "bottom50_jaccard",
            )
        },
        "primary_direction_metric": "sigma_cosine",
        "metric_note": (
            "Sigma-metric cosine is primary; raw weight cosine is reported for continuity only "
            "(N21: raw cos 0.07 between detectors whose scores correlate 0.938)."
        ),
    }
    agg.note(summary=summary)
    agg.save_json("summary.json", summary)
    agg.finish()

    s = summary["stability_across_seed_pairs"]
    print(
        f"\n  AUROC {per_seed['auroc'].mean():.4f} [{per_seed['auroc'].min():.4f}, "
        f"{per_seed['auroc'].max():.4f}]  mAP {per_seed['mAP'].mean():.4f}"
    )
    print(
        f"  Sigma-cosine    {s['sigma_cosine']['mean']:.4f} "
        f"[{s['sigma_cosine']['min']:.4f}, {s['sigma_cosine']['max']:.4f}]"
    )
    print(
        f"  raw cosine      {s['raw_cosine']['mean']:.4f} "
        f"[{s['raw_cosine']['min']:.4f}, {s['raw_cosine']['max']:.4f}]"
    )
    print(f"  logit Spearman  {s['logit_spearman']['mean']:.4f}")
    print(f"  cue-profile rho {s['cue_profile_spearman']['mean']:.4f}")
    print(f"  top-50 overlap  {s['top50_jaccard']['mean']:.4f}")


if __name__ == "__main__":
    main()
