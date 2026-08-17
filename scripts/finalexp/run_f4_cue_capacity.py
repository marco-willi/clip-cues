#!/usr/bin/env python
"""F4 (spec E4): how much of the linearly accessible signal does the named vocabulary carry?

A **restricted-information probe, not a detector variant** — the spec's key reframing. The input is
the 168 (or 128) named cue scores ``c_j = <e/||e||, v_j>``; the classifier is the same head under
the same recipe as F1/F3. The comparison is against F3's unrestricted 768-d ``D_e``, so the
difference isolates *what the vocabulary cannot express* rather than a modelling choice.

Reports AUROC for both, the paired cluster-bootstrap ΔAUROC, and the excess-AUROC recovery ratio
``(AUROC_cue - 0.5)/(AUROC_De - 0.5)``. Comparison target: N5c — ant168t −0.036 [−0.050, −0.022],
recovery 90.5%.

    uv run python scripts/finalexp/run_f4_cue_capacity.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from clip_cues_research.finalexp import spaces, stability
from clip_cues_research.finalexp.evaluation import (
    cluster_bootstrap_auroc_delta,
    evaluate_head,
    excess_auroc_recovery,
)
from clip_cues_research.finalexp.runner import EXPERIMENTS_ROOT, Run, run_context
from clip_cues_research.finalexp.trainer import RECIPE, train_head

EXPERIMENT = "F4-cue-capacity"
SEEDS = [123, 124, 125, 126, 127]
# The published snapshot carries only the antonym vocabulary (the one the manuscript uses);
# F4's headline number -- the cost of restricting to the 168 named cues -- is the antonyms row.
VOCABS = ["antonyms"]


def load_f3_logits(seed: int) -> np.ndarray:
    p = EXPERIMENTS_ROOT / "F3-projected-head" / f"runs/seed{seed}" / "logits_test.csv"
    if not p.exists():
        raise FileNotFoundError(f"F3 run missing at {p}; run run_f3_projected_head.py first.")
    return pd.read_csv(p)["logit"].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="synthclic")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--vocabs", nargs="+", default=VOCABS)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    ref = spaces.load(args.dataset, "projected")
    split = ref.df["split"].to_numpy()
    y = ref.df["label"].to_numpy().astype(int)
    dte = ref.df[split == "test"].reset_index(drop=True)
    yte = y[split == "test"]
    clusters = dte["image_id"].astype(str).to_numpy()

    print(f"F4: cue-capacity probe on {args.dataset}, vocabs {args.vocabs}, seeds {args.seeds}")
    all_rows, summaries = [], {}

    for vocab in args.vocabs:
        inputs = [
            f"cue_scores/{args.dataset}__{vocab}",
            f"vocab/{vocab}",
            f"projected/{args.dataset}",
        ]
        cue_space = spaces.load(args.dataset, "cue", vocab=vocab)
        C = cue_space.x
        xtr, ytr, _ = cue_space.split("train")
        xva, yva, _ = cue_space.split("validation")
        xte, _, _ = cue_space.split("test")
        print(f"    space: {cue_space.as_dict()}")

        runs = {}
        for seed in args.seeds:
            with run_context(EXPERIMENT, f"runs/{vocab}_seed{seed}", inputs) as run:
                head = train_head(xtr, ytr, xva, yva, seed=seed)
                metrics, z = evaluate_head(head, xte, dte, args.dataset)
                run.note(
                    seed=seed,
                    vocabulary=vocab,
                    n_cues=int(C.shape[1]),
                    head="LinearHead",
                    framing="restricted-information probe",
                    space=cue_space.as_dict(),
                    recipe=RECIPE.as_dict(),
                    metrics=metrics,
                )
                run.save_json("metrics.json", metrics)
                run.save_npz("weights.npz", weight=head.weight, bias=np.array([head.bias]))
                run.save_csv(
                    "logits_test.csv",
                    pd.DataFrame(
                        {"image_id": dte["image_id"].astype(str).values, "label": yte, "logit": z}
                    ),
                )
                print(
                    f"    {vocab} seed {seed}: AUROC {metrics['auroc']:.4f}  mAP {metrics['mAP']:.4f}"
                )
                runs[seed] = {"z": z, "metrics": metrics}

        rows = []
        for seed in args.seeds:
            z_e = load_f3_logits(seed)
            boot = cluster_bootstrap_auroc_delta(
                runs[seed]["z"], z_e, yte, clusters, n_boot=args.n_boot
            )
            rows.append(
                {
                    "vocabulary": vocab,
                    "seed": seed,
                    "n_cues": int(C.shape[1]),
                    "auroc_cue": boot["auroc_a"],
                    "auroc_De": boot["auroc_b"],
                    "delta_cue_minus_De": boot["delta"],
                    "ci_lo": boot["ci_lo"],
                    "ci_hi": boot["ci_hi"],
                    "excess_recovery": excess_auroc_recovery(boot["auroc_a"], boot["auroc_b"]),
                    "logit_spearman_vs_De": stability.score_agreement(runs[seed]["z"], z_e)[
                        "logit_spearman"
                    ],
                }
            )
        df = pd.DataFrame(rows)
        all_rows.append(df)
        summaries[vocab] = {
            "n_cues": int(C.shape[1]),
            "auroc_cue_mean": round(float(df["auroc_cue"].mean()), 6),
            "auroc_De_mean": round(float(df["auroc_De"].mean()), 6),
            "delta_mean": round(float(df["delta_cue_minus_De"].mean()), 6),
            "ci_lo_mean": round(float(df["ci_lo"].mean()), 6),
            "ci_hi_mean": round(float(df["ci_hi"].mean()), 6),
            "excess_recovery_mean": round(float(df["excess_recovery"].mean()), 6),
            "logit_spearman_vs_De_mean": round(float(df["logit_spearman_vs_De"].mean()), 6),
        }

    agg = Run(
        EXPERIMENT,
        "artifacts",
        [f"cue_scores/{args.dataset}__{v}" for v in args.vocabs] + [f"projected/{args.dataset}"],
    )
    capacity = pd.concat(all_rows, ignore_index=True)
    agg.save_csv("capacity.csv", capacity)

    summary = {
        "experiment": EXPERIMENT,
        "spec_id": "E4",
        "dataset": args.dataset,
        "seeds": args.seeds,
        "recipe": RECIPE.as_dict(),
        "framing": (
            "D_cue is a RESTRICTED-INFORMATION PROBE, not a detector variant: same head, same "
            "recipe, same images as D_e - only the input is restricted to named cue scores."
        ),
        "feature_construction": (
            "c_j = <e/||e||, v_j> (a cosine by definition), then rescaled by ONE global scalar to "
            "the pooler train mean row norm - the same treatment D_e gets - so the fixed weight "
            "decay is the same amount of regularization in both. Without this the comparison "
            "measures optimization, not capacity: unscaled, the 168-cue RESTRICTED probe scored "
            "ABOVE the 768-d unrestricted probe it is a strict subspace of."
        ),
        "by_vocabulary": summaries,
        "primary": "antonyms",
        "reference_N5c": "ant168t -0.036 [-0.050, -0.022], excess recovery 90.5%",
        "n_boot": args.n_boot,
        "clusters": "source photo (image_id)",
    }
    agg.note(summary=summary)
    agg.save_json("summary.json", summary)
    agg.finish()

    for v, s in summaries.items():
        print(
            f"\n  {v} ({s['n_cues']} cues): AUROC {s['auroc_cue_mean']:.4f} vs D_e "
            f"{s['auroc_De_mean']:.4f}"
        )
        print(
            f"    delta {s['delta_mean']:+.4f} [{s['ci_lo_mean']:+.4f}, {s['ci_hi_mean']:+.4f}]"
            f"   excess recovery {s['excess_recovery_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
