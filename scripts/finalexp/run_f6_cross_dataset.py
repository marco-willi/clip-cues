#!/usr/bin/env python
"""F6 (spec E6): cross-dataset projected heads + boundary decomposition, matched recipe.

E11b already computed the cross-dataset boundary cosines and the signed Delta decomposition in the
derived shared space — but with *CV-tuned scikit-learn* probes (C: SC 0.01, CNNSpot 0.001, SB+ 0.01).
F6 swaps **only the probe-fitting step**: the normals now come from the matched `LinearHead(768)`
that F3 trains, so there is no separate family of "cross-dataset boundary probes" and no
standardization/back-transformation machinery. Everything downstream is E11's.

N21 is the pre-registered comparison target:
    unit-normal cosines  sc~cnnspot -0.102,  sc~sb+ +0.161,  cnnspot~sb+ +0.082
    signed Delta(cnnspot - sc): CNNSpot side = compression/processing artifacts,
                                SynthCLIC side = photographic aesthetics/provenance
Agreement means the conclusion survives the recipe change; disagreement is a finding about probe
regularization, reported as such.

**Split discipline:** SynthBuster+ **train/val only**. The frozen protocol permits no further SB+
*test* reads, and E11b likewise stayed on train/val. Asserted in code.

**Caveat carried forward:** CNNSpot's 768-d problem is near-trivially separable (E11b used C=0.001;
E12 saw own-probe AUROC 1.000). With fixed wd 0.01 and no per-dataset tuning the matched CNNSpot
probe is *less* regularized than E11b's, so its normal may be even more weakly identified — the
train/val separation is reported alongside the cosines so a reader can judge.

    uv run python scripts/finalexp/run_f6_cross_dataset.py
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd

from clip_cues_research.finalexp import data as D
from clip_cues_research.finalexp import spaces, stability
from clip_cues_research.finalexp.evaluation import score_metrics
from clip_cues_research.finalexp.runner import Run, run_context
from clip_cues_research.finalexp.trainer import RECIPE, train_head
from clip_cues_research.vocab_opt.boundary import (
    knee_row,
    lasso_path_decompose,
    support_stability,
    unitv,
)

EXPERIMENT = "F6-cross-dataset"
DATASETS = ["synthclic", "cnnspot", "synthbuster-plus"]
SEEDS = [123, 124, 125]
ALPHAS = np.logspace(-4, -1.2, 24)[::-1]

_SPACES: dict[str, spaces.Space] = {}


def load_space(dataset: str, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Scale-matched derived 768-d features for one split. SB+ test reads are refused in-code."""
    if dataset not in _SPACES:
        _SPACES[dataset] = spaces.load(dataset, "projected")
    return _SPACES[dataset].split(split)


def train_dataset_heads(dataset: str, seeds: list[int], inputs: list[str]) -> dict[int, dict]:
    xtr, ytr, _ = load_space(dataset, "train")
    xva, yva, dva = load_space(dataset, "validation")
    out = {}
    for seed in seeds:
        with run_context(EXPERIMENT, f"runs/{dataset}_seed{seed}", inputs) as run:
            head = train_head(xtr, ytr, xva, yva, seed=seed)
            z = head.logits(xva)
            metrics = score_metrics(z, dva, dataset)
            run.note(
                seed=seed,
                dataset=dataset,
                eval_split="validation",
                head="LinearHead(768, matched recipe)",
                recipe=RECIPE.as_dict(),
                metrics=metrics,
                best_val_ce=head.best_val_ce,
            )
            run.save_json("metrics.json", metrics)
            run.save_npz("weights.npz", weight=head.weight, bias=np.array([head.bias]))
            print(
                f"    {dataset} seed {seed}: val AUROC {metrics['auroc']:.4f}  "
                f"mAP {metrics['mAP']:.4f}"
            )
            out[seed] = {"weight": head.weight, "bias": head.bias, "metrics": metrics, "z": z}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--vocab", default="antonyms")
    ap.add_argument("--n-boot", type=int, default=50)
    args = ap.parse_args()

    inputs = [f"projected/{d}" for d in args.datasets] + [f"vocab/{args.vocab}"]
    print(f"F6: cross-dataset matched 768-d heads on {args.datasets}, seeds {args.seeds}")

    heads = {ds: train_dataset_heads(ds, args.seeds, inputs) for ds in args.datasets}
    agg = Run(EXPERIMENT, "artifacts", inputs)

    # Mean unit normal per dataset (seeds are a stability check, not an ensemble claim).
    normals = {
        ds: unitv(np.mean([heads[ds][s]["weight"] for s in args.seeds], axis=0))
        for ds in args.datasets
    }

    # ── seed stability per dataset ───────────────────────────────────────────────────────────
    seed_rows = []
    for ds in args.datasets:
        xva, _, _ = load_space(ds, "validation")
        for a, b in combinations(args.seeds, 2):
            seed_rows.append(
                {
                    "dataset": ds,
                    "seed_a": a,
                    "seed_b": b,
                    **stability.direction_agreement(
                        heads[ds][a]["weight"], heads[ds][b]["weight"], xva
                    ),
                }
            )
    agg.save_csv("seed_stability.csv", pd.DataFrame(seed_rows))

    # ── pairwise boundary cosines (the N21 comparison) ───────────────────────────────────────
    cos_rows = []
    for a, b in combinations(args.datasets, 2):
        xa, _, _ = load_space(a, "validation")
        cos_rows.append(
            {
                "dataset_a": a,
                "dataset_b": b,
                "raw_cosine": round(float(normals[a] @ normals[b]), 6),
                "sigma_cosine_on_a": stability.sigma_cosine(normals[a], normals[b], xa),
            }
        )
    cos_df = pd.DataFrame(cos_rows)
    agg.save_csv("boundary_cosines.csv", cos_df)

    # ── signed Delta decomposition: what CNNSpot weights toward "synthetic" vs SynthCLIC ─────
    V, cue_names = D.get_vocab(f"vocab/{args.vocab}")
    decomposition = {}
    if {"cnnspot", "synthclic"} <= set(args.datasets):
        delta = normals["cnnspot"] - normals["synthclic"]
        xtr_sc, _, _ = load_space("synthclic", "train")
        xtr_cn, _, _ = load_space("cnnspot", "train")
        Vtr = np.vstack([xtr_sc, xtr_cn])  # data-weighted on the union, as in E11b
        xva_sc, yva_sc, dva_sc = load_space("synthclic", "validation")

        rows = lasso_path_decompose(Vtr, xva_sc, yva_sc, delta, V, ALPHAS)
        knee = knee_row(rows)
        freq = support_stability(
            Vtr,
            np.concatenate([np.arange(len(xtr_sc)), np.arange(len(xtr_cn))]),
            delta,
            V,
            knee["alpha"],
            n_boot=args.n_boot,
        )
        coef = knee["coef"]
        order = np.argsort(-np.abs(coef))
        top = [
            {
                "cue": cue_names[i],
                "alpha_coef": round(float(coef[i]), 6),
                "boot_freq": round(float(freq[i]), 3),
                "side": "cnnspot_synthetic" if coef[i] > 0 else "synthclic_synthetic",
            }
            for i in order[:25]
            if coef[i] != 0
        ]
        agg.save_csv("delta_axes.csv", pd.DataFrame(top))
        agg.save_csv(
            "delta_path.csv",
            pd.DataFrame([{k: v for k, v in r.items() if k != "coef"} for r in rows]),
        )
        decomposition = {
            "target": "Delta = w_hat(cnnspot) - w_hat(synthclic)",
            "vocabulary": args.vocab,
            "knee_nnz": int(knee["nnz"]),
            "knee_alpha": round(float(knee["alpha"]), 8),
            "val_score_r2": round(float(knee["val_score_r2"]), 6),
            "cos_coverage": round(float(knee["cos_coverage"]), 6),
            "top_axes": top,
            "reference_N21": (
                "CNNSpot side = compression/processing artifacts; SynthCLIC side = "
                "aesthetics/provenance; score-R2 0.96 with v2-128"
            ),
        }

    # ── identifiability caveat: how separable is each dataset in this space? ─────────────────
    separability = {
        ds: {
            "val_auroc_mean": round(
                float(np.mean([heads[ds][s]["metrics"]["auroc"] for s in args.seeds])), 6
            ),
            "val_mAP_mean": round(
                float(np.mean([heads[ds][s]["metrics"]["mAP"] for s in args.seeds])), 6
            ),
        }
        for ds in args.datasets
    }

    summary = {
        "experiment": EXPERIMENT,
        "spec_id": "E6",
        "datasets": args.datasets,
        "seeds": args.seeds,
        "recipe": RECIPE.as_dict(),
        "note": (
            "Normals come from the MATCHED LinearHead(768), not CV-tuned sklearn probes; only "
            "the probe-fitting step of E11b changed. No raw_normal/back-transformation is "
            "involved, which is the appendix simplification the spec asks for."
        ),
        "split_discipline": "SynthBuster+ train/val only; test reads refused in code.",
        "boundary_cosines": cos_df.to_dict("records"),
        "reference_N21_cosines": {
            "synthclic~cnnspot": -0.102,
            "synthclic~synthbuster-plus": 0.161,
            "cnnspot~synthbuster-plus": 0.082,
        },
        "seed_stability_sigma_cosine": {
            ds: stability.summarize_pairs(
                [r["sigma_cosine"] for r in seed_rows if r["dataset"] == ds]
            )
            for ds in args.datasets
        },
        "separability_caveat": separability,
        "delta_decomposition": decomposition,
    }
    agg.note(summary=summary)
    agg.save_json("summary.json", summary)
    agg.finish()

    print("\n  boundary cosines (mean unit normals, matched recipe):")
    for r in cos_df.to_dict("records"):
        ref = summary["reference_N21_cosines"].get(f"{r['dataset_a']}~{r['dataset_b']}")
        print(
            f"    {r['dataset_a']:18s} ~ {r['dataset_b']:18s} raw {r['raw_cosine']:+.4f}"
            + (f"   (N21: {ref:+.3f})" if ref is not None else "")
        )
    if decomposition:
        print(
            f"\n  Delta decomposition: knee {decomposition['knee_nnz']} axes, "
            f"val score-R2 {decomposition['val_score_r2']:.3f}"
        )
        for t in decomposition["top_axes"][:8]:
            print(
                f"    {t['cue']:32s} {t['alpha_coef']:+.4f}  ({t['side']}, freq {t['boot_freq']})"
            )


if __name__ == "__main__":
    main()
