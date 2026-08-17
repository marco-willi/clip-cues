#!/usr/bin/env python
"""F5 (spec E5): rank test images by the **matched canonical probe's** logit.

Replaces the retired auxiliary fixed-C=1 single-direction fit. Images are ranked by the actual
detector score ``z = w.h + b``; because ``b`` is constant, ranking by ``z`` is ranking by projection
onto ``w``. The figure caption becomes *"Images with the highest and lowest logits of the evaluated
canonical CLIP detector"* — no auxiliary model to explain.

Per the 2026-08-08 decision the ranking detector is the **matched** probe (F1's primary seed), not
the published augmented checkpoint: nothing is published yet, so the figure should show the detector
the rest of the paper reports. F7 records how closely the two agree.

Also emits the **seed-robustness** of the figure: the top/bottom-k overlap of the primary seed's
ranking against F1's other seeds. F1 found the direction far more stable than the extreme images, so
this number belongs next to the figure.

    uv run python scripts/finalexp/export_f5_rankings.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from clip_cues_research.finalexp import data as D
from clip_cues_research.finalexp import stability
from clip_cues_research.finalexp.runner import EXPERIMENTS_ROOT, Run
from clip_cues_research.finalexp.snapshot import register_artifact

EXPERIMENT = "F5-canonical-montage"
F1 = "F1-canonical-stability"
PRIMARY_SEED = 123
SEEDS = [123, 124, 125, 126, 127]


def run_dir(dataset: str, seed: int):
    name = f"seed{seed}" if dataset == "synthclic" else f"{dataset}_seed{seed}"
    return EXPERIMENTS_ROOT / F1 / "runs" / name


def load_logits(dataset: str, seed: int) -> pd.DataFrame:
    p = run_dir(dataset, seed) / "logits_test.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"No matched probe for {dataset} seed {seed} at {p}.\n"
            f"Train it: uv run python scripts/finalexp/run_f1_canonical_stability.py "
            f"--dataset {dataset}"
        )
    return pd.read_csv(p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["synthclic", "cnnspot"])
    ap.add_argument("--seed", type=int, default=PRIMARY_SEED)
    ap.add_argument("-k", type=int, default=8, help="images per pole in the montage")
    ap.add_argument("--overlap-n", type=int, default=50)
    args = ap.parse_args()

    out: dict[str, dict] = {}
    run = Run(EXPERIMENT, "artifacts", [f"pooler/{d}" for d in args.datasets])

    for ds in args.datasets:
        primary = load_logits(ds, args.seed)
        ranked = primary.sort_values("logit", ascending=False).reset_index(drop=True)
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
        path = run.save_csv(f"ranked_scores_{ds}.csv", ranked)
        # The ranking is also written INTO the snapshot: it is the artifact that stands in for the
        # montage's image pixels (not snapshottable), so it must carry a manifest checksum.
        rel = f"rankings/f5_{ds}.csv"
        snap = D.SNAPSHOT / rel
        snap.parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(snap, index=False)

        # Seed robustness of the figure itself.
        overlaps = []
        for seed in SEEDS:
            if seed == args.seed:
                continue
            try:
                other = load_logits(ds, seed)
            except FileNotFoundError:
                continue
            ov = stability.extreme_overlap(
                primary["logit"].to_numpy(), other["logit"].to_numpy(), n=args.overlap_n
            )
            overlaps.append({"seed": seed, **{k: v for k, v in ov.items() if k != "n"}})

        top = ranked.head(args.k)[["rank", "image_id", "source", "label", "logit"]]
        bottom = ranked.tail(args.k)[["rank", "image_id", "source", "label", "logit"]]
        out[ds] = {
            "ranking_csv": str(path),
            "n_test": int(len(ranked)),
            "primary_seed": args.seed,
            "detector": f"matched canonical probe (F1 {ds} seed {args.seed})",
            "logit_range": [
                round(float(ranked["logit"].min()), 4),
                round(float(ranked["logit"].max()), 4),
            ],
            "top_pole": top.to_dict("records"),
            "bottom_pole": bottom.to_dict("records"),
            "seed_overlap": overlaps,
            "seed_overlap_mean_top": (
                round(float(np.mean([o[f"top{args.overlap_n}_jaccard"] for o in overlaps])), 4)
                if overlaps
                else None
            ),
            "seed_overlap_mean_bottom": (
                round(float(np.mean([o[f"bottom{args.overlap_n}_jaccard"] for o in overlaps])), 4)
                if overlaps
                else None
            ),
        }

        # The ranking is the snapshot-registered artifact standing in for the image pixels, which
        # are far too large to snapshot (see EXCLUDED.md).
        register_artifact(
            artifact_id=f"ranking/f5_{ds}",
            path=rel,
            kind="ranking",
            space=D.SPACE_NA,
            used_by=["F5"],
            provenance=(
                f"scripts/finalexp/export_f5_rankings.py — test images ranked by the MATCHED "
                f"canonical probe (F1 {ds} seed {args.seed}) logit z = w.h + b. Stands in for the "
                f"montage's image pixels, which are not snapshottable; the figure is reproducible "
                f"from this ranking plus the HF dataset id + revision."
            ),
            derived_from=D.input_shas(f"pooler/{ds}"),
        )
        print(f"  {ds}: {len(ranked)} test images ranked -> {path}")
        if overlaps:
            print(
                f"    seed robustness (top-{args.overlap_n} Jaccard vs other seeds): "
                f"{out[ds]['seed_overlap_mean_top']:.3f}"
            )

    summary = {
        "experiment": EXPERIMENT,
        "spec_id": "E5",
        "datasets": args.datasets,
        "detector": "matched canonical probe (decision 3, 2026-08-08)",
        "caption": (
            "Images with the highest and lowest logits of the evaluated canonical CLIP detector."
        ),
        "retired": (
            "The auxiliary fixed-C=1 single-direction logistic fit previously used to rank "
            "montage images is retired; ranking now uses the detector's own logit."
        ),
        "by_dataset": out,
    }
    run.note(summary=summary)
    run.save_json("summary.json", summary)
    run.finish()


if __name__ == "__main__":
    main()
