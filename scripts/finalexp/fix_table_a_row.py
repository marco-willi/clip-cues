#!/usr/bin/env python
"""Step 11: make the Table A "CLIP linear probe (SynthCLIC)" row uniform on the matched probe.

The row in ``reproduction/revision_export/tex/e1_e3_e6_e7_detector_comparison.tex`` is **provenance-mixed**:

  * cross-dataset columns (0.92 / 0.79 / 0.42) come from ``results/e3_xdataset/`` — the re-trained
    probe on cached embeddings, **no augmentation**;
  * the CF-Eval column (0.73) comes from ``data/checkpoints/linear_probe_synthclic.ckpt`` — the
    **published, augmented** checkpoint.

T0 (config-audit.md §F) fixed this row's *head* mismatch (k=8 -> k=1) but not this
*checkpoint-provenance* mismatch, and T0's stated goal was "no table mixes heads/metrics". Per the
2026-08-08 decision the row is made uniform on the **matched** probe: F1's primary-seed head is
scored on the cached CF-Eval embeddings through the E7 pipeline, so all four columns describe one
model trained one way.

The deployed k=1 / k=8 CF-Eval numbers (0.732 / 0.734) are kept as the appendix ablation.

Requires the CF-Eval frame in the snapshot:

    make finalexp-data WITH_CFEVAL=1
    uv run python scripts/finalexp/fix_table_a_row.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from clip_cues_research.community_eval import cf_metrics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "export"))
from export_community_eval_tables import per_generator  # noqa: E402

from clip_cues_research.finalexp import data as D
from clip_cues_research.finalexp.runner import EXPERIMENTS_ROOT, Run

EXPERIMENT = "F5-canonical-montage"  # bookkeeping lives with the table-fix artifacts
OUT_EXPERIMENT = "TableA-uniform"
DETECTOR = "linear_probe_synthclic_matched"


def matched_head(seed: int) -> tuple[np.ndarray, float]:
    w = np.load(EXPERIMENTS_ROOT / "F1-canonical-stability" / f"runs/seed{seed}" / "weights.npz")
    return w["weight"], float(w["bias"][0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    if "pooler/cf_eval" not in D.load_manifest():
        raise SystemExit(
            "pooler/cf_eval is not in the snapshot.\n"
            "Build it first:  make finalexp-data WITH_CFEVAL=1"
        )

    run = Run(OUT_EXPERIMENT, "artifacts", ["pooler/cf_eval", "pooler/synthclic"])
    frame = D.get_frame("pooler/cf_eval", expected_space=D.SPACE_POOLER)
    w, b = matched_head(args.seed)

    z = frame.emb.astype(np.float64) @ w + b
    preds = frame.df.copy()
    preds["score"] = 1.0 / (1.0 + np.exp(-z))
    metrics = cf_metrics(preds)

    # Table A's CF-Eval cell is **mAP-by-generator**, not pooled AP, so it must be computed the way
    # the export does (mean AP within each `source` group) — quoting cf_metrics' pooled `mAP`
    # against the deployed row's 0.7316 would compare two different quantities.
    gen = per_generator(preds)
    map_by_gen = float(gen["ap"].mean())

    run.save_csv("cfeval_predictions_matched.csv", preds[["image_id", "label", "source", "score"]])
    run.save_csv("cfeval_per_generator.csv", gen)
    summary = {
        "experiment": OUT_EXPERIMENT,
        "detector": DETECTOR,
        "source": f"F1-canonical-stability/runs/seed{args.seed} (matched recipe, no augmentation)",
        "cfeval_metrics": {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in metrics.items()
            if not isinstance(v, (pd.DataFrame, dict))
        },
        "mAP_by_generator": round(map_by_gen, 6),
        "n_generators": int(len(gen)),
        "metric_note": (
            "`mAP_by_generator` is the Table A quantity (mean AP within each `source` group, the "
            "export's convention). cf_metrics' `mAP` field is the POOLED AP and is not comparable "
            "to the deployed row's mAP_by_gen."
        ),
        "deployed_reference": {
            "linear_probe_synthclic": {"overall_ap": 0.7326, "mAP_by_gen": 0.7316},
            "clip_orthogonal_synthclic": {"overall_ap": 0.7296, "mAP_by_gen": 0.7340},
        },
        "why": (
            "Makes the Table A row uniform: columns 1-3 already come from the matched no-aug probe "
            "(results/e3_xdataset), so the CF-Eval cell should too. The deployed numbers stay as the "
            "appendix ablation."
        ),
        "next_step": (
            "Point package_revision_export.py table_a()'s `clip_cf` lookup at this row, or add a "
            "MANIFEST footnote naming the mixed provenance."
        ),
    }
    run.note(summary=summary)
    run.save_json("summary.json", summary)
    run.finish()

    print(
        f"  matched probe on CF-Eval: overall_ap {metrics['overall_ap']:.4f}  "
        f"mAP_by_gen {map_by_gen:.4f}  ({len(gen)} generators)"
    )
    print("  deployed k=1 reference:   overall_ap 0.7326  mAP_by_gen 0.7316")
    print(
        f"  delta (matched - deployed): overall_ap {metrics['overall_ap'] - 0.7326:+.4f}  "
        f"mAP_by_gen {map_by_gen - 0.7316:+.4f}"
    )


if __name__ == "__main__":
    main()
