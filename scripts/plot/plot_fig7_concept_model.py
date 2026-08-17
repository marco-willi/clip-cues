#!/usr/bin/env python
"""Fig 7 — what the sparse concept model uses (CNNSpot, SynthCLIC).

Logic in ``src/clip_cues_research/figures/concept_model_profile.py``. Scores the **published**
concept-bottleneck checkpoints (``cm_antonyms_*``) on the checksummed projected snapshot; no GPU and
no retraining. The checkpoint's own text space is verified against ``vocab_canon/antonyms.pt`` before
any concept is named.

Two variants, both built by default:

  ``original``  a faithful rebuild of the published figure -- three panels (class separation,
                activation probability, predictive power), concepts ordered by single-concept AUC,
                dumbbells with the same real/synthetic encoding. ``--panels 2`` drops the AUC column.
  ``compact``   the leaner alternative: mean contribution bars with class-mean ticks, plus an
                activation strip.

    uv run python scripts/plot/plot_fig7_concept_model.py
    uv run python scripts/plot/plot_fig7_concept_model.py --variant original --panels 2
"""

from __future__ import annotations

import argparse

from clip_cues_research.figures.concept_model_profile import (
    concept_importance_figure,
    concept_model_figure,
)


def _report_compact(res, top_k: int) -> None:
    for ds, sub in res["table"].groupby("dataset"):
        n = int(sub["n_images"].iloc[0])
        top = sub.reindex(sub["contribution"].abs().sort_values(ascending=False).index).head(4)
        print(f"  {ds:10s} n={n:6d}  top contributions:")
        for r in top.itertuples():
            print(
                f"     {r.concept:26s} {r.contribution:+.4f}   activation {r.activation_prob:.2f}"
            )


def _report_original(res) -> None:
    for ds, sub in res["table"].groupby("dataset", sort=False):
        print(f"  {ds:10s} top concepts by single-concept AUC:")
        for r in sub.head(5).itertuples():
            sep = r.contribution_synth - r.contribution_real
            print(
                f"     {r.concept:26s} AUC {r.auc:.3f}   usage real {r.usage_real:.2f} / "
                f"synth {r.usage_synth:.2f}   class separation {sep:+.3f}"
            )
        strong = int((sub["auc"] > 0.6).sum())
        print(f"     {strong}/{len(sub)} shown concepts exceed AUC 0.6")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="both", choices=["both", "original", "compact"])
    ap.add_argument("--top-k", type=int, default=14, help="concepts per dataset (plan asks 10-15)")
    ap.add_argument(
        "--panels",
        type=int,
        default=3,
        choices=[2, 3],
        help="original variant: 3 keeps the AUC column, 2 drops it to a table",
    )
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-folder", default="fig7-concept-model")
    a = ap.parse_args()

    if a.variant in ("both", "original"):
        res = concept_importance_figure(
            top_k=a.top_k, panels=a.panels, split=a.split, out_folder=a.out_folder
        )
        print(f"original variant ({a.panels} panels, top {a.top_k} by AUC):")
        _report_original(res)
        for k, p in res["paths"].items():
            print(f"  {k}: {p}")

    if a.variant in ("both", "compact"):
        res = concept_model_figure(top_k=min(a.top_k, 12), split=a.split, out_folder=a.out_folder)
        print("compact variant:")
        _report_compact(res, a.top_k)
        for k, p in res["paths"].items():
            print(f"  {k}: {p}")
        print(f"  all_concepts: {res['all_concepts_csv']}")


if __name__ == "__main__":
    main()
