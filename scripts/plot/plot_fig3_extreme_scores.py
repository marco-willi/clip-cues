#!/usr/bin/env python
"""Fig 3 — extreme canonical detector-score montages (CNNSpot, SynthCLIC).

Logic in ``src/clip_cues_research/figures/extreme_scores.py``. Ranks come from F5's checksummed
``ranking/f5_*`` artifacts; images are decoded from the HF cache. Needs ``HF_HOME=data/hf_cache``.

    HF_HOME=data/hf_cache uv run python scripts/plot/plot_fig3_extreme_scores.py
"""

from __future__ import annotations

import argparse

from clip_cues_research.figures.extreme_scores import caption_facts, extreme_scores_figure


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-k", type=int, default=5, help="images per pole per dataset")
    ap.add_argument("--out-folder", default="fig3-extreme-scores")
    a = ap.parse_args()

    res = extreme_scores_figure(k=a.k, out_folder=a.out_folder)
    for ds, sub in res["table"].groupby("dataset"):
        lo, hi = sub["logit"].min(), sub["logit"].max()
        n_bad = int(
            ((sub["pole"] == "real_like") & (sub["label"] == 1)).sum()
            + ((sub["pole"] == "synthetic_like") & (sub["label"] == 0)).sum()
        )
        print(f"  {ds:10s} shown logits [{lo:+.2f}, {hi:+.2f}]  off-label images: {n_bad}")
    print("caption facts (from F5 summary.json):")
    for ds, f in caption_facts().items():
        print(
            f"  {ds:10s} n={f['n_test']:6d}  seed overlap top50 {f['seed_overlap_top50']:.2f} "
            f"/ bottom50 {f['seed_overlap_bottom50']:.2f}"
        )
    print("saved:")
    for k, p in res["paths"].items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
