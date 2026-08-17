#!/usr/bin/env python
"""Fig 6 — cross-dataset boundary difference (signed Delta decomposition, one panel).

Logic in ``src/clip_cues_research/figures/boundary_mechanism.py``. Reads F6's artifacts directly;
nothing is recomputed. The data-weighted (Sigma-metric) similarity is annotated on the figure with
the raw weight cosine beside it, because the two tell materially different stories on the same
normals and the choice should be visible. The full 3x3 matrix is printed here for the accompanying
table and exported in the figure's CSV.

    uv run python scripts/plot/plot_fig6_boundary.py
"""

from __future__ import annotations

import argparse

from clip_cues_research.figures.boundary_mechanism import boundary_delta_figure


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-n", type=int, default=14, help="Delta axes to show")
    ap.add_argument("--out-folder", default="fig6-boundary-delta")
    a = ap.parse_args()

    res = boundary_delta_figure(top_n=a.top_n, out_folder=a.out_folder)
    sig, raw = res["sigma"], res["raw"]
    print("  boundary similarity (Sigma-metric, raw in parentheses) — for the companion table:")
    for i, x in enumerate(sig.index):
        for y in sig.columns[i + 1 :]:
            print(f"    {x:18s} ~ {y:18s} {sig.loc[x, y]:+.3f}  ({raw.loc[x, y]:+.3f})")
    ax = res["axes"]
    print(f"  Delta axes shown: {len(ax)}")
    for side, sub in ax.groupby(ax["alpha_coef"] > 0):
        label = "CNNSpot-associated" if side else "SynthCLIC-associated"
        print(f"    {label}: {', '.join(sub['cue'].tolist())}")
    print("saved:")
    for k, p in res["paths"].items():
        print(f"  {k}: {p}")


if __name__ == "__main__":
    main()
