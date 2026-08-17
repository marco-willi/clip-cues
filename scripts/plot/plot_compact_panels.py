#!/usr/bin/env python
"""The information-restriction cascade (§8) and the stability summary (§9) — **tables**, not figures.

PLAN_FIGURES_2 makes both of these tables ("a compact table + prose is enough"; "no orthogonality
figure"). The LaTeX goes to ``reproduction/experiments/figures/tables/`` and *is* part of the manuscript; the
renders go to ``reproduction/experiments/figures/_retired/compact/`` and are not in ``make figures-all``. Both
come out of one builder, so the table cannot drift from the retired plot.

Logic in ``src/clip_cues_research/figures/compact_panels.py``. Every number is read from the
F-experiment ``summary.json`` files, so neither can drift from the results.

    uv run python scripts/plot/plot_compact_panels.py
"""

from __future__ import annotations

import argparse

import pandas as pd

from clip_cues_research.figures.compact_panels import (
    cascade_figure,
    latex_tables,
    stability_figure,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-folder", default="_retired/compact", help="where the retired renders go")
    ap.add_argument("--tex-folder", default="tables", help="where the manuscript LaTeX goes")
    ap.add_argument(
        "--tex-only",
        action="store_true",
        help="emit the LaTeX only. The renders are retired, and re-saving a PDF rewrites its "
        "embedded timestamp, so a table-only rebuild would otherwise show up as binary churn. "
        "This is what `make tables-compact` runs.",
    )
    a = ap.parse_args()

    tex = latex_tables(out_folder=a.tex_folder)
    if a.tex_only:
        print("saved:")
        for k, p in tex.items():
            print(f"  tex/{k}: {p}")
        return

    casc = cascade_figure(out_folder=a.out_folder)
    stab = stability_figure(out_folder=a.out_folder)

    print("  cascade:")
    for _, r in casc["table"].iterrows():
        # pandas stores the first row's absent delta as NaN, not None.
        d = (
            ""
            if pd.isna(r["delta"])
            else f"   {r['delta']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
        )
        print(f"    {r['stage']:24s} {r['auroc']:.3f}{d}")
    print("  stability:")
    for _, r in stab["table"].iterrows():
        print(
            f"    {r['representation']:26s} direction {r['direction']:.3f}  "
            f"cue profile {r['cue_profile']:.3f}"
        )
    print("saved:")
    for res in (casc, stab):
        for k, p in res["paths"].items():
            print(f"  {k}: {p}")
    for k, p in tex.items():
        print(f"  tex/{k}: {p}")


if __name__ == "__main__":
    main()
