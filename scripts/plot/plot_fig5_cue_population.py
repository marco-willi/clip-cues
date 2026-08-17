#!/usr/bin/env python
"""Fig 5 — population-level cue interpretation (A association, B synthesis shift).

Logic in ``src/clip_cues_research/figures/cue_population.py``. Builds **both layouts** from a single
bootstrap so they are directly comparable, plus the full 168-cue table:

  independent  each panel ranks its own top-12 (PLAN_FIGURES_2 spec); shared cues drawn filled
  shared       one y-axis, union of the top-6 per estimand, so rows read across

Panel B plots the standardized effect size ``d_q``; the raw mean shift and its cluster-bootstrap CI
stay in ``fig5-all-cues.csv``. Everything is read from the checksummed canonical snapshot.

    uv run python scripts/plot/plot_fig5_cue_population.py
"""

from __future__ import annotations

import argparse

from clip_cues_research.figures.cue_population import both_layouts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-show", type=int, default=12, help="cues per panel (independent layout)")
    ap.add_argument("--n-each", type=int, default=6, help="cues per estimand (shared layout)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-folder", default="fig5-cue-population")
    a = ap.parse_args()

    res = both_layouts(n_show=a.n_show, n_each=a.n_each, n_boot=a.n_boot, out_folder=a.out_folder)
    m = res["merged"]
    print(
        f"  {len(m)} cues; {int(m['n_pairs'].iloc[0])} pairs over "
        f"{int(m['n_clusters'].iloc[0])} source photos x {int(m['n_generators'].iloc[0])} generators"
    )

    ind = res["independent"]["table"]
    both = sorted(ind.loc[ind["in_both_panels"], "cue"].unique())
    print(f"  in both top-{a.n_show} lists ({len(both)}): {', '.join(both) if both else '(none)'}")

    print("  A  top association:")
    for r in (
        ind[ind.panel == "A"]
        .reindex(ind[ind.panel == "A"]["within_class_r"].abs().sort_values(ascending=False).index)
        .head(5)
        .itertuples()
    ):
        print(f"     {r.cue:26s} r {r.within_class_r:+.3f}")
    print("  B  top synthesis shift:")
    for r in (
        ind[ind.panel == "B"]
        .reindex(ind[ind.panel == "B"]["d_q"].abs().sort_values(ascending=False).index)
        .head(5)
        .itertuples()
    ):
        sig = "" if (r.delta_ci_lo <= 0 <= r.delta_ci_hi) else " *"
        print(
            f"     {r.cue:26s} d {r.d_q:+.3f}   mean {r.delta:+.4f} "
            f"[{r.delta_ci_lo:+.4f}, {r.delta_ci_hi:+.4f}]{sig}"
        )

    print("saved:")
    for lay in ("independent", "shared"):
        for k, p in res[lay]["paths"].items():
            print(f"  {lay}/{k}: {p}")
    print(f"  all_cues: {res['all_cues_csv']}")
    print(f"  caption:  {res['caption_path']}")


if __name__ == "__main__":
    main()
