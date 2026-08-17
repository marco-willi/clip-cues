#!/usr/bin/env python
"""Figures 2 and 3 — SynthBuster+ and SynthCLIC corpus example collages, as PDF.

Rebuilt from the published selection, not redrawn: the four `image_id`s per figure are pinned from
the archived notebook that made the submitted PNGs (`archive/detection-via-clip/notebooks/
51-mw-visualize-datasets.ipynb`, cells 30 and 40). Only the output format changes — PNG at 100 dpi
becomes PDF, so the column headings are vector text like the rest of the figure set. Logic in
``src/clip_cues_research/figures/corpus_examples.py``.

Needs the HF image cache (the `make` target sets `HF_HOME`).

    HF_HOME=data/hf_cache uv run python scripts/plot/plot_fig2_corpus_examples.py
    HF_HOME=data/hf_cache uv run python scripts/plot/plot_fig2_corpus_examples.py --corpus synthclic
"""

from __future__ import annotations

import argparse

from clip_cues_research.figures.corpus_examples import SPECS, corpus_examples_figure


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--corpus", nargs="+", default=sorted(SPECS), choices=sorted(SPECS), help="which collages"
    )
    ap.add_argument("--out-folder", default="fig2-corpus-examples")
    ap.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="resolution of the embedded photographs; the labels are vector regardless. "
        "150 puts ~550 ppi on the page at the widths the manuscript uses, comfortably above the "
        "300 ppi print standard (the submitted PNGs were 100). Raising it further only grows the "
        "file.",
    )
    a = ap.parse_args()

    for name in a.corpus:
        res = corpus_examples_figure(name, out_folder=a.out_folder, dpi=a.dpi)
        t = res["table"]
        print(
            f"  {name}: {t['grid_row'].nunique()} x {t['grid_col'].nunique()} cells, "
            f"{t['image_id'].nunique()} source photos"
        )
        for k, p in res["paths"].items():
            print(f"    {k}: {p}")


if __name__ == "__main__":
    main()
