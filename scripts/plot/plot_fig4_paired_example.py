#!/usr/bin/env python
"""Fig 4 — content-controlled SynthCLIC pairs and the named cues that move.

Logic in ``src/clip_cues_research/figures/paired_example.py``. Deltas come from the checksummed
snapshot (``cue_scores/synthclic__antonyms``, canonical space) -- the identical array behind Fig 5's
population panel -- so each example is provably an instance of the aggregate. Needs the HF image
cache for pixels.

Renders **several variants** on different source photos, each named for its content id
(``fig4-paired-example-<id8>``), so the figure can be chosen on how clearly it reads rather than
accepted sight unseen. Selection is by typicality (agreement of a pair's delta profile with the
population mean), so every candidate is a defensible choice, not a curated one.

    HF_HOME=data/hf_cache uv run python scripts/plot/plot_fig4_paired_example.py
    HF_HOME=data/hf_cache uv run python scripts/plot/plot_fig4_paired_example.py --layout row
    HF_HOME=data/hf_cache uv run python scripts/plot/plot_fig4_paired_example.py --image-ids <id> <id>
"""

from __future__ import annotations

import argparse

from clip_cues_research.figures.paired_example import paired_example_variants


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-n", type=int, default=10, help="variants to render (the ledger documents all of them)"
    )
    ap.add_argument("--image-ids", nargs="+", default=None, help="pin specific content ids")
    ap.add_argument(
        "--layout",
        default="row",
        choices=["row", "grid"],
        help="row = 1x4 images over 3 bar panels; grid = 2x2 cells (more room for cue names)",
    )
    ap.add_argument("--top-k", type=int, default=6, help="cues shown per synthetic image")
    ap.add_argument(
        "--label-style",
        default="phrase",
        choices=["phrase", "name"],
        help="phrase = the antonym phrase the synthetic moved toward; name = the cue id",
    )
    ap.add_argument("--out-folder", default="fig4-paired-example")
    a = ap.parse_args()

    res = paired_example_variants(
        n=a.n,
        image_ids=a.image_ids,
        layout=a.layout,
        top_k=a.top_k,
        label_style=a.label_style,
        out_folder=a.out_folder,
    )
    r = res["ranking"]
    print(
        f"  {len(res['variants'])} variants from {res['n_candidate_ids']} fully-paired ids "
        f"(typicality median {r['score'].median():.3f})"
    )
    for v in res["variants"]:
        print(
            f"\n  {v['image_id'][:8]}  typicality {v['typicality']:.3f}  "
            f"layout {v['layout']}  x-limit +-{v['shared_xlim']:.3f}"
        )
        for g, sub in v["table"].groupby("generator", sort=False):
            top = sub.reindex(sub["delta"].abs().sort_values(ascending=False).index).head(3)
            print(f"    {g:15s} " + ", ".join(f"{t.cue} {t.delta:+.3f}" for t in top.itertuples()))
        print(f"    -> {v['paths']['png']}")


if __name__ == "__main__":
    main()
