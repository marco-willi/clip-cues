#!/usr/bin/env python
"""Appendix figures — Fig 8 (CNNSpot examples) and Fig 9 (CLIP-IQA perceptual axes).

Both land in ``reproduction/experiments/figures/appendix/``. Fig 8 needs the HF image cache; Fig 9 replots
precomputed E8 tables from ``outputs/e8/clipiqa/`` and refits only the reference full-detector AUROC.

    HF_HOME=data/hf_cache uv run python scripts/plot/plot_appendix_figures.py
    HF_HOME=data/hf_cache uv run python scripts/plot/plot_appendix_figures.py --only fig9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from clip_cues_research.figures.clipiqa import (
    clipiqa_distribution_figure,
    clipiqa_figure,
    load_clipiqa,
    reference_aurocs,
)
from clip_cues_research.figures.dataset_examples import cnnspot_examples_figure, logit_spread
from clip_cues_research.figures.style import FIGURES_ROOT

OUT = FIGURES_ROOT / "appendix"


def _write_caption(name: str, text: str) -> Path:
    """Captions are generated, never hand-written, so their numbers cannot drift from the figure."""
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}-caption.txt"
    p.write_text(text.strip() + "\n")
    return p


def build_fig8() -> None:
    res = cnnspot_examples_figure(out_folder="appendix")
    spread = logit_spread()
    print("  Fig 8 — CNNSpot examples (median-logit image per group):")
    for r in res["table"].itertuples():
        cls = "real" if r.label == 0 else "synthetic"
        print(
            f"    {r.source:16s} {cls:10s} z={r.logit:+.2f} (group median {r.group_median_logit:+.2f},"
            f" n={r.n_in_group})"
        )
    p = OUT / "fig8-cnnspot-examples-logit-spread.csv"
    spread.to_csv(p, index=False)
    for k, v in res["paths"].items():
        print(f"    {k}: {v}")
    print(f"    spread: {p}")

    t = res["table"]
    n_groups = t["source"].nunique()
    origins = t.loc[t["label"] == 0, "origin"]
    # "<x> subset" is the fallback for groups whose real corpus the benchmark never documented; it
    # must not be listed as though it were a corpus name.
    corpora = sorted({o.split(" / ")[0] for o in origins if not o.endswith("subset")})
    n_unknown = int(sum(o.endswith("subset") for o in origins))
    cap = _write_caption(
        "fig8-cnnspot-examples",
        f"Example images from the CNNSpot test set, one column per generator group "
        f"({n_groups} of the 23 groups shown). Each column heading names the generator that produced "
        "the image in the Synthetic row; it does NOT describe the Real row. CNNSpot files each real "
        "photograph under the generator group it is paired into, so the real images come from "
        "whatever corpus the benchmark drew them from - a different one per group, annotated beneath "
        f"each panel ({', '.join(corpora)}). For {n_unknown} of the groups shown the source corpus is "
        "not documented in the benchmark description, so those panels are annotated with the subset "
        "directory the image sits in rather than a corpus name. "
        "Within each group the displayed pair is the real and the "
        "synthetic image whose canonical detector logit is closest to that group's median, so the "
        "panels are typical rather than extreme; the selection is reproducible from the "
        "accompanying CSV. The visual contrast with SynthCLIC - low-resolution GAN-era crops, heavy "
        "processing, class-centric content - is why the two training distributions induce nearly "
        "orthogonal decision boundaries.",
    )
    print(f"    caption: {cap}")


def build_fig9(datasets: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # (a) the published variant: per-attribute degree distributions.
    # Deliberately NOT restricted to `datasets`: that list exists for (b), which replots precomputed
    # `outputs/e8/clipiqa/` tables that were only ever produced for SynthCLIC and CNNSpot. (a) is
    # computed from the snapshot, so it can and should include SynthBuster+.
    dist = clipiqa_distribution_figure(out_folder="appendix")
    print("  Fig 9a — CLIP-IQA attribute distributions:")
    for k, v in dist["paths"].items():
        print(f"    {k}: {v}")
    d = dist["table"]
    n_attr = len(dist["attributes"])
    per_ds = "; ".join(
        f"{g['dataset'].iloc[0]} ({g['split'].iloc[0]})"
        for _, g in d.groupby("dataset", sort=False)
    )
    cap = _write_caption(
        "fig9-clipiqa-distributions",
        f"Distribution of the {n_attr} CLIP-IQA perceptual attributes across the three corpora "
        f"[{per_ds}], by image source. CLIP-IQA scores an attribute as a softmax between two antonym "
        "prompts at CLIP's own logit scale - degree = softmax(100 * [cos(e, t+), cos(e, t-)])[0], "
        "e.g. 'Sharp photo.' against 'Blurry photo.' - so a degree near 1 means the positive prompt "
        "describes the image and 0.5 means neither does. Every axis is therefore a CLIP text-prompt "
        "PERCEPT rather than a pixel measurement: 'noisiness' is how noisy an image looks, not its "
        "measured noise. Boxes show quartiles with outliers suppressed; the real source is the "
        "left-most box of every group. SynthCLIC and SynthBuster+ are broken out per generator, and "
        "SynthBuster+ is restricted to the four generators it shares with SynthCLIC (Imagen 3, "
        "FLUX.1-dev, FLUX.1-schnell, SD3-medium) so the same generator can be compared across the "
        "two corpora; its nine pre-existing generators are omitted. CNNSpot is shown as real against "
        "synthetic because its `source` column names the evaluation group rather than the "
        "provenance - on its training split that column reads 'progan' for both classes, the real "
        "half being LSUN photography paired into the ProGAN group. All three corpora are characterised on their training splits: the figure is "
        "descriptive rather than an evaluation, and a common split makes the panels comparable "
        "(SynthBuster+'s test split is held out in any case). Class sizes are given per panel and "
        "are unbalanced, so a box's width carries no information about how much data supports it. "
        "The figure characterises how the corpora differ in perceived quality, independently of any "
        "detector.",
    )
    print(f"    caption: {cap}")

    # (b) the per-axis separability variant
    res = clipiqa_figure(datasets, out_dir=OUT, stem="fig9-clipiqa-axes")
    # clipiqa_figure saves its own PNG/PDF; export the source table alongside for the same contract
    # every other figure follows.
    table = pd.concat(
        [load_clipiqa("outputs/e8/clipiqa", ds).assign(dataset=ds) for ds in datasets]
    )
    p = OUT / "fig9-clipiqa-axes.csv"
    table.to_csv(p, index=False)
    print("  Fig 9b — CLIP-IQA per-axis separability:")
    for f in res.get("saved", []):
        print(f"    {f}")
    print(f"    csv: {p}")
    refs = {ds: reference_aurocs(ds) for ds in datasets}
    ref_txt = "; ".join(
        f"{ds}: eight axes combined {r['combined_8']:.2f} versus the full CLIP detector "
        f"{r['full_detector']:.2f}"
        for ds, r in refs.items()
        if r.get("combined_8") is not None and r.get("full_detector") is not None
    )
    cap = _write_caption(
        "fig9-clipiqa-axes",
        "How well each CLIP-IQA perceptual axis separates real from synthetic on its own, as a "
        "single-feature classifier AUROC (sign-optimal, so always at or above 0.5). Colour gives the "
        "direction: orange where synthetic images score higher on the axis, blue where they score "
        "lower. Dashed lines mark two references - all eight axes combined, and the full CLIP "
        f"detector on the same split ({ref_txt}). Each axis alone is a weak detector for diffusion "
        "imagery but a strong one for GANs, and the gap to the full detector is the diffuse residual "
        "that no small set of named perceptual axes recovers. As above, these axes are text-prompt "
        "percepts, not pixel measurements.",
    )
    print(f"    caption: {cap}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["fig8", "fig9"], default=None)
    ap.add_argument("--datasets", nargs="+", default=["synthclic", "cnnspot"])
    a = ap.parse_args()

    if a.only in (None, "fig8"):
        build_fig8()
    if a.only in (None, "fig9"):
        if not Path("outputs/e8/clipiqa/synthclic_clipiqa.csv").exists():
            raise SystemExit("outputs/e8/clipiqa/ missing — run scripts/analyze/analyze_clipiqa.py")
        build_fig9(a.datasets)


if __name__ == "__main__":
    main()
