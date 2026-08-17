"""Appendix Fig 8 — CNNSpot example images.

CNNSpot looks nothing like SynthCLIC, and that visual difference is doing real explanatory work in
the paper: it is why the two training distributions induce near-orthogonal boundaries (Fig 6), and
why a detector trained on one transfers so poorly to the other. The figure makes that concrete --
low-resolution GAN-era crops, heavy processing, class-centric content -- next to SynthCLIC's
full-resolution photographic pairs.

**Which images.** Not the extremes, and not an arbitrary first-N: for each generator group the real
and synthetic image whose canonical logit is closest to that group's **median** are shown, using
F5's checksummed ranking (``ranking/f5_cnnspot``). So every panel is a typical image for its group
under the same detector Fig 3 uses at the poles, and the selection rule is reproducible from the CSV.

**What a column heading means.** The heading names a generator group within the CNNSpot **test set**,
not the origin of both rows. Only the bottom row was produced by the named generator; the top row is that subset's
paired real photographs, which come from whatever corpus the benchmark drew them from (a different
one per group). Reading the heading as applying to the real row is the error this figure has to
avoid, so real panels are captioned from their own path -- see
``extreme_scores.cnnspot_real_origin``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clip_cues_research.figures.extreme_scores import _square, cnnspot_real_origin
from clip_cues_research.figures.style import apply_style, save_figure
from clip_cues_research.finalexp import data as D

HF_NAME = "marco-willi/cnnspot-small"
SPLIT = "test"
#: Generator groups spanning CNNSpot's regimes: GAN (ProGAN/StyleGAN2/BigGAN/CycleGAN), conditional
#: image synthesis (GauGAN/CRN), face manipulation (deepfake/WhichFaceIsReal), diffusion (LDM/GLIDE).
DEFAULT_SOURCES = (
    "progan",
    "stylegan2",
    "biggan",
    "cyclegan",
    "gaugan",
    "deepfake",
    "whichfaceisreal",
    "ldm_200",
)
PRETTY = {
    "progan": "ProGAN",
    "stylegan": "StyleGAN",
    "stylegan2": "StyleGAN2",
    "biggan": "BigGAN",
    "cyclegan": "CycleGAN",
    "gaugan": "GauGAN",
    "crn": "CRN",
    "imle": "IMLE",
    "deepfake": "Deepfake",
    "whichfaceisreal": "WhichFaceIsReal",
    "stargan": "StarGAN",
    "ldm_200": "LDM-200",
    "glide_100_10": "GLIDE",
    "guided": "Guided diff.",
    "dalle": "DALL-E",
}


def median_examples(sources: tuple[str, ...] = DEFAULT_SOURCES) -> pd.DataFrame:
    """One real and one synthetic image per source: the median-logit image of each (source, label)."""
    df = pd.read_csv(D.resolve("ranking/f5_cnnspot"))
    picks = []
    for src in sources:
        for label in (0, 1):
            sub = df[(df["source"] == src) & (df["label"] == label)]
            if sub.empty:
                raise ValueError(f"no label={label} images for source {src!r} in the F5 ranking")
            med = sub["logit"].median()
            row = sub.iloc[(sub["logit"] - med).abs().to_numpy().argmin()]
            picks.append(
                {
                    "source": src,
                    "label": int(label),
                    "image_id": row["image_id"],
                    "logit": float(row["logit"]),
                    "group_median_logit": float(med),
                    "n_in_group": int(len(sub)),
                }
            )
    return pd.DataFrame(picks)


def cnnspot_examples_figure(
    sources: tuple[str, ...] = DEFAULT_SOURCES,
    out_folder: str = "appendix",
) -> dict:
    """Build appendix Fig 8: real (top) over synthetic (bottom), one column per generator group."""
    from datasets import load_dataset

    apply_style()
    picks = median_examples(sources)

    ds = load_dataset(HF_NAME)[SPLIT]
    meta = ds.select_columns(["image_id"]).to_pandas()
    row_of = {str(i): k for k, i in enumerate(meta["image_id"])}

    picks["origin"] = [
        cnnspot_real_origin(r.image_id, r.source)
        if r.label == 0
        else PRETTY.get(r.source, r.source)
        for r in picks.itertuples()
    ]

    ncol = len(sources)
    fig, axes = plt.subplots(2, ncol, figsize=(1.55 * ncol, 3.75))
    for c, src in enumerate(sources):
        for r, label in enumerate((0, 1)):
            p = picks[(picks["source"] == src) & (picks["label"] == label)].iloc[0]
            ax = axes[r, c]
            _ = ax.imshow(_square(ds[row_of[str(p["image_id"])]]["image"].convert("RGB"), 256))
            _ = ax.set_xticks([])
            _ = ax.set_yticks([])
            _ = ax.grid(False)
            if r == 0:
                _ = ax.set_title(PRETTY.get(src, src), fontsize=8)
                # The heading above names the generator, which did NOT make this image — caption the
                # real panel with what its own path proves instead.
                _ = ax.set_xlabel(p["origin"], fontsize=6.5, color="0.35")
            if c == 0:
                _ = ax.set_ylabel(
                    "Real" if label == 0 else "Synthetic", fontsize=9.5, fontweight="bold"
                )
    _ = fig.subplots_adjust(wspace=0.06, hspace=0.16)

    paths = save_figure(fig, out_folder, "fig8-cnnspot-examples", table=picks)
    plt.close(fig)
    return {"paths": paths, "table": picks}


def logit_spread(sources: tuple[str, ...] = DEFAULT_SOURCES) -> pd.DataFrame:
    """Per-group logit quartiles — context for how typical the chosen panels are."""
    df = pd.read_csv(D.resolve("ranking/f5_cnnspot"))
    sub = df[df["source"].isin(sources)]
    return (
        sub.groupby(["source", "label"])["logit"]
        .agg(
            n="size",
            q25=lambda s: np.percentile(s, 25),
            median="median",
            q75=lambda s: np.percentile(s, 75),
        )
        .reset_index()
    )
