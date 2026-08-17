"""Fig 3 — extreme canonical detector scores.

What does the detector the paper actually reports consider most real-like and most synthetic-like?
Images are ranked by the **canonical logit** of the matched probe,

    z_h(x) = w_h . h(x) + b_h,

so the ordering is the projection onto the single decision direction (``b_h`` is constant). This
replaces the previous manuscript's k=8 per-axis montages, which F2 shows are not a reproducible
object: individual axes agree across seeds at Sigma-cos **0.289**.

Nothing is recomputed here. The ranking is F5's checksummed artifact (``ranking/f5_*`` in the
snapshot manifest); this module joins ``(source, image_id)`` to the HF rows and lays out the poles.

**Two things the caption must carry.**

1. *Stability.* The direction is stable (F1: Sigma-cos 0.989, cue profile rho 0.991) but the extreme
   *image sets* are much less so -- top-50 seed overlap **0.73** (SynthCLIC) and **0.66** (CNNSpot).
   The montage illustrates a stable direction; it is not a claim about these particular images.
2. *CNNSpot's ``source`` column is an evaluation subset, not a provenance.* CNNSpot files each real
   photograph under the generator group it is paired into, so ``source == "progan"`` on a real image
   means "the real half of the ProGAN evaluation subset" -- **not** that ProGAN produced it.
   Annotating such a panel "real / ProGAN" is simply wrong. Real CNNSpot panels are instead labelled
   from ``cnnspot_real_source_map`` in ``reproduction/config/mappings.yaml`` (the corpus the benchmark actually
   drew them from) plus the subsplit in their own path -- see ``cnnspot_real_origin``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from clip_cues_research.figures.style import apply_style, save_caption, save_figure, title_case
from clip_cues_research.finalexp import data as D

HF_NAME = {"synthclic": "marco-willi/synthclic", "cnnspot": "marco-willi/cnnspot-small"}
RANKING = {"synthclic": "ranking/f5_synthclic", "cnnspot": "ranking/f5_cnnspot"}
PANEL_LABEL = {"cnnspot": "A  CNNSpot", "synthclic": "B  SynthCLIC"}
SPLIT = "test"

# Mirrors reproduction/config/mappings.yaml for the sources that actually reach the poles; anything unmapped is
# shown verbatim rather than silently renamed.
PRETTY_SOURCE = {
    "clic2020": "CLIC2020",
    "FLUX.1-dev": "FLUX.1-dev",
    "FLUX.1-schnell": "FLUX.1-schnell",
    "SD3-medium": "SD3-medium",
    "imagen3": "Imagen 3",
    "progan": "ProGAN",
    "stylegan": "StyleGAN",
    "stylegan2": "StyleGAN2",
    "gaugan": "GauGAN",
    "biggan": "BigGAN",
    "whichfaceisreal": "WhichFaceIsReal",
}

MAPPINGS = Path("reproduction/config/mappings.yaml")


@lru_cache(maxsize=1)
def real_source_map(path: Path = MAPPINGS) -> dict[str, str]:
    """``{group -> corpus}`` for CNNSpot **real** photographs, from ``reproduction/config/mappings.yaml``.

    Lives in config rather than here because it is a property of the dataset, not of this figure --
    the same distinction any other consumer needs. See that file for provenance (Wang et al. 2020
    appendix B.1) and for which groups are deliberately absent. A missing group is not an error: the
    caller falls back to the subsplit directory, which the path itself proves.
    """
    return _mapping("cnnspot_real_source_map", path)


@lru_cache(maxsize=8)
def _mapping(key: str, path: Path = MAPPINGS) -> dict[str, str]:
    import yaml

    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()).get(key) or {}


def _group_label(source: str) -> str:
    """Display name for a CNNSpot group, preferring config over this module's short list.

    ``PRETTY_SOURCE`` only covers the groups that reach Fig 3's poles; without this fallback the
    others render as raw ids (``ldm_200 subset`` rather than ``LDM-200 subset``).
    """
    return PRETTY_SOURCE.get(source) or _mapping("cnnspot_source_map").get(source, source)


def cnnspot_real_origin(image_id: str, source: str) -> str:
    """Best *verifiable* description of where a CNNSpot real photograph came from.

    Groups differ in path depth: ``progan``/``stylegan``/``stylegan2``/``cyclegan`` store
    ``<group>/<subsplit>/0_real/<file>`` (subsplit = the object category, e.g. ``church``), everything
    else stores ``<group>/0_real/<file>`` and carries no subsplit at all. So the label is the corpus
    from ``cnnspot_real_source_map`` plus the subsplit where both exist, and degrades gracefully:
    corpus alone, subsplit alone, or the bare group explicitly marked as a *subset*.
    """
    corpus = real_source_map().get(source)
    parts = str(image_id).split("/")
    subsplit = parts[1] if len(parts) == 4 else None
    if corpus and subsplit:
        return f"{corpus} / {subsplit}"
    if corpus:
        return corpus
    if subsplit:
        return f"{subsplit} subset"
    return f"{_group_label(source)} subset"


def annotate(dataset: str, row) -> tuple[str, str]:
    """``(class, origin)`` for one panel — the two lines under an image.

    SynthCLIC's ``source`` *is* a provenance for both classes (``clic2020`` really is where its real
    photographs come from), so it is used as-is. CNNSpot's is not, for real images only.
    """
    cls = "real" if row["label"] == 0 else "synthetic"
    if dataset == "cnnspot" and row["label"] == 0:
        return cls, cnnspot_real_origin(row["image_id"], row["source"])
    return cls, PRETTY_SOURCE.get(row["source"], row["source"])


def poles(dataset: str, k: int = 5) -> pd.DataFrame:
    """The ``k`` lowest- and ``k`` highest-logit test images, ordered left-to-right by logit.

    Returns one frame with a ``pole`` column (``real_like`` / ``synthetic_like``) and the rank,
    source, label and logit carried straight from F5's ranking.
    """
    df = pd.read_csv(D.resolve(RANKING[dataset]))
    df = df.sort_values("logit").reset_index(drop=True)
    low = df.head(k).assign(pole="real_like")
    high = df.tail(k).assign(pole="synthetic_like")
    out = pd.concat([low, high], ignore_index=True)
    out["dataset"] = dataset
    return out


def _square(img: Image.Image, size: int = 320) -> Image.Image:
    """Centre-crop to square then resize -- every panel gets an identical bounding box."""
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return img.resize((size, size), Image.LANCZOS)


def _load_images(dataset: str, sel: pd.DataFrame, size: int = 320) -> list[Image.Image]:
    """Decode only the ~10 selected rows, keyed on ``(source, image_id)``.

    SynthCLIC reuses one ``image_id`` across its real photo and every synthetic counterpart, so the
    key must include ``source``; CNNSpot's ids are paths and are unique on their own.
    """
    from datasets import load_dataset

    ds = load_dataset(HF_NAME[dataset])[SPLIT]
    meta = ds.select_columns(["source", "image_id"]).to_pandas()
    row_of = {(s, i): k for k, (s, i) in enumerate(zip(meta["source"], meta["image_id"]))}

    imgs = []
    for _, r in sel.iterrows():
        key = (r["source"], r["image_id"])
        if key not in row_of:
            raise KeyError(f"{key} not found in {HF_NAME[dataset]}/{SPLIT}")
        imgs.append(_square(ds[row_of[key]]["image"].convert("RGB"), size))
    return imgs


def extreme_scores_figure(
    datasets: tuple[str, ...] = ("cnnspot", "synthclic"),
    k: int = 5,
    out_folder: str = "fig3-extreme-scores",
) -> dict:
    """Build Fig 3: one row of 2k montage panels per dataset, poles separated by white space."""
    apply_style()
    tables, images = {}, {}
    for ds in datasets:
        t = poles(ds, k)
        # Carry the rendered annotation into the exported CSV, so the figure's claim about each
        # image's provenance is auditable rather than living only in the PNG.
        ann = [annotate(ds, r) for _, r in t.iterrows()]
        t["class"] = [a[0] for a in ann]
        t["origin"] = [a[1] for a in ann]
        tables[ds] = t
        images[ds] = _load_images(ds, t)

    ncol = 2 * k + 1  # a spacer column between the two poles
    fig = plt.figure(figsize=(1.42 * ncol, 2.05 * len(datasets)))
    gs = fig.add_gridspec(
        len(datasets),
        ncol,
        width_ratios=[1.0] * k + [0.22] + [1.0] * k,
        wspace=0.08,
        hspace=0.50,
    )

    for r, ds in enumerate(datasets):
        sel, imgs = tables[ds], images[ds]
        row_axes = []
        for c in range(2 * k):
            col = c if c < k else c + 1  # skip the spacer column
            ax = fig.add_subplot(gs[r, col])
            row_axes.append(ax)
            _ = ax.imshow(imgs[c])
            _ = ax.set_xticks([])
            _ = ax.set_yticks([])
            _ = ax.grid(False)
            row = sel.iloc[c]
            cls, origin = annotate(ds, row)
            # Three lines, not "class / source" on one: FLUX.1-schnell and WhichFaceIsReal are
            # wider than a panel and collide with their neighbours.
            _ = ax.set_xlabel(
                f"$z$ = {row['logit']:+.2f}\n{cls}\n{origin}", fontsize=6.8, linespacing=1.45
            )
            if c == 0:
                _ = ax.set_ylabel(PANEL_LABEL.get(ds, ds), fontsize=9, fontweight="bold")

        # Pole headings over the first panel of each half.
        _ = row_axes[0].set_title(
            title_case("most real-like"), fontsize=8, loc="left", fontweight="normal"
        )
        _ = row_axes[k].set_title(
            title_case("most synthetic-like"), fontsize=8, loc="left", fontweight="normal"
        )

    table = pd.concat(tables.values(), ignore_index=True)
    paths = save_figure(fig, out_folder, "fig3-extreme-scores", table=table)
    plt.close(fig)

    facts = caption_facts()
    paths["caption"] = save_caption(
        out_folder,
        "fig3-extreme-scores",
        f"""
        Test-set images with the lowest and highest logits of the canonical CLIP detector,
        z = w.h + b, for CNNSpot (A) and SynthCLIC (B); {k} per pole. Because the bias is constant,
        ranking by z is ranking by projection onto the detector's single decision direction. Each
        panel gives the raw logit, the class, and the image's origin. **The direction is far more
        stable than these particular images:** across five seeds the decision direction agrees at
        Sigma-cosine 0.989 and its cue profile at rho 0.991, but the top-50 image sets agree at only
        {facts["synthclic"]["seed_overlap_top50"]:.2f} (SynthCLIC) and
        {facts["cnnspot"]["seed_overlap_top50"]:.2f} (CNNSpot) - so the montage illustrates a stable
        direction rather than making a claim about these images. CNNSpot files each real photograph
        under the generator group it is paired into, so its real panels are annotated with the corpus
        the benchmark actually drew them from (LSUN, with the object subsplit) rather than the group
        name; the ProGAN and BigGAN reals were additionally centre-cropped and resized to 256x256 by
        the benchmark authors, so they are preprocessed crops rather than raw photographs.
    """,
    )
    return {"paths": paths, "table": table}


def caption_facts(out_folder: str = "fig3-extreme-scores") -> dict:
    """Numbers the caption needs, read from F5's summary rather than retyped."""
    import json

    p = Path("reproduction/experiments/final_consolidation/F5-canonical-montage/artifacts/summary.json")
    s = json.loads(p.read_text())
    return {
        ds: {
            "n_test": v["n_test"],
            "logit_range": v["logit_range"],
            "seed_overlap_top50": v["seed_overlap_mean_top"],
            "seed_overlap_bottom50": v["seed_overlap_mean_bottom"],
        }
        for ds, v in s["by_dataset"].items()
    }
