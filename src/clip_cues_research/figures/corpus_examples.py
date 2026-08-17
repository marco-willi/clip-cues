"""Figures 2 and 3 — the corpus example collages for SynthBuster+ and SynthCLIC.

These two figures were never rebuilt for the revision: their content is unchanged from the initial
submission, and the figure ledger records them as out of scope. This module exists for one reason —
to emit them as **PDF** rather than the submitted 100-dpi PNG, so that the column headings are
vector text like every other figure in the paper. The photographs stay raster; nothing else about
the figures changes.

**Provenance.** The originals came from `archive/detection-via-clip/notebooks/51-mw-visualize-
datasets.ipynb` (cells 30 and 40, `synthbusterplus_samples.png` / `synthclic_samples.png`). That
notebook cannot run here: it reads `$DATA_PATH/datasets/{SynthbusterPlus,CLIC}/raw/` through a
metadata parquet that is not in this repo. It is still the source of truth for *what the figures
contain*, and it does not have to be re-run to establish that — the executed notebook stores both
the four sampled `image_id`s per figure (its `rng.choice(..., seed=123)` draw) and the rendered
images. Those ids are pinned in :data:`SPECS` below, so this module reproduces the published
selection exactly instead of redrawing it against a different row order. Column order and display
names come from the same notebook's `reproduction/config/plotting.yaml`.

Verified by rendering against the notebook's stored output: same rows, same columns, same images.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from clip_cues.visualization import plot_collage
from clip_cues_research.figures.style import FIGURES_ROOT, apply_style


@dataclass(frozen=True)
class Spec:
    """Everything that fixes one collage: which images, which columns, how big."""

    hf_name: str
    title: str
    #: The `rng.choice(image_id.unique(), 4, seed=123)` draw recorded in the archived notebook.
    #: One row per id, in this order.
    image_ids: tuple[str, ...]
    #: Column order, source id -> display name (`reproduction/config/plotting.yaml` of the archived repo).
    sources: dict[str, str]
    figsize: tuple[float, float]


SPECS: dict[str, Spec] = {
    "synthbuster-plus": Spec(
        hf_name="marco-willi/synthbuster-plus",
        title="SynthBuster+",
        image_ids=("r056f233bt", "r1986e42bt", "r0f36ba1bt", "r1b106abdt"),
        sources={
            "raise1k": "Raise1k",
            "dalle2": "DALLE2",
            "dalle3": "DALLE3",
            "firefly": "Firefly",
            "glide": "Glide",
            "midjourney-v5": "MJv5",
            "stable-diffusion-1-3": "SD1.3",
            "stable-diffusion-1-4": "SD1.4",
            "stable-diffusion-2": "SD2",
            "stable-diffusion-xl": "SDXL",
            "imagen3": "Imagen3",
            "SD3-medium": "SD3M",
            "FLUX.1-schnell": "FluxSchnell",
            "FLUX.1-dev": "FluxDev",
        },
        figsize=(24, 8),
    ),
    "synthclic": Spec(
        hf_name="marco-willi/synthclic",
        title="SynthCLIC",
        image_ids=(
            "IMG_20170718_191130",
            "IMG_20170114_195650",
            "alex-wong-17997",
            "claudio-testa-135408",
        ),
        sources={
            "clic2020": "CLIC",
            "imagen3": "Imagen3",
            "SD3-medium": "SD3M",
            "FLUX.1-dev": "FLUXDev",
            "FLUX.1-schnell": "FLUXSchnell",
        },
        figsize=(12, 8),
    ),
}


def _row_index(spec: Spec) -> tuple[pd.DataFrame, object]:
    """Locate every (image_id, source) cell of the collage, and return it with the dataset.

    All splits are searched: the notebook sampled from the full metadata parquet, so a pinned id
    can sit in any of train/validation/test.
    """
    from datasets import concatenate_datasets, load_dataset

    ds = load_dataset(spec.hf_name)
    full = concatenate_datasets([ds[k] for k in sorted(ds)])
    meta = full.select_columns(["image_id", "source"]).to_pandas().reset_index(names="row")

    cells = []
    for r, image_id in enumerate(spec.image_ids):
        for c, source in enumerate(spec.sources):
            hit = meta[(meta["image_id"] == image_id) & (meta["source"] == source)]
            if len(hit) != 1:
                raise ValueError(
                    f"{spec.hf_name}: expected exactly one {source!r} image for {image_id!r}, "
                    f"found {len(hit)} — the pinned selection no longer resolves"
                )
            cells.append(
                {
                    "grid_row": r,
                    "grid_col": c,
                    "image_id": image_id,
                    "source": source,
                    "col_label": spec.sources[source],
                    "row": int(hit["row"].iloc[0]),
                }
            )
    return pd.DataFrame(cells), full


def corpus_examples_figure(
    name: str,
    out_folder: str = "fig2-corpus-examples",
    dpi: int = 150,
    root: Path | None = None,
) -> dict:
    """Build one corpus collage and write ``<name>-examples.pdf`` plus its cell manifest.

    PDF only, by design: this figure is rebuilt *because* the submitted version was a PNG. The CSV
    next to it lists the exact image behind every cell, so the figure can be audited without
    rerunning it.

    ``dpi`` sets the resolution of the embedded photographs only -- the headings are vector at any
    value. These figures are drawn far wider than they are printed (24 in and 12 in of figure into a
    6.4 in `cas-sc` text block), so the on-page resolution is ~3.7x the dpi: the default 150 lands
    ~550 ppi, comfortably above the 300 ppi print standard. Size scales steeply with it -- 100 dpi
    gives a 2 MB PDF, 150 about 4 MB, 300 about 17 MB for resolution the page cannot show.
    """
    if name not in SPECS:
        raise KeyError(f"unknown corpus {name!r}; expected one of {sorted(SPECS)}")
    spec = SPECS[name]
    apply_style()

    cells, full = _row_index(spec)
    images = [full[int(r)]["image"].convert("RGB") for r in cells["row"]]
    cells = cells.assign(
        width=[im.size[0] for im in images], height=[im.size[1] for im in images]
    ).drop(columns="row")  # the HF row index is a cache detail, not provenance

    fig, _ = plot_collage(
        images=images,
        col_labels=list(spec.sources.values()),
        nrows=len(spec.image_ids),
        ncols=len(spec.sources),
        figsize=spec.figsize,
        title=spec.title,
        # The published figures let each photograph keep its own shape — the corpora mix portrait,
        # landscape and square, and a stretched panel would misrepresent the generator's output.
        # `plot_collage` defaults to "auto", which fills the cell.
        aspect="equal",
    )
    plt.tight_layout()

    out = (root or FIGURES_ROOT) / out_folder
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / f"{name}-examples.pdf"
    fig.savefig(pdf, dpi=dpi, bbox_inches="tight")
    csv = out / f"{name}-examples.csv"
    cells.to_csv(csv, index=False)
    plt.close(fig)
    return {"paths": {"pdf": pdf, "csv": csv}, "table": cells}
