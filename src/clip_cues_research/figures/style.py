"""Shared style, cue-family taxonomy and saving helper for the revision figure set.

One definition of the things that must not drift between figures: the palette, the diverging
colormap Fig 4 and Fig 6 both use, the six-family cue taxonomy, and the save path convention.

Figures are written to ``reproduction/experiments/figures/<folder>/`` as PNG + PDF alongside the exact CSV they
were drawn from, so a reader can check any bar against a number.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
import pandas as pd
import seaborn as sns
import yaml

FIGURES_ROOT = Path("reproduction/experiments/figures")

# ── cue families ─────────────────────────────────────────────────────────────────────────────
# The project's six-family taxonomy, defined by mapping a vocabulary's `category` column. This is
# the single source of truth: figures and analyses both import it from here.
#
# Coverage differs by vocabulary and this matters when choosing one for a figure:
#   antonyms (168)           -> 5 of 7 families, 40% fall to `other`
#   optimized_v2_canon (128) -> 7 of 7 families, 20% `other`  (reproduces final_test/families.csv)
FAM = {
    "detail_floor": "detail_texture",
    "micro_detail": "detail_texture",
    "texture": "detail_texture",
    "technique": "detail_texture",
    "medium": "medium_style",
    "style": "medium_style",
    "idealization": "medium_style",
    "provenance": "provenance_process",
    "camera": "provenance_process",
    "specificity": "scene_specificity",
    "color": "tone_color",
    "clipiqa": "tone_color",
    "physics": "physics_optics",
    "optics": "physics_optics",
    "lense": "physics_optics",
}

FAMILY_ORDER = [
    "detail_texture",
    "physics_optics",
    "tone_color",
    "provenance_process",
    "medium_style",
    "scene_specificity",
    "other",
]

FAMILY_LABEL = {
    "detail_texture": "detail / texture",
    "physics_optics": "physics / optics",
    "tone_color": "tone / color",
    "provenance_process": "provenance / process",
    "medium_style": "medium / style",
    "scene_specificity": "scene",
    "other": "other",
}


def cue_families(vocab_csv: str | Path) -> pd.Series:
    """``{cue -> family}`` for a vocabulary CSV, unmapped categories falling to ``other``."""
    df = pd.read_csv(vocab_csv)
    return pd.Series(
        df["category"].map(lambda c: FAM.get(c, "other")).values,
        index=df["attribute_name"].values,
        name="family",
    )


# ── style ───────────────────────────────────────────────────────────────────────────────────
#: Presentation choices live in `reproduction/config/figures.yaml`, not here: they must not drift between
#: figures, and a reviewer changing the palette should not have to edit Python. This module is the
#: only reader — every figure imports its colours from here.
FIGURE_CONFIG = Path("reproduction/config/figures.yaml")


@lru_cache(maxsize=1)
def config(path: Path = FIGURE_CONFIG) -> dict:
    """The global figure configuration, or the built-in defaults if the file is absent."""
    if not path.exists():
        return _DEFAULTS
    loaded = yaml.safe_load(path.read_text()) or {}
    return {**_DEFAULTS, **loaded}


_DEFAULTS: dict = {
    "palette": {
        "real": "#1f77b4",
        "synthetic": "#ff7f0e",
        "positive": "#ff7f0e",
        "negative": "#1f77b4",
        "neutral": "#1f77b4",
        "strong": "#2ca02c",
        "weak": "#7f7f7f",
        "real_dark": "#08306b",
        "synthetic_dark": "#7f3b08",
    },
    "categorical": "tab10",
    "colormaps": {"diverging": "vlag", "sequential": "Greens"},
    "typography": {
        "base_font_size": 9,
        "title_size": 10,
        "label_size": 9,
        "tick_size": 8,
        "legend_size": 8,
        "annotation_size": 7,
        "title_weight": "bold",
        "title_case": True,
    },
    "output": {"screen_dpi": 150, "save_dpi": 300, "pdf_fonttype": 42, "ps_fonttype": 42},
    "style": {"seaborn_style": "whitegrid", "seaborn_context": "paper", "grid_alpha": 0.3},
}


def color(name: str) -> str:
    """One palette colour by role (``real``, ``synthetic``, ``positive``, …)."""
    return config()["palette"][name]


#: Kept as module constants so existing imports keep working; both now come from the config.
DIVERGING = config()["colormaps"]["diverging"]
SEQUENTIAL = config()["colormaps"]["sequential"]
REAL_COLOR = color("real")
SYNTH_COLOR = color("synthetic")
POS_COLOR = color("positive")
NEG_COLOR = color("negative")
NEUTRAL_COLOR = color("neutral")
FAMILY_PALETTE = dict(
    zip(FAMILY_ORDER, sns.color_palette(config()["categorical"], len(FAMILY_ORDER)))
)

#: Words that stay lower-case inside a Title Case heading.
_MINOR = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "vs",
    "with",
    "per",
}


def title_case(text: str) -> str:
    """Title Case a heading, leaving acronyms, maths and hyphenated technical terms intact.

    Written rather than using ``str.title()`` because that mangles everything this project cares
    about: ``str.title()`` turns "CNNSpot" into "Cnnspot", "AUROC" into "Auroc" and "real-like" into
    "Real-Like". Only the first character of a word is ever touched, and only when the word is
    entirely lower-case.
    """
    if not config()["typography"].get("title_case", True):
        return text
    out = []
    for i, word in enumerate(text.split(" ")):
        if not word or not word[0].isalpha() or not word.islower():
            out.append(word)  # acronym, maths, mixed case, or punctuation — leave alone
        elif i > 0 and word.strip(",:") in _MINOR:
            out.append(word)
        else:
            out.append(word[0].upper() + word[1:])
    return " ".join(out)


def apply_style() -> None:
    """Project-wide figure defaults from ``reproduction/config/figures.yaml``. Call once per figure builder."""
    cfg = config()
    t, o, st = cfg["typography"], cfg["output"], cfg["style"]
    sns.set_theme(style=st["seaborn_style"], context=st["seaborn_context"])
    mpl.rcParams.update(
        {
            "figure.dpi": o["screen_dpi"],
            "savefig.dpi": o["save_dpi"],
            "font.size": t["base_font_size"],
            "axes.titlesize": t["title_size"],
            "axes.labelsize": t["label_size"],
            "axes.titleweight": t["title_weight"],
            "xtick.labelsize": t["tick_size"],
            "ytick.labelsize": t["tick_size"],
            "legend.fontsize": t["legend_size"],
            "axes.grid": True,
            "grid.alpha": st["grid_alpha"],
            "axes.prop_cycle": mpl.cycler(color=sns.color_palette(cfg["categorical"])),
            "pdf.fonttype": o["pdf_fonttype"],
            "ps.fonttype": o["ps_fonttype"],
        }
    )


def save_caption(folder: str, name: str, text: str) -> Path:
    """Write ``<folder>/<name>-caption.txt`` under ``reproduction/experiments/figures/``.

    Captions are **generated from the data**, never hand-written, so the counts and values they
    quote cannot drift from the figure they describe on a rebuild.
    """
    out = FIGURES_ROOT / folder
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}-caption.txt"
    p.write_text(" ".join(text.split()) + "\n")
    return p


def save_figure(fig, folder: str, name: str, table: pd.DataFrame | None = None) -> dict[str, Path]:
    """Write ``<folder>/<name>.{png,pdf}`` under ``reproduction/experiments/figures/``, plus its source table.

    Args:
        fig: the matplotlib figure.
        folder: e.g. ``fig3-generalization``.
        name: lowercase-kebab file stem.
        table: the exact data the figure was drawn from; saved next to it as ``<name>.csv``.
    """
    out = FIGURES_ROOT / folder
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ext in ("png", "pdf"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths[ext] = p
    if table is not None:
        p = out / f"{name}.csv"
        table.to_csv(p, index=False)
        paths["csv"] = p
    return paths
