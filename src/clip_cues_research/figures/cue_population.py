"""Fig 5 — population-level cue interpretation.

Fig 4 explains one pair; this shows the patterns hold across the dataset. The two panels answer
**different questions**, and conflating them is the misreading the figure exists to prevent:

  A  *Cues associated with detector score* -- within-class association

         r_q = 1/2 * sum_y Corr(z_h, c_q | y)

     "among images of the same class, which cues track the detector's confidence?"

  B  *Cues changed by synthesis* -- paired effect size over content-controlled pairs

         d_q = E[dc_q] / SD(dc_q),      dc_q = c_q(synthetic) - c_q(real)

     "which photographic characteristics change when the same content is synthesized?"

A cue can be large in B and absent from A (synthesis changes it, the detector ignores it) or the
reverse. That is the point.

**Two layouts are built** (``layout=``), because the choice is a real trade-off:

- ``independent`` -- each panel ranks its own top ``n_show`` cues, per PLAN_FIGURES_2. Faithful to
  "the strongest cues by each estimand", but the panels no longer share rows. Cues that make *both*
  lists are drawn with a filled marker so the overlap is still visible.
- ``shared`` -- both panels show the union of the top ``n_each`` per estimand on one y-axis, so every
  row can be read across. Costs some of each panel's tail.

**Canonical space only.** ``outputs/e8/paired/paired_cue_shifts_antonyms.csv`` looks like it answers
Panel B but was computed against ``antonyms_diff_embeddings.pt``, i.e. the retracted W-squared text
space (bug fixed 2026-07-17). Panel B is recomputed here from the checksummed F-snapshot
(``cue_scores/synthclic__antonyms``, canonical) -- the same array Fig 4 draws its single example
from, and the same one behind Panel A's F1 profile.

Uncertainty on the raw mean shift is a cluster bootstrap by source photo: SynthCLIC's real image and
its four synthetic counterparts share an ``image_id`` and are not independent. ``d_q`` itself is a
standardized effect size and carries no interval; both statistics are kept in the exported CSV so
neither reading is locked in by the figure.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clip_cues_research.figures.style import (
    apply_style,
    color,
    cue_families,
    save_figure,
    title_case,
)
from clip_cues_research.finalexp import data as D

REAL_SOURCE = "clic2020"
F1_PROFILE = Path(
    "reproduction/experiments/final_consolidation/F1-canonical-stability/runs/seed123/cue_profile.csv"
)
VOCAB_CSV = "data/vocabularies/antonyms.csv"


def paired_deltas(dataset: str = "synthclic", split: str = "test", vocab: str = "antonyms"):
    """``(delta_matrix, image_ids, cue_names, generators)`` for every real<->synthetic pair.

    One row per (source photo, generator) pair, pooled across all generators. Rows align
    positionally with ``image_ids``, which double as bootstrap cluster labels.
    """
    npz = D.get_npz(f"cue_scores/{dataset}__{vocab}")
    scores, names = npz["scores"].astype(np.float64), [str(c) for c in npz["cues"]]
    frame = D.get_frame(f"projected/{dataset}", expected_space=D.SPACE_CANON)
    df = frame.df.reset_index(drop=True)
    df["image_id"] = df["image_id"].astype(str)

    sel = df["split"] == split
    sub = df[sel]
    real_rows = {
        i: k for k, (s, i) in enumerate(zip(df["source"], df["image_id"])) if s == REAL_SOURCE
    }

    deltas, ids, gens = [], [], []
    for gen in sorted(g for g in sub["source"].unique() if g != REAL_SOURCE):
        g_rows = [
            (i, k)
            for k, (s, i) in enumerate(zip(df["source"], df["image_id"]))
            if s == gen and sel.iloc[k]
        ]
        for image_id, k in g_rows:
            r = real_rows.get(image_id)
            if r is not None and sel.iloc[r]:
                deltas.append(scores[k] - scores[r])
                ids.append(image_id)
                gens.append(gen)
    if not deltas:
        raise ValueError(f"no {REAL_SOURCE}<->synthetic pairs in {dataset}/{split}")
    return np.vstack(deltas), np.asarray(ids), names, np.asarray(gens)


def population_cue_deltas(
    dataset: str = "synthclic",
    split: str = "test",
    vocab: str = "antonyms",
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-cue paired effect size ``d_q`` plus the raw mean shift with a cluster-bootstrap 95% CI."""
    Dm, ids, names, gens = paired_deltas(dataset, split, vocab)
    rng = np.random.default_rng(seed)
    uids = np.unique(ids)
    idx_of = {u: np.where(ids == u)[0] for u in uids}

    draws = np.empty((n_boot, Dm.shape[1]))
    for b in range(n_boot):
        rows = np.concatenate([idx_of[u] for u in rng.choice(uids, len(uids), replace=True)])
        draws[b] = Dm[rows].mean(0)

    sd = Dm.std(0, ddof=1)
    return pd.DataFrame(
        {
            "cue": names,
            "d_q": Dm.mean(0) / np.clip(sd, 1e-12, None),
            "delta": Dm.mean(0),
            "delta_sd": sd,
            "delta_ci_lo": np.percentile(draws, 2.5, axis=0),
            "delta_ci_hi": np.percentile(draws, 97.5, axis=0),
            "n_pairs": len(ids),
            "n_clusters": len(uids),
            "n_generators": int(len(np.unique(gens))),
        }
    )


def detector_association(profile_csv: Path = F1_PROFILE) -> pd.DataFrame:
    """F1's within-class cue profile for the canonical detector (already canonical)."""
    if not profile_csv.exists():
        raise FileNotFoundError(
            f"{profile_csv} missing — run scripts/finalexp/run_f1_canonical_stability.py"
        )
    df = pd.read_csv(profile_csv)
    return df.rename(columns={"within_macro_r": "within_class_r"})[["cue", "within_class_r"]]


def merged_cues(
    dataset: str = "synthclic", split: str = "test", n_boot: int = 2000
) -> pd.DataFrame:
    """All cues with both estimands and their family — the exported superset behind either layout."""
    shifts = population_cue_deltas(dataset, split, n_boot=n_boot)
    assoc = detector_association()
    fams = cue_families(VOCAB_CSV)
    merged = shifts.merge(assoc, on="cue", how="inner")
    merged["family"] = merged["cue"].map(fams).fillna("other")
    phrases = cue_labels()
    merged["label"] = [textwrap.fill(phrases.get(c, c), width=LABEL_WRAP) for c in merged["cue"]]
    return merged


def _top(df: pd.DataFrame, column: str, n: int) -> pd.DataFrame:
    return df.reindex(df[column].abs().sort_values(ascending=False).index).head(n)


#: One colour for every point. Cue *families* are still computed and exported in the CSV, but they
#: are not drawn: colouring a dozen rows by a seven-category taxonomy needed a legend that competed
#: with the two panels while adding nothing to the comparison the figure actually makes.
DOT_COLOR = color("neutral")

#: What the two ends of each panel mean. Rows are labelled with the cue's **positive** phrase, so a
#: point's sign says which way that named property pushes -- without this the reader has to remember
#: both what `instant_camera_cues` means and which pole a positive score corresponds to.
POLES = {
    "within_class_r": ("$\\leftarrow$ more real", "more synthetic $\\rightarrow$"),
    "d_q": ("$\\leftarrow$ less in synthetic", "more in synthetic $\\rightarrow$"),
}
#: Positive phrases run to 38 characters; 26 keeps every one to at most two lines.
LABEL_WRAP = 26


def cue_labels(vocab_csv: str = VOCAB_CSV) -> dict[str, str]:
    """``{cue -> positive phrase}``: the readable name of the cue's ``+`` end."""
    df = pd.read_csv(vocab_csv)
    return dict(zip(df["attribute_name"], df["positive"]))


def _panel(
    ax,
    sel: pd.DataFrame,
    value: str,
    title: str,
    xlabel: str,
    lo: str | None = None,
    hi: str | None = None,
):
    y = np.arange(len(sel))
    if lo is not None:
        err = np.vstack([sel[value] - sel[lo], sel[hi] - sel[value]])
        _ = ax.errorbar(
            sel[value], y, xerr=err, fmt="none", ecolor="0.35", elinewidth=1.1, capsize=2
        )
    _ = ax.scatter(sel[value], y, color=DOT_COLOR, s=44, zorder=3, edgecolor="white", linewidth=0.6)
    _ = ax.axvline(0, color="0.3", linewidth=1.0, zorder=1)
    _ = ax.set_yticks(y)
    _ = ax.set_yticklabels(sel["label"], fontsize=7.5)
    _ = ax.set_title(title_case(title), fontsize=9)
    _ = ax.set_xlabel(xlabel, fontsize=8, labelpad=14)
    _ = ax.tick_params(axis="x", labelsize=7)
    _ = ax.grid(axis="y", alpha=0.15)
    left, right = POLES[value]
    for x, text, ha in ((0.0, left, "left"), (1.0, right, "right")):
        _ = ax.text(
            x, -0.055, text, transform=ax.transAxes, ha=ha, va="top", fontsize=7, color="0.4"
        )


TITLE_A = "A  Cues associated with detector score"
TITLE_B = "B  Cues changed by synthesis"
XLABEL_A = r"within-class $\mathrm{Corr}(z_h,\, c_q \mid y)$"
XLABEL_B = r"paired effect size  $d_q = \mathbb{E}[\Delta c_q]\,/\,\mathrm{SD}(\Delta c_q)$"


def cue_population_figure(
    dataset: str = "synthclic",
    split: str = "test",
    layout: str = "independent",
    n_show: int = 12,
    n_each: int = 6,
    n_boot: int = 2000,
    out_folder: str = "fig5-cue-population",
    merged: pd.DataFrame | None = None,
) -> dict:
    """Build Fig 5 in one of the two layouts and return paths + the plotted table."""
    apply_style()
    merged = merged_cues(dataset, split, n_boot=n_boot) if merged is None else merged

    if layout == "independent":
        a = (
            _top(merged, "within_class_r", n_show)
            .sort_values("within_class_r")
            .reset_index(drop=True)
        )
        b = _top(merged, "d_q", n_show).sort_values("d_q").reset_index(drop=True)
        n_rows = max(len(a), len(b))
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 0.32 * n_rows + 1.5))
        _panel(axes[0], a, "within_class_r", TITLE_A, XLABEL_A)
        _panel(axes[1], b, "d_q", TITLE_B, XLABEL_B)
        plotted = pd.concat([a.assign(panel="A"), b.assign(panel="B")], ignore_index=True)
        # Still exported even though it is no longer drawn: the overlap between the two lists is the
        # quantity the text quotes, it just does not need its own marker style on the figure.
        plotted["in_both_panels"] = plotted["cue"].isin(set(a["cue"]) & set(b["cue"]))
    elif layout == "shared":
        sel = pd.concat([_top(merged, "d_q", n_each), _top(merged, "within_class_r", n_each)])
        sel = sel.drop_duplicates("cue").sort_values("d_q").reset_index(drop=True)
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 0.32 * len(sel) + 1.5), sharey=True)
        _panel(axes[0], sel, "within_class_r", TITLE_A, XLABEL_A)
        _panel(axes[1], sel, "d_q", TITLE_B, XLABEL_B)
        _ = axes[1].tick_params(axis="y", left=False)
        plotted = sel.assign(panel="AB", in_both_panels=True)
    else:
        raise ValueError(f"unknown layout {layout!r} (independent|shared)")

    _ = fig.tight_layout()

    name = f"fig5-cue-population-{layout}"
    paths = save_figure(fig, out_folder, name, table=plotted)
    plt.close(fig)
    return {"paths": paths, "table": plotted, "all_cues": merged, "layout": layout}


def caption_text(merged: pd.DataFrame, overlap: int, n_show: int, n_each: int) -> str:
    """A suggested caption, with every number read from the data rather than retyped."""
    n_pairs = int(merged["n_pairs"].iloc[0])
    n_clusters = int(merged["n_clusters"].iloc[0])
    n_gen = int(merged["n_generators"].iloc[0])
    return (
        f"Population-level cue interpretation on SynthCLIC (test split; {n_pairs:,} "
        f"content-controlled pairs over {n_clusters:,} source photographs and {n_gen} generators). "
        "(A) Within-class association between each named cue and the canonical detector's logit, "
        "r_q = 1/2 sum_y Corr(z_h, c_q | y), computed within each class so the association is not "
        "driven by the class difference itself. (B) Paired effect size of synthesis, "
        "d_q = E[dc_q] / SD(dc_q), where dc_q = c_q(synthetic) - c_q(real) for the same source "
        "photograph. Rows are labelled with each cue's positive phrase and the panel poles are "
        "annotated, so a point's position reads directly; cue identifiers are in the accompanying "
        "CSV. Cue scores are cosines against named antonym directions in the canonical 768-d shared "
        "image-text space. "
        f"The two panels answer different questions and should not be conflated: of the {n_show} "
        f"strongest cues by each estimand, only {overlap} appear in both. A cue that synthesis "
        "changes is not thereby a cue the detector relies on. Uncertainty on the underlying mean "
        "shift is a cluster bootstrap by source photograph (2,000 draws), since a photograph and "
        "its synthetic counterparts are not independent; per-cue intervals are in "
        "fig5-all-cues.csv."
    )


def both_layouts(
    dataset: str = "synthclic",
    split: str = "test",
    n_show: int = 12,
    n_each: int = 6,
    n_boot: int = 2000,
    out_folder: str = "fig5-cue-population",
) -> dict:
    """Build both layouts from **one** bootstrap, and export the full 168-cue table alongside."""
    merged = merged_cues(dataset, split, n_boot=n_boot)
    out = {
        lay: cue_population_figure(
            dataset,
            split,
            layout=lay,
            n_show=n_show,
            n_each=n_each,
            out_folder=out_folder,
            merged=merged,
        )
        for lay in ("independent", "shared")
    }
    folder = Path(out["independent"]["paths"]["csv"]).parent
    all_path = folder / "fig5-all-cues.csv"
    merged.sort_values("d_q", ascending=False).to_csv(all_path, index=False)
    out["all_cues_csv"] = all_path

    ind = out["independent"]["table"]
    overlap = int(ind.loc[ind["in_both_panels"], "cue"].nunique())
    caption_path = folder / "fig5-cue-population-caption.txt"
    caption = caption_text(merged, overlap, n_show, n_each)
    caption_path.write_text(caption + "\n")
    out["caption_path"] = caption_path
    out["caption"] = caption
    out["merged"] = merged
    return out
