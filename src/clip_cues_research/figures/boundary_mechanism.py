"""Fig 6 — cross-dataset boundary difference.

The strongest statement the completed work supports about *why* transfer fails: CLIP does not learn
one invariant "synthetic-image mechanism". The semantic boundary changes with the training
distribution, and its signed difference separates processing artifacts from photographic optics.

**One panel** (PLAN_FIGURES_2): the signed decomposition of ``D_e(CNNSpot) - D_e(SynthCLIC)`` onto
named cue axes, with the data-weighted similarity carried as a text annotation rather than a 3x3
heatmap. Everything is read from F6's artifacts; nothing is recomputed here.

**The annotated similarity is the data-weighted (Sigma-metric) cosine, and the figure says so.** The
raw weight cosine tells a materially weaker story on the same normals (sc~cnnspot -0.06 vs -0.21),
and E11a's conclusion was exactly that boundaries must be compared where the images lie. Both
numbers are in the exported CSV, and the raw value is annotated alongside, so the choice is
auditable rather than hidden.

The full 3x3 matrix -- including SynthBuster+, which shows the near-orthogonality is not a
CNNSpot-specific artifact -- stays available as ``F6-cross-dataset/artifacts/boundary_cosines.csv``
and belongs in the text as a small table. ``similarity_matrices()`` is kept here so that table can
be regenerated from the same code path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clip_cues_research.figures.style import apply_style, color, save_caption, save_figure

F6 = Path("reproduction/experiments/final_consolidation/F6-cross-dataset/artifacts")
COSINES = F6 / "boundary_cosines.csv"
DELTA_AXES = F6 / "delta_axes.csv"
SUMMARY = F6 / "summary.json"

ORDER = ["synthclic", "cnnspot", "synthbuster-plus"]
PRETTY = {"synthclic": "SynthCLIC", "cnnspot": "CNNSpot", "synthbuster-plus": "SynthBuster+"}
#: Empty y-margin, in bar slots, reserved at each end of the panel for the two half-labels. Two
#: lines at 8pt occupy ~0.8 of a slot, so this clears the extreme bars with a little air, no more.
LABEL_PAD = 1.2


def similarity_matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(sigma_metric, raw)`` symmetric 3x3 boundary-similarity matrices."""
    if not COSINES.exists():
        raise FileNotFoundError(f"{COSINES} missing — run scripts/finalexp/run_f6_cross_dataset.py")
    df = pd.read_csv(COSINES)
    sig = pd.DataFrame(np.eye(len(ORDER)), index=ORDER, columns=ORDER)
    raw = sig.copy()
    for _, r in df.iterrows():
        a, b = r["dataset_a"], r["dataset_b"]
        sig.loc[a, b] = sig.loc[b, a] = float(r["sigma_cosine_on_a"])
        raw.loc[a, b] = raw.loc[b, a] = float(r["raw_cosine"])
    return sig, raw


def delta_axes(top_n: int = 14) -> pd.DataFrame:
    """Top signed Delta axes by |coefficient|, ordered for a diverging coefficient plot."""
    if not DELTA_AXES.exists():
        raise FileNotFoundError(
            f"{DELTA_AXES} missing — run scripts/finalexp/run_f6_cross_dataset.py"
        )
    df = pd.read_csv(DELTA_AXES)
    df = df.reindex(df["alpha_coef"].abs().sort_values(ascending=False).index).head(top_n)
    return df.sort_values("alpha_coef").reset_index(drop=True)


def boundary_delta_figure(top_n: int = 14, out_folder: str = "fig6-boundary-delta") -> dict:
    """Build Fig 6: one diverging panel, similarity carried as an annotation."""
    apply_style()
    sig, raw = similarity_matrices()
    axes_df = delta_axes(top_n)

    fig, ax = plt.subplots(figsize=(7.6, 0.34 * (len(axes_df) + 2 * LABEL_PAD) + 1.6))

    y = np.arange(len(axes_df))
    # Same orange/blue as every other signed quantity in the set (reproduction/config/figures.yaml):
    # positive = CNNSpot-associated. np.where cannot broadcast RGB tuples — pick per bar.
    colors = [color("positive") if v > 0 else color("negative") for v in axes_df["alpha_coef"]]
    _ = ax.barh(y, axes_df["alpha_coef"], color=colors, edgecolor="white", linewidth=0.6)
    _ = ax.axvline(0, color="0.25", linewidth=1.0)
    _ = ax.set_yticks(y)
    _ = ax.set_yticklabels(axes_df["cue"], fontsize=8)
    _ = ax.set_xlabel(
        r"signed coefficient on  $\hat{w}_{\mathrm{CNNSpot}} - \hat{w}_{\mathrm{SynthCLIC}}$"
    )
    _ = ax.grid(axis="y", alpha=0.12)

    lim = float(np.abs(axes_df["alpha_coef"]).max()) * 1.55
    _ = ax.set_xlim(-lim, lim)
    # The two half-labels sit at the extreme ends, which is also where the longest bars are — so
    # reserve empty slots beyond the last bar at each end and put the labels there. Without this
    # they overprint `color_bleeding` at the top and `posterization` at the bottom.
    top, bottom = len(axes_df) - 0.5 + LABEL_PAD, -0.5 - LABEL_PAD
    _ = ax.set_ylim(bottom, top)
    # SIGN CONVENTION — do not swap these labels. Delta = w(CNNSpot) - w(SynthCLIC), so a positive
    # coefficient means the CNNSpot boundary weights that cue toward "synthetic" relative to
    # SynthCLIC's. Each label must therefore sit on the half it describes; inverting them inverts
    # the figure's central claim. `test_fig6_annotation_sides_match_the_data` pins this.
    _ = ax.text(
        lim * 0.96,
        top - 0.15,
        "CNNSpot-associated\n(compression, upscaling, processing)",
        fontsize=8,
        color="0.25",
        ha="right",
        va="top",
    )
    # "tone" belongs in this list: `posterization` is the largest coefficient on this half, ahead of
    # the optics cues. Omitting it would make the subtitle describe a half it does not lead.
    _ = ax.text(
        -lim * 0.96,
        bottom + 0.15,
        "SynthCLIC-associated\n(optics, grain, tone, composition)",
        fontsize=8,
        color="0.25",
        ha="left",
        va="bottom",
    )

    s = float(sig.loc["synthclic", "cnnspot"])
    r = float(raw.loc["synthclic", "cnnspot"])
    _ = ax.text(
        0.5,
        -0.155,
        rf"$\cos_\Sigma(D_e^{{\mathrm{{CNNSpot}}}},\, D_e^{{\mathrm{{SynthCLIC}}}}) = {s:+.2f}$"
        f"     (raw weight cosine {r:+.2f})",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.5,
        color="0.3",
    )
    _ = fig.tight_layout()

    sim_long = sig.copy()
    sim_long.index.name = "dataset_a"
    sim_long = (
        sim_long.reset_index()
        .melt(id_vars="dataset_a", var_name="dataset_b", value_name="sigma_cosine")
        .assign(panel="similarity_annotation")
    )
    out_table = pd.concat([sim_long, axes_df.assign(panel="delta_axes")], ignore_index=True)

    paths = save_figure(fig, out_folder, "fig6-boundary-delta", table=out_table)
    plt.close(fig)

    pos = axes_df[axes_df["alpha_coef"] > 0]["cue"].tolist()
    neg = axes_df[axes_df["alpha_coef"] < 0]["cue"].tolist()
    paths["caption"] = save_caption(
        out_folder,
        "fig6-boundary-delta",
        f"""
        Signed decomposition of the difference between two CLIP detectors' decision directions,
        w(CNNSpot) - w(SynthCLIC), onto named cue axes ({len(axes_df)} largest coefficients). A
        positive coefficient means the CNNSpot boundary weights that cue toward "synthetic" relative
        to the SynthCLIC boundary. The split is interpretable: the CNNSpot side is dominated by
        processing and compression artifacts ({", ".join(pos)}), the SynthCLIC side by optics, grain,
        tone and composition ({", ".join(neg)}). The two boundaries are close to orthogonal where the
        images actually lie - data-weighted cosine {s:+.2f}, against {r:+.2f} for the raw weight
        cosine, both reported because they tell materially different stories on the same normals.
        This is the mechanism behind the cross-family transfer failure: the detectors do not learn one
        invariant synthetic-image cue, they learn what distinguishes their own training corpus.
    """,
    )
    return {"paths": paths, "sigma": sig, "raw": raw, "axes": axes_df}
