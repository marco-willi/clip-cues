"""Compact visuals: the information-restriction cascade (§8) and the stability summary (§9).

Neither result justifies a full-width figure, but both are load-bearing, so they are built small and
shipped alongside a LaTeX table — if the paper gets tight, the table can replace the figure without
recomputing anything.

  cascade   D_h -> D_e -> named cues, with the paired CIs on the arrows. Reads as successive
            information restriction, which is exactly the claim.
  stability F1/F2 in three rows: the effective boundary is stable, its eight axes are not.

Every number is read from the F-experiments' `summary.json` — nothing is hard-coded, so the figures
cannot drift from the results they describe.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from clip_cues_research.figures.latex import MISSING, escape_text, paper_table, signed_ci
from clip_cues_research.figures.style import FIGURES_ROOT, apply_style, save_figure

ROOT = Path("reproduction/experiments/final_consolidation")
F1 = ROOT / "F1-canonical-stability/artifacts/summary.json"
F2 = ROOT / "F2-matched-k8/artifacts/summary.json"
F3 = ROOT / "F3-projected-head/artifacts/summary.json"
F4 = ROOT / "F4-cue-capacity/artifacts/summary.json"


def _load(p: Path) -> dict:
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run the F-experiments first")
    return json.loads(p.read_text())


def cascade_table() -> pd.DataFrame:
    """The three AUROC levels and the two paired deltas between them."""
    f3, f4 = _load(F3), _load(F4)
    pc = f3["projection_cost_Dh_minus_De"]
    cue = f4["by_vocabulary"]["antonyms"]
    return pd.DataFrame(
        [
            {
                "stage": "Full pooler $D_h$",
                "auroc": f3["Dh_detection"]["auroc_mean"],
                "delta": None,
                "ci_lo": None,
                "ci_hi": None,
            },
            {
                "stage": "Shared-space $D_e$",
                "auroc": f3["De_detection"]["auroc_mean"],
                "delta": -pc["mean_delta"],
                "ci_lo": -pc["ci_hi_mean"],
                "ci_hi": -pc["ci_lo_mean"],
            },
            {
                "stage": "168 named cues",
                "auroc": cue["auroc_cue_mean"],
                "delta": cue["delta_mean"],
                "ci_lo": cue["ci_lo_mean"],
                "ci_hi": cue["ci_hi_mean"],
            },
        ]
    )


def stability_table() -> pd.DataFrame:
    """F1/F2's three representations by direction and cue-profile stability."""
    f1, f2 = _load(F1), _load(F2)
    st = f2["stability_table"]
    return pd.DataFrame(
        [
            {
                "representation": "$D_h$ (canonical, k=1)",
                "direction": f1["stability_across_seed_pairs"]["sigma_cosine"]["mean"],
                "cue_profile": f1["stability_across_seed_pairs"]["cue_profile_spearman"]["mean"],
            },
            {
                "representation": "k=8 effective direction",
                "direction": st["k8_effective_direction"]["sigma_cosine"]["mean"],
                "cue_profile": st["k8_effective_direction"]["cue_profile_spearman"]["mean"],
            },
            {
                "representation": "k=8 individual axes",
                "direction": st["k8_individual_axes"]["matched_abs_cosine"]["mean"],
                "cue_profile": st["k8_individual_axes"]["cue_profile_spearman"]["mean"],
            },
        ]
    )


def cascade_figure(out_folder: str = "compact") -> dict:
    apply_style()
    df = cascade_table()
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    _ = ax.axis("off")

    ys = [0.86, 0.5, 0.14]
    for y, (_, r) in zip(ys, df.iterrows()):
        _ = ax.text(0.03, y, r["stage"], fontsize=10, va="center")
        _ = ax.text(
            0.97,
            y,
            f"{r['auroc']:.3f}",
            fontsize=11,
            va="center",
            ha="right",
            fontweight="bold",
            family="monospace",
        )
    for y0, y1, (_, r) in zip(ys[:-1], ys[1:], df.iloc[1:].iterrows()):
        mid = (y0 + y1) / 2
        _ = ax.annotate(
            "",
            xy=(0.1, y1 + 0.06),
            xytext=(0.1, y0 - 0.06),
            arrowprops=dict(arrowstyle="-|>", color="0.35", linewidth=1.2),
        )
        _ = ax.text(
            0.16,
            mid,
            f"{r['delta']:+.3f}  [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]",
            fontsize=8.5,
            va="center",
            color="0.3",
        )
    _ = ax.set_title("Successive information restriction", fontsize=10, fontweight="bold")
    _ = ax.text(
        0.03,
        -0.02,
        "SynthCLIC test AUROC; paired cluster-bootstrap 95% CI",
        fontsize=7.5,
        color="0.45",
        transform=ax.transAxes,
    )

    paths = save_figure(fig, out_folder, "cascade-information-restriction", table=df)
    plt.close(fig)
    return {"paths": paths, "table": df}


def stability_figure(out_folder: str = "compact") -> dict:
    apply_style()
    df = stability_table()
    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    y = range(len(df))
    height = 0.34
    _ = ax.barh(
        [i + height / 2 for i in y],
        df["direction"],
        height=height,
        label="direction (Sigma-cosine)",
        color="#4C72B0",
        edgecolor="white",
    )
    _ = ax.barh(
        [i - height / 2 for i in y],
        df["cue_profile"],
        height=height,
        label="cue profile (Spearman)",
        color="#DD8452",
        edgecolor="white",
    )
    for i, r in df.iterrows():
        _ = ax.text(
            r["direction"] + 0.02, i + height / 2, f"{r['direction']:.3f}", va="center", fontsize=8
        )
        _ = ax.text(
            r["cue_profile"] + 0.02,
            i - height / 2,
            f"{r['cue_profile']:.3f}",
            va="center",
            fontsize=8,
        )
    _ = ax.set_yticks(list(y))
    _ = ax.set_yticklabels(df["representation"], fontsize=9)
    _ = ax.set_xlim(0, 1.15)
    _ = ax.set_xlabel("agreement across seed pairs")
    _ = ax.invert_yaxis()
    _ = ax.legend(loc="lower right", frameon=False, fontsize=8)
    _ = ax.set_title(
        "The boundary is stable; its eight axes are not", fontsize=10, fontweight="bold"
    )

    paths = save_figure(fig, out_folder, "stability-summary", table=df)
    plt.close(fig)
    return {"paths": paths, "table": df}


# ── LaTeX table forms ─────────────────────────────────────────────────────────────────────────
# The tables *are* the manuscript artifact (PLAN_FIGURES_2 retired both renders), so they go
# through the house emitter in `figures/latex.py` rather than bare `to_latex`. Everything below is
# display-layer only: the DataFrames the figures and CSVs use keep raw floats and plain labels,
# because matplotlib mathtext cannot read `$k{=}8$` and a CSV should not carry `$`.

#: Feature-space dimensions (`finalexp/spaces.py`); D_e's is cross-checked against F3's summary.
_DIM_DH, _DIM_DE = 1024, 768

#: Data label -> manuscript label. A KeyError here means `cascade_table` was renamed; that is the
#: intended failure — a silently unrenamed row would ship a snake_case cell into the paper.
_CASCADE_TEX = {
    "Full pooler $D_h$": rf"$D_h$ (full pooler, {_DIM_DH}-d)",
    "Shared-space $D_e$": rf"$D_e$ (shared space, {_DIM_DE}-d)",
    "168 named cues": r"$D_{\mathrm{cue}}$",  # the cue count is appended, read from F4
}
_STABILITY_TEX = {
    "$D_h$ (canonical, k=1)": r"$D_h$ (canonical, $k{=}1$)",
    "k=8 effective direction": r"$k{=}8$ effective direction",
    "k=8 individual axes": r"$k{=}8$ individual axes",
}


def _three_dp(x: float) -> str:
    """These are AUROCs and correlations, not the export bundle's 2-decimal per-generator mAPs."""
    return f"{x:.3f}"


def _cascade_tex() -> tuple[pd.DataFrame, str]:
    """Display frame + caption for the cascade, both derived from F3/F4's summaries."""
    f3, f4 = _load(F3), _load(F4)
    cue = f4["by_vocabulary"]["antonyms"]
    n_cues, n_seeds = cue["n_cues"], len(f4["seeds"])
    if f3["space"]["dim"] != _DIM_DE:  # the shared space is the one that could plausibly move
        raise ValueError(f"F3 shared space is {f3['space']['dim']}-d, table says {_DIM_DE}-d")

    df = cascade_table()
    stage = [_CASCADE_TEX[s] for s in df["stage"]]
    stage[-1] = f"{stage[-1]} ({n_cues} named cues)"
    delta = [
        MISSING if pd.isna(r["delta"]) else signed_ci(r["delta"], r["ci_lo"], r["ci_hi"])
        for _, r in df.iterrows()
    ]
    out = pd.DataFrame(
        {
            "Stage": stage,
            "AUROC": df["auroc"].to_numpy(),
            r"$\Delta$ vs.\ previous row [95\% CI]": delta,
        }
    )
    caption = (
        "Successive information restriction on the SynthCLIC test split "
        f"(mean over {n_seeds} seeds). Each $\\Delta$ is the change in AUROC relative to the row "
        f"above, with a 95\\% cluster bootstrap confidence interval ({f4['n_boot']} resamples, "
        # `clusters` is a summary.json key, not prose: it reads "source photo (image_id)", and that
        # `_` is a text-mode compile error — drop the machine-readable half, escape what is left.
        f"clustered on {escape_text(f4['clusters'].split(' (')[0])}); negative values are losses. "
        "Projection into the shared "
        "image--text space costs little, restriction to the named cue span costs roughly twice "
        "as much."
    )
    return out, caption


def _stability_tex() -> tuple[pd.DataFrame, str]:
    """Display frame + caption for the stability summary, derived from F1/F2's summaries."""
    f1, f2 = _load(F1), _load(F2)
    n_cues = _load(F4)["by_vocabulary"]["antonyms"]["n_cues"]
    n_seeds = len(f1["seeds"])
    n_pairs = f1["stability_across_seed_pairs"]["sigma_cosine"]["n_pairs"]
    k = f2["k"]

    df = stability_table()
    out = pd.DataFrame(
        {
            "Representation": [_STABILITY_TEX[r] for r in df["representation"]],
            "Direction agreement": df["direction"].to_numpy(),
            r"Cue profile $\rho$": df["cue_profile"].to_numpy(),
        }
    )
    caption = (
        f"Stability across {n_seeds} seed refits ({n_pairs} seed pairs): agreement of the decision "
        f"direction and of the {n_cues}-cue association profile (Spearman $\\rho$). Direction "
        "agreement is the covariance-weighted cosine $\\cos_\\Sigma$ for the two effective "
        "directions and the Hungarian-matched $|\\cos|$ for the individual axes, which carry no "
        "calibrated score scale of their own. The predictive boundary is stable while its "
        f"decomposition into $k{{=}}{k}$ axes is not."
    )
    return out, caption


def latex_tables(out_folder: str = "tables", root: Path | None = None) -> dict[str, Path]:
    """Table forms of both compacts, so either can replace its figure without recomputation.

    ``root`` defaults to the figure tree; tests point it at a temporary directory so a run of the
    suite cannot rewrite the shipped manuscript artifact.
    """
    out = (root or FIGURES_ROOT) / out_folder
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    casc, casc_caption = _cascade_tex()
    paths["cascade"] = out / "cascade-information-restriction.tex"
    _ = paths["cascade"].write_text(
        paper_table(casc, casc_caption, "tab:rev:cascade", float_format=_three_dp)
    )
    stab, stab_caption = _stability_tex()
    paths["stability"] = out / "stability-summary.tex"
    _ = paths["stability"].write_text(
        paper_table(stab, stab_caption, "tab:rev:stability", float_format=_three_dp)
    )
    return paths
