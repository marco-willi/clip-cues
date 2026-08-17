"""Fig 7 — what the sparse concept model uses, per dataset.

Figures 4-6 interpret a *linear* detector post hoc. This one answers a different question: can an
**intrinsically text-grounded** classifier -- one whose only route to a prediction is a sparse set of
named concepts -- make image-specific predictions, and which concepts does it use?

The published concept model (``cm_antonyms_{dataset}``) is a concept-bottleneck head over the 168
antonym directions. For a normalized image embedding ``e`` it computes, per concept ``q``::

    sim_q  = <e, t_q>                                   (t_q = concept text direction, unit)
    a_q    = sigmoid(W_concepts . e)_q                  activation probability (the sparsity gate)
    z      = sum_q  W_cls[q] * (sim_q * a_q)            class logit
    contribution_q = W_cls[q] * sim_q * a_q             per-concept logit contribution

Each dataset panel shows the strongest concepts by **mean contribution** across the test split, with
each concept's **mean activation probability** beside it -- a concept can carry a large contribution
because it fires on nearly every image, or because it fires rarely but decisively, and the two
readings differ.

**Space.** The deployed checkpoint carries its own ``model.text_embeddings``; those were verified
identical to ``data/embeddings/vocab_canon/antonyms.pt`` (diagonal cosine 1.0000, against 0.0021 for
the retracted W-squared ``antonyms_diff_embeddings.pt``). So this model is canonical, and the
vocabulary CSV can be used for its concept *names* -- ``assert_canonical_vocabulary`` pins that
rather than trusting it. Image embeddings come from the checksummed snapshot
(``projected/{dataset}``), which is the derived ``e = Wp h`` under the both-sides-derived rule.

**Caption caveat (keep it).** This is an illustration of one trained concept model. E8 found the
population concept signal diffuse and seed-sensitive, so the individual concept names are
qualitative; the claim is about the *form* of the explanation, not the identity of any one concept.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from clip_cues_research.figures.style import (
    apply_style,
    color,
    save_caption,
    save_figure,
    title_case,
)
from clip_cues_research.finalexp import data as D

CKPT = {
    "cnnspot": "data/checkpoints/cm_antonyms_cnnspot.ckpt",
    "synthclic": "data/checkpoints/cm_antonyms_synthclic.ckpt",
}
PRETTY = {"cnnspot": "CNNSpot", "synthclic": "SynthCLIC"}
VOCAB_CANON = "data/embeddings/vocab_canon/antonyms.pt"
POS_COLOR = color("synthetic")  # pushes the logit toward "synthetic"
NEG_COLOR = color("real")  # ... toward "real"


def assert_canonical_vocabulary(
    ckpt_path: str, vocab_path: str = VOCAB_CANON, tol: float = 1e-3
) -> list[str]:
    """Return concept names, having verified the checkpoint's own text space matches ``vocab_path``.

    The names are only meaningful if the row order of the vocabulary file is the row order of
    ``model.text_embeddings``. That currently holds by coincidence of build history, which is exactly
    the kind of thing that breaks silently and mislabels every bar in the figure. Checking the
    diagonal cosine turns it into a loud failure.
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    T = sd["model.text_embeddings"].numpy().astype(np.float64)
    v = torch.load(vocab_path, weights_only=False)
    E = np.asarray(v["embeddings"], dtype=np.float64)
    if T.shape != E.shape:
        raise ValueError(f"{ckpt_path} text space {T.shape} != {vocab_path} {E.shape}")

    def unit(x):
        return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)

    cos = (unit(T) * unit(E)).sum(1)
    if cos.min() < 1.0 - tol:
        raise ValueError(
            f"{vocab_path} is not the text space of {ckpt_path} "
            f"(min diagonal cosine {cos.min():.4f}); concept names would be wrong"
        )
    return [str(x) for x in v["vocabulary"]]


def concept_profile(dataset: str, split: str = "test") -> pd.DataFrame:
    """Per-concept class-split contribution, activation, usage rate and single-concept AUC.

    Statistic definitions follow the **original** manuscript figure
    (``concept_modeling/analyse.py::plot_concept_importance_summary`` in the archived research repo)
    so a rebuild is comparable to the published version rather than merely similar:

    - ``usage_*`` is the **fraction of images with ``a_q > 0.5``**, not the mean of ``a_q``. The two
      differ substantially here: the gate is rarely above 0.5 *on average*, so the mean understates
      how often a concept is actually switched on for individual images -- which is the whole point
      of a sparse per-example model.
    - ``auc`` uses ``a_q`` as a single feature and is **orientation-corrected** (``max(auc, 1-auc)``),
      so it measures separability regardless of which class the concept fires on.
    """
    from sklearn.metrics import roc_auc_score

    from clip_cues.model import load_concept_model

    names = assert_canonical_vocabulary(CKPT[dataset])
    model, _ = load_concept_model(CKPT[dataset], cache_dir="data/hf_cache", device="cpu")
    model.eval()

    frame = D.get_frame(f"projected/{dataset}", expected_space=D.SPACE_CANON)
    emb, labels, _ = frame.split(split)
    with torch.no_grad():
        out = model(torch.from_numpy(np.asarray(emb, dtype=np.float32)))
    contrib = out["per_concept_logit_contribution"].numpy()
    act = out["per_image_concept_samples"].numpy()

    real, synth = labels == 0, labels == 1
    auc = []
    for j in range(act.shape[1]):
        try:
            a = float(roc_auc_score(labels, act[:, j]))
        except ValueError:
            a = 0.5
        auc.append(max(a, 1.0 - a))

    return pd.DataFrame(
        {
            "dataset": dataset,
            "concept": names,
            "contribution": contrib.mean(0),
            "contribution_synth": contrib[synth].mean(0),
            "contribution_real": contrib[real].mean(0),
            "activation_prob": act.mean(0),
            "usage_synth": (act[synth] > 0.5).mean(0),
            "usage_real": (act[real] > 0.5).mean(0),
            "auc": auc,
            "n_images": len(labels),
        }
    )


def _panel(ax_bar, ax_act, sub: pd.DataFrame, title: str, lim: float, show_xlabel: bool):
    y = np.arange(len(sub))
    colors = [POS_COLOR if v > 0 else NEG_COLOR for v in sub["contribution"]]
    _ = ax_bar.barh(y, sub["contribution"], color=colors, edgecolor="white", linewidth=0.6)
    # Class means on the same axis. Without them the panel is ambiguous: a large mean contribution
    # can come from a concept the model applies to *every* image (no discriminative work) as easily
    # as from one that separates the classes. The gap between the two ticks is the discriminative part.
    _ = ax_bar.scatter(
        sub["contribution_real"],
        y,
        marker="|",
        s=95,
        linewidth=1.8,
        color=color("real_dark"),
        zorder=4,
        label="real-class mean",
    )
    _ = ax_bar.scatter(
        sub["contribution_synth"],
        y,
        marker="|",
        s=95,
        linewidth=1.8,
        color=color("synthetic_dark"),
        zorder=4,
        label="synthetic-class mean",
    )
    _ = ax_bar.axvline(0, color="0.3", linewidth=1.0)
    _ = ax_bar.set_xlim(-lim, lim)
    # Labels INSIDE the panel on the empty side of zero: as y-tick labels they run out of the axes
    # and collide with the neighbouring dataset's activation strip.
    _ = ax_bar.set_yticks(y)
    _ = ax_bar.set_yticklabels([])
    _ = ax_bar.tick_params(axis="y", length=0)
    pad = 0.06 * lim
    for row, r in enumerate(sub.itertuples()):
        right = r.contribution > 0
        _ = ax_bar.text(
            -pad if right else pad,
            row,
            r.concept,
            fontsize=7.5,
            ha="right" if right else "left",
            va="center",
            color="0.2",
        )
    _ = ax_bar.set_title(title, loc="left")
    _ = ax_bar.grid(axis="y", alpha=0.12)
    _ = ax_bar.tick_params(axis="x", labelsize=7)

    _ = ax_act.scatter(sub["activation_prob"], y, s=34, color="0.35", zorder=3)
    _ = ax_act.set_xlim(0, 1)
    _ = ax_act.set_ylim(*ax_bar.get_ylim())
    _ = ax_act.set_yticks(y)
    _ = ax_act.set_yticklabels([])
    _ = ax_act.set_xticks([0, 0.5, 1])
    _ = ax_act.tick_params(axis="x", labelsize=7)
    _ = ax_act.grid(axis="y", alpha=0.12)
    _ = ax_act.set_title(r"$\bar a_q$", fontsize=8, loc="center")
    if show_xlabel:
        _ = ax_bar.set_xlabel(
            "mean per-concept logit contribution  " r"($\rightarrow$ synthetic)", fontsize=8
        )
        _ = ax_act.set_xlabel("activation\nprobability", fontsize=8)


def concept_model_figure(
    datasets: tuple[str, ...] = ("cnnspot", "synthclic"),
    split: str = "test",
    top_k: int = 12,
    out_folder: str = "fig7-concept-model",
) -> dict:
    """Build Fig 7: one contribution panel + activation-probability strip per dataset."""
    apply_style()
    profiles = {ds: concept_profile(ds, split) for ds in datasets}
    tops = {
        ds: p.reindex(p["contribution"].abs().sort_values(ascending=False).index)
        .head(top_k)
        .sort_values("contribution")
        .reset_index(drop=True)
        for ds, p in profiles.items()
    }
    lim = (
        max(
            float(
                pd.concat([t["contribution"], t["contribution_real"], t["contribution_synth"]])
                .abs()
                .max()
            )
            for t in tops.values()
        )
        * 1.55
    )

    ncol = 2 * len(datasets)
    fig = plt.figure(figsize=(5.9 * len(datasets), 0.36 * top_k + 2.3))
    gs = fig.add_gridspec(1, ncol, width_ratios=[1.0, 0.24] * len(datasets), wspace=0.22)
    axes = []
    for i, ds in enumerate(datasets):
        ax_bar = fig.add_subplot(gs[0, 2 * i])
        _panel(
            ax_bar,
            fig.add_subplot(gs[0, 2 * i + 1]),
            tops[ds],
            f"{'AB'[i]}  {PRETTY.get(ds, ds)}",
            lim,
            show_xlabel=True,
        )
        axes.append(ax_bar)
    h, lab = axes[0].get_legend_handles_labels()
    _ = fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.075))

    table = pd.concat(tops.values(), ignore_index=True)
    paths = save_figure(fig, out_folder, "fig7-concept-model", table=table)
    plt.close(fig)

    full = pd.concat(profiles.values(), ignore_index=True)
    full_path = Path(paths["csv"]).parent / "fig7-all-concepts.csv"
    full.to_csv(full_path, index=False)
    return {"paths": paths, "table": table, "all_concepts": full, "all_concepts_csv": full_path}


# ── original-manuscript variant ───────────────────────────────────────────────────────────────
# A faithful rebuild of the published Fig 7 (`plot_concept_importance_summary` in the archived
# research repo): three panels, concepts ordered by single-concept AUC.
#
#   1  Class separation      dumbbell, mean logit contribution per class
#   2  Activation probability dumbbell, usage rate (fraction of images with a_q > 0.5) per class
#   3  Predictive power      horizontal bars, orientation-corrected single-concept AUC
#
# Kept from the original: the statistics, the dumbbell form, the AUC ordering, the blue/orange
# real/synthetic encoding with circle/square markers, the connecting line coloured by which class is
# higher, and the 0.6 AUC threshold that greys out the weak concepts.
# Changed: K drops from 30 to ~14 (PLAN_FIGURES_2 asks for 10-15); both datasets appear as rows of
# one figure instead of separate figures; typography follows `style.apply_style`.
REAL_COLOR = color("real")
SYNTH_COLOR = color("synthetic")
AUC_STRONG, AUC_WEAK = color("strong"), color("weak")
AUC_THRESHOLD = 0.6


def _dumbbell(
    ax,
    real_vals,
    synth_vals,
    y,
    *,
    title,
    xlabel,
    show_ylabels,
    labels=None,
    zero_line=None,
    xlim=None,
):
    for i, (r, s) in enumerate(zip(real_vals, synth_vals)):
        ax.plot([r, s], [i, i], color=SYNTH_COLOR if s > r else REAL_COLOR, linewidth=2, alpha=0.6)
    _ = ax.scatter(real_vals, y, color=REAL_COLOR, s=36, zorder=3, marker="o", label="real")
    _ = ax.scatter(synth_vals, y, color=SYNTH_COLOR, s=36, zorder=3, marker="s", label="synthetic")
    if zero_line is not None:
        _ = ax.axvline(
            zero_line, color="0.4", linestyle="-" if zero_line == 0 else "--", linewidth=0.7
        )
    _ = ax.set_yticks(y)
    _ = ax.set_yticklabels(labels if show_ylabels else [], fontsize=8)
    _ = ax.set_xlabel(xlabel, fontsize=8)
    if title:
        _ = ax.set_title(title_case(title), fontsize=9)
    _ = ax.tick_params(axis="x", labelsize=7)
    _ = ax.grid(axis="y", alpha=0.12)
    if xlim:
        _ = ax.set_xlim(*xlim)
    _ = ax.set_ylim(-0.6, len(y) - 0.4)
    _ = ax.invert_yaxis()


def concept_importance_figure(
    datasets: tuple[str, ...] = ("cnnspot", "synthclic"),
    split: str = "test",
    top_k: int = 14,
    panels: int = 3,
    out_folder: str = "fig7-concept-model",
) -> dict:
    """Rebuild the published Fig 7. ``panels=2`` drops the AUC column (it can live in a table)."""
    if panels not in (2, 3):
        raise ValueError(f"panels must be 2 or 3, got {panels}")
    apply_style()
    profiles = {ds: concept_profile(ds, split) for ds in datasets}
    tops = {
        ds: p.sort_values("auc", ascending=False).head(top_k).reset_index(drop=True)
        for ds, p in profiles.items()
    }

    # The AUC panel gets half the width of the dumbbells: it carries one number per row, so equal
    # width just stretched 14 bars across space the other panels needed.
    width_ratios = [1.0, 1.0, 0.5][:panels]
    fig, axes = plt.subplots(
        len(datasets),
        panels,
        figsize=(3.9 * sum(width_ratios), 0.26 * top_k * len(datasets) + 1.5 * len(datasets)),
        gridspec_kw={"width_ratios": width_ratios},
        squeeze=False,
    )

    for r, ds in enumerate(datasets):
        sub = tops[ds]
        y = np.arange(len(sub))
        first, last = r == 0, r == len(datasets) - 1

        cmax = (
            float(pd.concat([sub["contribution_real"], sub["contribution_synth"]]).abs().max())
            * 1.15
        )
        _dumbbell(
            axes[r][0],
            sub["contribution_real"],
            sub["contribution_synth"],
            y,
            title="class separation" if first else "",
            xlabel="mean logit contribution" if last else "",
            show_ylabels=True,
            labels=sub["concept"],
            zero_line=0.0,
            xlim=(-cmax, cmax),
        )
        # The dataset belongs to the whole row, not to one panel, so it is a rotated row label
        # rather than part of a panel title.
        _ = axes[r][0].set_ylabel(PRETTY.get(ds, ds), fontsize=11, fontweight="bold", labelpad=10)
        _dumbbell(
            axes[r][1],
            sub["usage_real"],
            sub["usage_synth"],
            y,
            title="activation probability" if first else "",
            xlabel="usage rate  ($a_q > 0.5$)" if last else "",
            show_ylabels=False,
            zero_line=0.5,
            xlim=(-0.05, 1.05),
        )
        if panels == 3:
            ax = axes[r][2]
            colors = [AUC_STRONG if a > AUC_THRESHOLD else AUC_WEAK for a in sub["auc"]]
            _ = ax.barh(y, sub["auc"], color=colors, alpha=0.85, height=0.7)
            # Value labels sit INSIDE the bars: outside they forced the axis past AUC 1.0, which is
            # not a value the statistic can take.
            for i, a in enumerate(sub["auc"]):
                _ = ax.text(
                    a - 0.012,
                    i,
                    f"{a:.2f}",
                    va="center",
                    ha="right",
                    fontsize=6.5,
                    color="white",
                    fontweight="bold",
                )
            _ = ax.set_yticks(y)
            _ = ax.set_yticklabels([])
            _ = ax.set_xlim(0.5, 1.0)
            _ = ax.set_xticks([0.5, 0.75, 1.0])
            _ = ax.axvline(0.5, color="0.4", linewidth=0.7)
            if first:
                _ = ax.set_title(title_case("predictive power"), fontsize=9)
            _ = ax.set_xlabel("single-concept AUC" if last else "", fontsize=8)
            _ = ax.tick_params(axis="x", labelsize=7)
            _ = ax.grid(axis="y", alpha=0.12)
            _ = ax.set_ylim(-0.6, len(y) - 0.4)
            _ = ax.invert_yaxis()

    # One legend for the whole figure — the encoding is identical in every panel, so per-panel
    # legends were both repetitive and (being drawn only on the first row) inconsistent.
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=REAL_COLOR, markersize=7, label="real"),
        plt.Line2D(
            [], [], marker="s", linestyle="", color=SYNTH_COLOR, markersize=7, label="synthetic"
        ),
    ]
    _ = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.004),
    )
    _ = fig.tight_layout(rect=(0, 0.045, 1, 1))
    # `tight_layout` packs the rows so that only the tick labels fit between them, leaving nowhere to
    # put a provenance line -- it landed on top of them. Open the ROW gap (and only the row gap:
    # `wspace`, the space between panels, is untouched) by just enough for one 7 pt line.
    if len(datasets) > 1:
        _ = fig.subplots_adjust(hspace=0.16)

    # Provenance, per row: a concept model trained on one corpus and scored on another would look
    # identical here, so which is which has to be on the figure rather than in the caption alone.
    #
    # Drawn in FIGURE coordinates after `tight_layout`, anchored to the row's own axes box. Placed as
    # in-axes text above the panels it needed title padding to clear, which opened a gap across the
    # whole figure; here it sits in the space that already exists between rows and reserves none of
    # its own.
    for r, ds in enumerate(datasets):
        last = r == len(datasets) - 1
        # Bottom-RIGHT of the row: anchored to the right edge of the row's last panel.
        right = axes[r][-1].get_position().x1
        label = (
            f"Train: {PRETTY.get(ds, ds)}  ·  Inference: {PRETTY.get(ds, ds)} ({split})"
            rf"  ·  Vocabulary: $C_{{\text{{gpt}}}}$ ({len(profiles[ds])})"
        )
        if last:
            # Below the x-axis label, which only the last row carries.
            _ = fig.text(
                right,
                axes[r][0].get_position().y0 - 0.052,
                label,
                ha="right",
                va="top",
                fontsize=7,
                color="0.45",
            )
        else:
            # Sit in the gap that already exists between rows, anchored to the TOP of the next row
            # rather than dropped a guessed distance from this one: a fixed offset overshot the gap
            # and printed the line across the panels below. Anchoring upward from the next row also
            # puts the maximum available daylight between this line and this row's tick labels.
            _ = fig.text(
                right,
                axes[r + 1][0].get_position().y1 + 0.014,
                label,
                ha="right",
                va="bottom",
                fontsize=7,
                color="0.45",
            )
    table = pd.concat(tops.values(), ignore_index=True)
    paths = save_figure(fig, out_folder, f"fig7-concept-importance-{panels}panel", table=table)
    plt.close(fig)

    rng = {ds: (t["auc"].min(), t["auc"].max()) for ds, t in tops.items()}
    spans = "; ".join(f"{PRETTY.get(k, k)} {v[0]:.2f}-{v[1]:.2f}" for k, v in rng.items())
    paths["caption"] = save_caption(
        out_folder,
        f"fig7-concept-importance-{panels}panel",
        f"""
        What the sparse concept model uses, per dataset: the {top_k} concepts with the highest
        single-concept AUC, for models trained and evaluated on each corpus over the 168-term antonym
        vocabulary. The model reaches a prediction only through named concepts - for a normalized
        image embedding it scores each concept q as sim_q = <e, t_q>, gates it with an activation
        probability a_q, and sums W[q] * sim_q * a_q. Left: mean logit contribution per class.
        Middle: usage rate, the fraction of images with a_q > 0.5 - note this is the published
        CLIP-IQA-style definition and differs sharply from the mean of a_q, which exceeds 0.5 for
        essentially no concept; the gate fires decisively on some images rather than weakly on all.
        Right: single-concept AUC, orientation-corrected. Concept separability differs markedly by
        corpus ({spans}), and the direction differs too: CNNSpot's top concepts fire on synthetic
        images, whereas several of SynthCLIC's fire on real ones, meaning the model is partly
        detecting capture artifacts absent from synthetics rather than artifacts present in them.
        This is an illustrative explanation from one trained model - E8 found the population concept
        signal diffuse and seed-sensitive, so individual concept names are qualitative.
    """,
    )
    return {"paths": paths, "table": table, "profiles": profiles, "panels": panels}
