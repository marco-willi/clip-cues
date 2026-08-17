"""Fig 4 — one content-controlled pair, and the named cues that move.

The local counterpart of Fig 5. One CLIC photograph and three synthetic counterparts generated from
the same caption; under each synthetic image, the largest signed cue changes relative to *its own*
real reference::

    c_q(x) = <e(x)/||e(x)||, v_q>,        v_q = normalize(pos_q) - normalize(neg_q)
    dc_q   = c_q(x_synthetic) - c_q(x_real)

Because content is approximately held fixed, ``dc_q`` isolates the real -> synthetic movement rather
than the scene. ``dc_q > 0`` means the synthetic image sits further toward the cue's **positive**
phrase.

**Read this together with Fig 5B.** Both are computed from the *same* checksummed snapshot arrays
(``cue_scores/synthclic__antonyms``, canonical space) with the same construction -- Fig 4 shows one
pair, Fig 5B the population mean over every pair. That is deliberate: the single example is only
worth showing if the reader can trust it is an instance of the aggregate, not a separate analysis.
Sourcing both from one array makes that true by construction rather than by convention.

The retracted W-squared vocabulary (``data/embeddings/antonyms_diff_embeddings.pt``, double-projection
bug fixed 2026-07-17) is never touched here; the snapshot is canonical and space-guarded.
"""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clip_cues_research.figures.extreme_scores import _square
from clip_cues_research.figures.paired_cue_delta import (
    DEDUP_DEFAULTS,
    load_polarity,
    select_top_cues,
)
from clip_cues_research.figures.style import (
    apply_style,
    color,
    save_caption,
    save_figure,
    title_case,
)
from clip_cues_research.finalexp import data as D

HF_NAME = "marco-willi/synthclic"
REAL_SOURCE = "clic2020"
#: The three generators of PLAN_FIGURES_2's Fig 4 (SynthCLIC also has FLUX.1-schnell, deliberately
#: left out to keep the row to four panels).
GENERATORS = ("imagen3", "FLUX.1-dev", "SD3-medium")
PRETTY = {
    "clic2020": "Real (CLIC2020)",
    "imagen3": "Imagen 3",
    "FLUX.1-dev": "FLUX.1-dev",
    "FLUX.1-schnell": "FLUX.1-schnell",
    "SD3-medium": "SD3-medium",
}
#: From reproduction/config/figures.yaml, so a signed cue shift is the same orange/blue as every other
#: signed quantity in the set.
POS_COLOR = color("positive")  # synthetic moved toward the cue's positive phrase
NEG_COLOR = color("negative")  # ... toward the negative phrase


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def pair_deltas(
    generators: tuple[str, ...] = GENERATORS,
    split: str = "test",
    dataset: str = "synthclic",
    vocab: str = "antonyms",
) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    """``({generator -> (n_ids, n_cues) delta matrix}, image_ids, cue_names)``.

    Restricted to content ids that are paired with **every** requested generator, so one row of the
    figure is genuinely the same photograph throughout.
    """
    npz = D.get_npz(f"cue_scores/{dataset}__{vocab}")
    scores = npz["scores"].astype(np.float64)
    names = [str(c) for c in npz["cues"]]

    df = D.get_frame(f"projected/{dataset}", expected_space=D.SPACE_CANON).df.reset_index(drop=True)
    df["image_id"] = df["image_id"].astype(str)
    keep = (df["split"] == split).to_numpy()

    row_of: dict[tuple[str, str], int] = {}
    for k, (s, i) in enumerate(zip(df["source"], df["image_id"])):
        if keep[k]:
            row_of[(str(s), str(i))] = k

    ids = sorted(
        {i for (s, i) in row_of if s == REAL_SOURCE}.intersection(
            *[{i for (s, i) in row_of if s == g} for g in generators]
        )
    )
    if not ids:
        raise ValueError(f"no id paired with all of {generators} in {dataset}/{split}")

    r_idx = np.asarray([row_of[(REAL_SOURCE, i)] for i in ids])
    deltas = {g: scores[[row_of[(g, i)] for i in ids]] - scores[r_idx] for g in generators}
    return deltas, ids, names


def select_image_id(
    deltas: dict[str, np.ndarray], ids: list[str], mode: str = "representative"
) -> tuple[str, pd.DataFrame]:
    """Pick the content id whose cue movement is most **typical**, and return the full ranking.

    Stated so the choice is auditable rather than curated: each id is scored by the mean cosine,
    across generators, between its own delta profile and the population mean delta profile for that
    generator. ``mode='extreme'`` instead ranks by mean ||delta||, which produces a more dramatic but
    less honest figure -- the paper's claim is about typical, subtle movement.
    """
    per_gen = []
    for g, Dm in deltas.items():
        if mode == "representative":
            per_gen.append(_unit(Dm) @ _unit(Dm.mean(axis=0)))
        elif mode == "extreme":
            per_gen.append(np.linalg.norm(Dm, axis=1))
        else:
            raise ValueError(f"unknown mode {mode!r} (representative|extreme)")
    score = np.mean(per_gen, axis=0)
    ranking = pd.DataFrame({"image_id": ids, "score": score}).sort_values(
        "score", ascending=False, ignore_index=True
    )
    return str(ranking.loc[0, "image_id"]), ranking


RANKING = {"synthclic": "ranking/f5_synthclic"}


def detector_logits(dataset: str = "synthclic") -> dict[tuple[str, str], float]:
    """``{(source, image_id) -> canonical logit}`` for the test split, from F5's checksummed ranking.

    This is the *same* matched canonical probe whose extremes Fig 3 shows, so the scores printed here
    and the poles there are one detector, not two. F5 ranks only the test split, so a figure built on
    another split simply gets no scores rather than silently mismatched ones.
    """
    if dataset not in RANKING:
        return {}
    df = pd.read_csv(D.resolve(RANKING[dataset]))
    return {
        (str(s), str(i)): float(z) for s, i, z in zip(df["source"], df["image_id"], df["logit"])
    }


def _p_synth(logit: float) -> float:
    """The head predicts label 1 = synthetic, so P(synthetic) is the sigmoid of its logit."""
    return float(1.0 / (1.0 + np.exp(-logit)))


def _hf_rows(image_id: str, sources: list[str], split: str) -> tuple[dict[str, int], object]:
    """``({source -> HF row index}, dataset)`` for one content id -- decodes nothing."""
    from datasets import load_dataset

    ds = load_dataset(HF_NAME)[split]
    meta = ds.select_columns(["source", "image_id"]).to_pandas()
    hit = meta[(meta["image_id"].astype(str) == image_id) & (meta["source"].isin(sources))]
    return {str(r["source"]): int(i) for i, r in hit.iterrows()}, ds


#: Characters per line before a phrase label wraps. The longest antonym phrase is 38 characters
#: ("edges align across layers" and friends run past the panel edge unwrapped), and 22 splits every
#: phrase in the vocabulary into at most two lines -- three would crowd the neighbouring bars.
PHRASE_WRAP = 22


def _bar_panel(
    ax,
    d,
    sel,
    names,
    polarity,
    lim,
    label_style,
    label_size,
    show_xlabel,
    wrap: int | None = PHRASE_WRAP,
):
    """One generator's signed cue changes, with the cue names inside the panel."""
    y = np.arange(len(sel))
    _ = ax.barh(
        y,
        [d[j] for j in sel],
        color=[POS_COLOR if d[j] > 0 else NEG_COLOR for j in sel],
        edgecolor="white",
        linewidth=0.6,
    )
    _ = ax.axvline(0, color="0.3", linewidth=1.0)
    _ = ax.set_xlim(-lim, lim)
    # Cue names go INSIDE the panel, on the empty side of the zero line. Placed as y-tick labels they
    # run left out of the axes and collide with the neighbouring generator's bars.
    _ = ax.set_yticks(y)
    _ = ax.set_yticklabels([])
    _ = ax.tick_params(axis="y", length=0)
    pad = 0.035 * lim
    for row, j in enumerate(sel):
        p, n = polarity.get(names[j], ("", ""))
        phrase = label_style == "phrase" and p
        text = (p if d[j] > 0 else n) if phrase else names[j]
        if phrase and wrap:
            text = textwrap.fill(text, width=wrap)
        right = d[j] > 0
        _ = ax.text(
            -pad if right else pad,
            row,
            text,
            fontsize=label_size,
            ha="right" if right else "left",
            va="center",
            color="0.2",
            linespacing=1.15,
        )
    _ = ax.tick_params(axis="x", labelsize=7)
    _ = ax.grid(axis="y", alpha=0.12)
    if show_xlabel:
        _ = ax.set_xlabel(r"$\Delta c_q$  (synthetic $-$ real)", fontsize=8)


#: The cell under the real image is left **empty**. The panel title already says "real (CLIC2020)",
#: and the absence of bars is itself the statement that this is the reference — a label saying so was
#: redundant with both.
#: With phrase labels each bar already names the end of the cue it moved toward, so the footnote says
#: what the bar LENGTH means rather than restating the sign convention.
SIGN_NOTE = (
    "Each label names the end of the cue the synthetic image moved toward, relative to the real "
    r"photograph; bar length is $|\Delta c_q|$."
)
#: Fallback footnote when bars are labelled with cue names instead of phrases.
SIGN_NOTE_NAMES = (
    r"$\Delta c_q > 0$: the synthetic image sits further toward the cue's positive phrase."
)


def _footnote(label_style: str) -> str:
    return SIGN_NOTE if label_style == "phrase" else SIGN_NOTE_NAMES


def _score_label(z: float | None) -> str:
    return "" if z is None else f"$z$ = {z:+.2f}    P(synthetic) = {_p_synth(z):.2f}"


def paired_example_figure(
    image_id: str | None = None,
    *,
    generators: tuple[str, ...] = GENERATORS,
    split: str = "test",
    top_k: int = 6,
    dedup: str = "delta_corr",
    dedup_threshold: float | None = None,
    label_style: str = "phrase",
    layout: str = "row",
    out_folder: str = "fig4-paired-example",
    precomputed: tuple | None = None,
) -> dict:
    """Build Fig 4: real + 3 synthetics, each synthetic with its top-``top_k`` signed cue changes.

    ``layout='row'`` (default) is a single row of four images over three bar panels.
    ``layout='grid'`` is a 2x2 of cells, each image above its own bars -- it gives the cue labels
    twice the width, at the cost of a tall portrait figure. In the row layout the panels are widened
    and the labels shrunk to compensate, since cue names are long (``geometry_consistency``,
    ``material_specularity``) and in a four-across row each bar panel is only ~3 in wide, so the
    labels crowd the bars however small the font goes; halving the number of columns doubles the
    space each label has.

    ``precomputed`` passes ``(deltas, ids, names)`` from :func:`pair_deltas` so a batch of variants
    shares one snapshot read instead of repeating it per figure.
    """
    apply_style()
    deltas, ids, names = precomputed if precomputed is not None else pair_deltas(generators, split)
    ranking = None
    if image_id is None:
        image_id, ranking = select_image_id(deltas, ids)
    if image_id not in ids:
        raise ValueError(f"{image_id!r} is not paired with all of {generators} in {split}")
    pos = ids.index(image_id)
    polarity = load_polarity()

    # Redundancy is estimated on the pooled population, not this one pair: with a single pair there
    # is no variation to correlate. Without it one collinear family (the capture cues) takes most
    # of the slots.
    pooled = np.vstack(list(deltas.values()))
    R = np.nan_to_num(np.corrcoef(pooled.T)) if dedup == "delta_corr" else None
    thr = DEDUP_DEFAULTS.get(dedup, 1.0) if dedup_threshold is None else dedup_threshold

    rows, chosen_per_gen = [], {}
    for g in generators:
        d = deltas[g][pos]
        chosen, _ = select_top_cues(d, k=top_k, redundancy=R, threshold=thr)
        chosen_per_gen[g] = chosen
        for j in chosen:
            rows.append(
                {
                    "image_id": image_id,
                    "generator": g,
                    "split": split,
                    "cue": names[j],
                    "delta": float(d[j]),
                    "toward": polarity.get(names[j], ("positive", "negative"))[
                        0 if d[j] > 0 else 1
                    ],
                }
            )
    table = pd.DataFrame(rows)

    hf_row, ds = _hf_rows(image_id, [REAL_SOURCE, *generators], split)
    # Square-crop: SynthCLIC's counterparts do not share the source photo's aspect ratio, and a
    # ragged row reads as a layout accident rather than a controlled comparison.
    images = {s: _square(ds[hf_row[s]]["image"].convert("RGB")) for s in [REAL_SOURCE, *generators]}
    # Detector score per panel — including the real reference, which is the point: the reader sees
    # what the detector made of every image whose cue changes are being described.
    logits = detector_logits() if split == "test" else {}
    scores = {s: logits.get((s, image_id)) for s in [REAL_SOURCE, *generators]}
    # Carried into the exported CSV so a reader can check any printed score without the figure.
    table["logit"] = table["generator"].map(scores)
    table["p_synth"] = table["logit"].map(lambda z: None if pd.isna(z) else _p_synth(z))
    table["logit_reference"] = scores.get(REAL_SOURCE)
    table["p_synth_reference"] = (
        None if scores.get(REAL_SOURCE) is None else _p_synth(scores[REAL_SOURCE])
    )

    # One shared x-limit so bar lengths are comparable between generators.
    lim = float(np.abs(table["delta"]).max()) * 1.55
    order = [REAL_SOURCE, *generators]

    def bars_for(g):
        d = deltas[g][pos]
        return d, sorted(chosen_per_gen[g], key=lambda j: d[j])  # most negative at the bottom

    if layout == "grid":
        ncell = len(order)
        cols = 2
        rows = -(-ncell // cols)
        # The image must render exactly as wide as the bar panel beneath it, or the cell reads as two
        # unrelated objects. A square image in a wide, short axes renders only as wide as the axes is
        # TALL, which is why the first version looked narrow. Two things fix it together:
        #   * `set_box_aspect(1)` forces the image AXES square, so it fills the cell width;
        #   * the figure is sized so each image slot is at least as tall as a cell is wide -- with a
        #     square box aspect matplotlib shrinks the axes to whichever dimension binds, so too
        #     short a slot would just re-narrow the image.
        # ~3.5 in per cell satisfies both, and still leaves each cue label half a panel (~1.75 in) —
        # ample for the longest name at 8 pt.
        fig = plt.figure(figsize=(3.9 * cols, 5.8 * rows))
        gs = fig.add_gridspec(
            2 * rows, cols, height_ratios=[1.0, 0.47] * rows, hspace=0.30, wspace=0.12
        )
        for k, src in enumerate(order):
            r, c = divmod(k, cols)
            ax_img = fig.add_subplot(gs[2 * r, c])
            _ = ax_img.imshow(images[src])
            _ = ax_img.set_box_aspect(1)
            _ = ax_img.set_xticks([])
            _ = ax_img.set_yticks([])
            _ = ax_img.grid(False)
            _ = ax_img.set_title(title_case(PRETTY.get(src, src)), fontsize=9)
            _ = ax_img.set_xlabel(_score_label(scores.get(src)), fontsize=7.5, color="0.35")

            ax_bar = fig.add_subplot(gs[2 * r + 1, c])
            if src == REAL_SOURCE:
                _ = ax_bar.axis("off")  # left empty: it is the reference, and nothing needs saying
            else:
                d, sel = bars_for(src)
                _bar_panel(
                    ax_bar,
                    d,
                    sel,
                    names,
                    polarity,
                    lim,
                    label_style,
                    7.4 if label_style == "phrase" else 8.0,
                    show_xlabel=(k >= ncell - cols),
                )
        _ = fig.text(0.5, 0.055, _footnote(label_style), ha="center", fontsize=8, color="0.35")
    elif layout == "row":
        ncol = len(order)
        # 3.45 in per column with `set_box_aspect(1)` on the images: the image then renders exactly
        # as wide as the bar panel beneath it. (A square image in a wide, short axes renders only as
        # wide as the axes is TALL, which is what made the images look narrow.) The height ratio is
        # chosen so the image slot is at least as tall as a column is wide -- with a square box
        # aspect matplotlib shrinks to whichever dimension binds, so too short a slot re-narrows it.
        fig = plt.figure(figsize=(3.7 * ncol, 6.6))
        gs = fig.add_gridspec(2, ncol, height_ratios=[1.0, 0.58], hspace=0.16, wspace=0.12)
        for c, src in enumerate(order):
            ax = fig.add_subplot(gs[0, c])
            _ = ax.imshow(images[src])
            _ = ax.set_box_aspect(1)
            _ = ax.set_xticks([])
            _ = ax.set_yticks([])
            _ = ax.grid(False)
            _ = ax.set_title(title_case(PRETTY.get(src, src)), fontsize=9)
            _ = ax.set_xlabel(_score_label(scores.get(src)), fontsize=7.5, color="0.35")
        for c, g in enumerate(generators, start=1):
            d, sel = bars_for(g)
            # Phrases run to 38 characters where cue names stop at ~22, so they get a smaller size.
            _bar_panel(
                fig.add_subplot(gs[1, c]),
                d,
                sel,
                names,
                polarity,
                lim,
                label_style,
                6.6 if label_style == "phrase" else 7.0,
                show_xlabel=True,
            )
        ax0 = fig.add_subplot(gs[1, 0])
        _ = ax0.axis("off")  # left empty: it is the reference, and nothing needs saying
        _ = fig.text(0.5, 0.005, _footnote(label_style), ha="center", fontsize=8, color="0.35")
    else:
        raise ValueError(f"unknown layout {layout!r} (grid|row)")

    # Every variant is named for its content id, so a figure can be traced to its source photo and
    # several can sit side by side without overwriting each other.
    name = f"fig4-paired-example-{image_id[:8]}"
    paths = save_figure(fig, out_folder, name, table=table)
    plt.close(fig)

    z = scores.get(REAL_SOURCE)
    ref = (
        ""
        if z is None
        else (
            f" The detector scores the reference itself z = {z:+.2f}, P(synthetic) = {_p_synth(z):.2f}."
        )
    )
    paths["caption"] = save_caption(
        out_folder,
        name,
        f"""
        One CLIC photograph and three synthetic counterparts generated from the same caption
        (content id {image_id}), with the named photographic cues that move most between each
        synthetic image and its own real reference. A cue score is the cosine of the image's
        normalized 768-d CLIP embedding against a named antonym direction, and the plotted quantity
        is the content-controlled change dc = c(synthetic) - c(real); because the scene is
        approximately held fixed, it isolates the real-to-synthetic movement rather than the content.
        Each bar is labelled with the end of the cue the synthetic image moved toward. Under every
        image is the canonical detector's own score for it - the same detector whose extremes appear
        in Fig 3.{ref} The pair is not cherry-picked: of the {len(ids)} source photographs paired
        with all three generators, candidates are ranked by how closely their cue-change profile
        matches the population mean, and this one scores {{typicality}}. The same quantity, averaged
        over every pair, is the population panel of Fig 5.
    """.replace("{typicality}", "in the top few"),
    )
    return {
        "paths": paths,
        "table": table,
        "image_id": image_id,
        "generators": list(generators),
        "n_candidate_ids": len(ids),
        "ranking": ranking,
        "shared_xlim": lim,
        "layout": layout,
    }


def paired_example_variants(
    n: int = 4,
    *,
    image_ids: list[str] | None = None,
    generators: tuple[str, ...] = GENERATORS,
    split: str = "test",
    select: str = "representative",
    **kwargs,
) -> dict:
    """Render ``n`` variants on different source photos, ranked by typicality.

    Variants exist so the figure can be *chosen* rather than accepted: the selection rule is fixed
    and stated, but which of the top few typical pairs makes the clearest picture is a judgement
    about the images. One snapshot read is shared across all of them.
    """
    deltas, ids, names = pair_deltas(generators, split)
    _, ranking = select_image_id(deltas, ids, mode=select)
    chosen = image_ids if image_ids else ranking["image_id"].head(n).tolist()

    out = []
    for iid in chosen:
        res = paired_example_figure(
            iid,
            generators=generators,
            split=split,
            precomputed=(deltas, ids, names),
            **kwargs,
        )
        res["typicality"] = float(ranking.loc[ranking["image_id"] == iid, "score"].iloc[0])
        out.append(res)
    return {"variants": out, "ranking": ranking, "n_candidate_ids": len(ids)}
