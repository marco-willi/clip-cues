"""Appendix figure — CLIP-IQA perceptual axes (perceived quality, not pixels).

Visualises the E8 CLIP-IQA result: how well each CLIP-IQA *perceptual* axis (realness, quality,
noisiness, …) separates real vs synthetic, and that it is decoupled from measured pixel statistics. Data
is precomputed in ``outputs/e8/clipiqa/`` (``scripts/analyze/analyze_clipiqa.py``); this only plots it.

- ``{ds}_clipiqa.csv``: clipiqa_axis, real_synth_AUROC, corr_with_detector, abs_AUROC_dev
- ``{ds}_semantic_vs_pixel.csv``: per-axis correlation of the CLIP-IQA percept with the matching
  *measured* pixel statistic (e.g. CLIP-IQA noisiness vs real image-noise) — near zero ⇒ semantic, not pixel.

Caveat (caption): CLIP-IQA axes are CLIP *text-prompt* percepts, not pixel measurements — they describe
*perceived* quality/naturalness, which is why "noisiness" is ⟂ real pixel noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from clip_cues_research.figures.style import color as _color  # noqa: E402
from clip_cues_research.figures.style import config as _cfg  # noqa: E402
from clip_cues_research.figures.style import title_case as _title_case  # noqa: E402

DEFAULT_DIR = "outputs/e8/clipiqa"
POOLER = {
    "synthclic": "data/embeddings/synthclic_clip_large_patch14.pkl",
    "cnnspot": "data/embeddings/cnnspot_clip_large_patch14.pkl",
}
PRETTY = {"synthclic": "SynthCLIC", "cnnspot": "CNNSpot"}


def load_clipiqa(csv_dir: str, ds: str) -> pd.DataFrame:
    df = pd.read_csv(f"{csv_dir}/{ds}_clipiqa.csv").sort_values("abs_AUROC_dev", ascending=False)
    return df.reset_index(drop=True)


def reference_aurocs(ds: str, csv_dir: str = DEFAULT_DIR) -> dict:
    """Reference ceilings so the (weak) single-axis bars are read in context:
    8-axes-combined (from ``{ds}_iqa_detection.txt``) and the full CLIP detector AUROC (logreg on the
    cached 1024-d pooler embeddings, train→eval). The gap (single/8-axes ≪ full) = the diffuse residual."""
    import pickle

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    out: dict = {}
    try:
        txt = Path(f"{csv_dir}/{ds}_iqa_detection.txt").read_text()
        out["combined_8"] = float(txt.split("=")[1])
    except Exception:
        out["combined_8"] = None
    try:
        d = pickle.load(open(POOLER[ds], "rb"))
        df, emb = d["df"], d["embeddings"].astype("float64")
        ev = "test" if (df["split"] == "test").any() else "validation"
        tr, m = (df["split"] == "train").to_numpy(), (df["split"] == ev).to_numpy()
        sc = StandardScaler().fit(emb[tr])
        lr = LogisticRegression(C=1.0, max_iter=5000).fit(
            sc.transform(emb[tr]), df.loc[tr, "label"]
        )
        out["full_detector"] = float(
            roc_auc_score(df.loc[m, "label"], lr.decision_function(sc.transform(emb[m])))
        )
    except Exception:
        out["full_detector"] = None
    return out


def clipiqa_figure(
    datasets: list[str],
    out_dir: str | Path = "outputs/e8/figures",
    *,
    csv_dir: str = DEFAULT_DIR,
    stem: str = "clipiqa_axes",
    also_dirs: list[str | Path] | None = None,
) -> dict:
    """Per-axis real-vs-synth AUROC (centred at 0.5), colour = direction; corr-with-detector annotated."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    tables = {ds: load_clipiqa(csv_dir, ds) for ds in datasets}
    refs = {ds: reference_aurocs(ds, csv_dir) for ds in datasets}
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.6 * len(datasets), 4.8), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        df = tables[ds].sort_values("abs_AUROC_dev", ascending=False).reset_index(drop=True)
        y = np.arange(len(df))[::-1]
        au = df["real_synth_AUROC"].to_numpy()
        fitted = 0.5 + np.abs(
            au - 0.5
        )  # single-axis classifier AUROC (sign-optimal), always >= 0.5
        # direction: synthetic scores higher (synthetic colour) or lower (real colour)
        colors = [_color("synthetic") if v > 0.5 else _color("real") for v in au]
        ax.barh(y, fitted - 0.5, left=0.5, color=colors, edgecolor="0.3")  # bars emanate from 0.5
        for yy, fv in zip(y, fitted):  # label the AUROC value at each bar tip
            ax.text(fv + 0.006, yy, f"{fv:.2f}", va="center", ha="left", fontsize=7, color="0.3")
        ax.set_yticks(y)
        ax.set_yticklabels(df["clipiqa_axis"], fontsize=9)
        ax.axvline(0.5, color="0.4", lw=0.9)
        r = refs[ds]
        if r.get("combined_8") is not None:
            ax.axvline(
                r["combined_8"],
                color=_color("strong"),
                ls=":",
                lw=1.7,
                label=f"8 axes combined ({r['combined_8']:.2f})",
            )
        if r.get("full_detector") is not None:
            ax.axvline(
                r["full_detector"],
                color="0.2",
                ls="--",
                lw=1.7,
                label=f"full CLIP detector ({r['full_detector']:.2f})",
            )
        ax.set_xlim(0.5, 1.04)
        ax.set_xlabel("single-axis classifier AUROC (real vs synthetic)")
        ax.set_title(_title_case(PRETTY.get(ds, ds)), fontsize=10)
        handles = [
            Patch(fc=_color("synthetic"), label="synthetic scores higher"),
            Patch(fc=_color("real"), label="synthetic scores lower"),
        ]
        handles += ax.get_legend_handles_labels()[0]
        # Opaque white frame, drawn above everything: the legend sits inside the axes and the two
        # dashed reference lines run straight through it when the frame is off.
        leg = ax.legend(
            handles=handles,
            fontsize=7.5,
            loc="lower right",
            frameon=True,
            facecolor="white",
            edgecolor="0.8",
            framealpha=1.0,
        )
        leg.set_zorder(10)
    fig.tight_layout()

    targets = [Path(out_dir)] + [Path(x) for x in (also_dirs or [])]
    saved = []
    for t in targets:
        t.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            p = t / f"{stem}.{ext}"
            fig.savefig(p, dpi=200, bbox_inches="tight")
            saved.append(str(p))
    plt.close(fig)
    return {"tables": tables, "saved": saved}


# ── original-manuscript variant: per-attribute score DISTRIBUTIONS ────────────────────────────
# The published appendix figure was a boxplot of the CLIP-IQA attribute distributions by source
# ("Image Attribute Distribution", archived notebook 50), not the per-axis AUROC bars. Both are kept:
# the distributions characterise the datasets, the AUROC bars quantify how weakly each axis separates.
#
# Scores are recomputed rather than read from a table, because `outputs/e8/clipiqa/` stores only
# aggregates. The definition matches `scripts/analyze/analyze_clipiqa.py`:
#
#     c_a(x) = < e(x)/||e(x)|| , v_a >,   v_a = unit( normalize(pos_a) - normalize(neg_a) )
#
# with the eight standard CLIP-IQA prompt pairs (Wang et al. 2022). Image embeddings come from the
# checksummed canonical snapshot and the prompt directions from `vocab_canon/clipiqa_prompts.pt`,
# so this needs no GPU and decodes no images.
IQA_VOCAB = "data/embeddings/vocab_canon/clipiqa_full.pt"
IQA_POLES = "data/embeddings/vocab_canon/clipiqa_full_poles.pt"
#: CLIP's own logit scale, the temperature the published CLIP-IQA softmax uses. Measured on this
#: corpus: at tau=100 the degree spreads across 0.05-0.99 with a healthy IQR; at tau=1 every
#: attribute collapses onto 0.50 +- 0.006 and the figure carries no information.
IQA_TAU = 100.0
#: All three corpora are characterised on **train**. This figure is descriptive -- it says how the
#: datasets look, not how well anything classifies them -- so the training split is the right frame
#: and the largest one available. It also makes the three panels directly comparable, which a
#: test/test/validation mixture was not: SynthBuster+'s test split is frozen
#: (`finalexp.spaces.CLOSED_SPLITS`) and could never have joined the others.
IQA_SPLIT = {"synthclic": "train", "cnnspot": "train", "synthbuster-plus": "train"}
REAL_SOURCE = {"synthclic": "clic2020", "synthbuster-plus": "raise1k"}
#: The four generators SynthCLIC and SynthBuster+ have in common. SynthBuster+ additionally carries
#: nine pre-existing generators (DALL-E 2/3, Firefly, GLIDE, Midjourney, Stable Diffusion 1.x, ...);
#: showing all fourteen is unreadable and, worse, makes the two panels incomparable. Restricting to
#: the shared four lets the same generator be read across corpora.
SHARED_GENERATORS = ("imagen3", "FLUX.1-dev", "FLUX.1-schnell", "SD3-medium")
IQA_SOURCES = {"synthbuster-plus": ("raise1k", *SHARED_GENERATORS)}
SOURCE_LABEL = {
    "clic2020": "CLIC2020 (real)",
    "raise1k": "RAISE-1k (real)",
    "progan": "ProGAN",
    "imagen3": "Imagen 3",
    "FLUX.1-dev": "FLUX.1-dev",
    "FLUX.1-schnell": "FLUX.1-schnell",
    "SD3-medium": "SD3-medium",
}
PRETTY_DS = {"synthclic": "SynthCLIC", "cnnspot": "CNNSpot", "synthbuster-plus": "SynthBuster+"}


def clipiqa_degree(dataset: str, split: str | None = None, tau: float = IQA_TAU) -> pd.DataFrame:
    """Long-form per-image CLIP-IQA **degree** in (0, 1): one row per (image, attribute).

    This is the published CLIP-IQA quantity -- a softmax between the two antonym prompts at CLIP's
    own logit scale::

        degree_a(x) = softmax( tau * [ cos(e, t_a^+) , cos(e, t_a^-) ] )[0]

    It is a strictly increasing function of the signed difference-direction score used elsewhere in
    this repo (``degree = sigmoid(tau * ||t+ - t-|| * score)``), so every ranking statistic -- the
    AUROCs in the companion figure included -- is identical either way. Only the axis differs: a
    bounded 0-1 "degree" reads far better than an unlabelled +-0.15 cosine, and matches how CLIP-IQA
    is reported elsewhere.
    """
    import torch

    from clip_cues_research.finalexp import data as D

    v = torch.load(IQA_VOCAB, weights_only=False)
    poles = np.asarray(torch.load(IQA_POLES, weights_only=False)["embeddings"], dtype=np.float64)
    names = [str(a) for a in v["vocabulary"]]
    n = len(names)
    pos, neg = poles[:n], poles[n:]

    split = split or IQA_SPLIT.get(dataset, "test")
    frame = D.get_frame(f"projected/{dataset}", expected_space=D.SPACE_CANON)
    emb, labels, sub = frame.split(split)
    e = np.asarray(emb, dtype=np.float64)
    e = e / np.clip(np.linalg.norm(e, axis=1, keepdims=True), 1e-12, None)

    z = np.stack([e @ pos.T, e @ neg.T], axis=-1) * tau
    z = z - z.max(axis=-1, keepdims=True)
    ez = np.exp(z)
    degree = ez[..., 0] / ez.sum(axis=-1)

    return pd.DataFrame(
        {
            "dataset": dataset,
            "split": split,
            "source": np.repeat(sub["source"].to_numpy(), n),
            "label": np.repeat(labels, n),
            "attribute": np.tile(names, len(labels)),
            "degree": degree.ravel(),
        }
    )


def _class_names(dataset: str, df: pd.DataFrame) -> tuple[str, str]:
    """``(real label, synthetic label)`` naming each class by what it actually contains.

    The real side is resolved through ``cnnspot_real_source_map`` where the benchmark documents it
    (CNNSpot's ProGAN group pairs **LSUN** photographs), and the synthetic side names the generator
    when there is only one, or counts them when there are several.
    """
    from clip_cues_research.figures.extreme_scores import real_source_map

    corpora = real_source_map()
    real_src = sorted(df.loc[df["label"] == 0, "source"].unique())
    syn_src = sorted(df.loc[df["label"] == 1, "source"].unique())

    named = sorted({corpora[s] for s in real_src if s in corpora})
    if named:
        real = f"Real ({', '.join(named)})"
    elif len(real_src) == 1:
        real = f"Real ({SOURCE_LABEL.get(real_src[0], real_src[0])})"
    else:
        real = "Real"

    if len(syn_src) == 1:
        synth = f"Synthetic ({SOURCE_LABEL.get(syn_src[0], syn_src[0])})"
    else:
        synth = f"Synthetic ({len(syn_src)} generators)"
    return real, synth


def clipiqa_distribution_figure(
    datasets: tuple[str, ...] = ("synthclic", "synthbuster-plus", "cnnspot"),
    out_folder: str = "appendix",
) -> dict:
    """Rebuild the published distribution figure: attribute on x, degree on y, boxes by source."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    from clip_cues_research.figures.style import apply_style, save_figure

    apply_style()
    frames = {}
    for ds in datasets:
        f = clipiqa_degree(ds)
        keep = IQA_SOURCES.get(ds)
        if keep:
            f = f[f["source"].isin(keep)].reset_index(drop=True)
        frames[ds] = f
    attributes = sorted(next(iter(frames.values()))["attribute"].unique())  # lexicographic

    fig, axes = plt.subplots(len(datasets), 1, figsize=(15.0, 3.7 * len(datasets)), squeeze=False)
    for r, ds in enumerate(datasets):
        df = frames[ds]
        ax = axes[r][0]
        # Per-source boxes require two things, not one. The source count must be legible, AND the
        # sources must partition by class -- CNNSpot's train split is ProGAN-only, so `source` takes
        # a single value covering BOTH classes and per-source boxes silently merge real with
        # synthetic. Where either fails, fall back to the real/synthetic contrast the figure is
        # making anyway. Either way the REAL box is first.
        splits_by_class = df.groupby("source")["label"].nunique().max() == 1
        if df["source"].nunique() <= 6 and splits_by_class:
            real = REAL_SOURCE.get(ds)
            gens = sorted(x for x in df["source"].unique() if x != real)
            order = [SOURCE_LABEL.get(x, x) for x in ([real] if real else []) + gens]
            df = df.assign(source=df["source"].map(lambda x: SOURCE_LABEL.get(x, x)))
            df = df.assign(source=pd.Categorical(df["source"], order, ordered=True))
            hue = "source"
        else:
            # CNNSpot's `source` names the evaluation GROUP, not the provenance: on the train split
            # it reads "progan" for both classes, because the real half is LSUN photography paired
            # into the ProGAN group. So the split is made on `label`, and the two boxes are named
            # from what each class actually is.
            real_name, synth_name = _class_names(ds, frames[ds])
            hue, order = "class", [real_name, synth_name]
            df = df.assign(**{"class": np.where(df["label"] == 1, synth_name, real_name)})
        # Two-class panels get the paper's real/synthetic pair; per-source panels get the shared
        # categorical palette, so SynthCLIC and SynthBuster+ colour the same generator identically.
        pal = (
            [_color("real"), _color("synthetic")]
            if hue == "class"
            else sns.color_palette(_cfg()["categorical"], len(order))
        )
        _ = sns.boxplot(
            df,
            x="attribute",
            y="degree",
            hue=hue,
            hue_order=order,
            order=attributes,
            palette=pal,
            ax=ax,
            fliersize=0,
            linewidth=0.7,
        )
        n_real = int((frames[ds]["label"] == 0).sum() // len(attributes))
        n_syn = int((frames[ds]["label"] == 1).sum() // len(attributes))
        _ = ax.set_title(
            f"{PRETTY_DS.get(ds, ds)}  —  {_title_case(df['split'].iloc[0])} Split, "
            f"{n_real:,} Real / {n_syn:,} Synthetic",
            fontsize=10,
            loc="left",
        )
        _ = ax.set_xlabel("")
        _ = ax.set_ylabel("Degree", fontsize=9)
        _ = ax.set_ylim(0, 1)
        _ = ax.tick_params(axis="y", labelsize=8)
        _ = ax.tick_params(axis="x", labelsize=8, rotation=30)
        for lab in ax.get_xticklabels():
            lab.set_ha("right")
        _ = ax.legend(
            title=None,
            fontsize=8,
            ncol=min(len(order), 6),
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0, -0.22),
        )
    _ = fig.tight_layout()

    table = pd.concat(
        [
            f.groupby(["dataset", "split", "source", "attribute"])["degree"]
            .agg(
                n="size",
                q25=lambda x: np.percentile(x, 25),
                median="median",
                q75=lambda x: np.percentile(x, 75),
            )
            .reset_index()
            for f in frames.values()
        ],
        ignore_index=True,
    )
    paths = save_figure(fig, out_folder, "fig9-clipiqa-distributions", table=table)
    plt.close(fig)
    return {"paths": paths, "table": table, "degrees": frames, "attributes": attributes}
