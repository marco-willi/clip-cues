"""Figure — per-pair real↔synthetic images next to that pair's top-k **cue score changes**.

The per-pair counterpart of the population-level ``paired_cue_shift`` figure and of ``paired_montage``
(which ranks many pairs by *one* cue). Here the content id is fixed and the *cues* vary: each row shows
the real photo, its synthetic counterpart from the same SynthCLIC ``image_id``, and the k cues whose
score moves most between the two.

Cue score of an image is the cosine of its projected 768-d CLIP embedding onto a canonical antonym cue
direction, and the plotted quantity is the content-controlled change::

    s_j(x) = unit(emb(x)) · unit(cue_j)
    Δ_j    = s_j(synthetic) − s_j(real)

Because the two images share content, Δ isolates the real→synthetic shift rather than the scene.

Vocabulary is the **canonical** antonym set (``data/embeddings/vocab_canon/antonyms.pt``), i.e. the
post-double-projection-fix embeddings used by the E12 score-alignment work — *not* the older
``antonyms_diff_embeddings.pt``, which lives in the double-projected W² space. Each cue vector is
``normalize(pos) − normalize(neg)``, so Δ_j > 0 means the synthetic image sits further toward the cue's
**positive** phrase.

Image ids default to the concept model's analysis/visualisation example (Fig 1's
``concept_explanation.DEFAULT_IMAGE_ID``) plus further ids picked from the same split — by default the
most *representative* pairs (whose Δ profile aligns best with the population mean Δ), so the panel is
not a cherry-picked extreme.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clip_cues_research.figures.style import color

#: Content id of the published paired example (was defined in the retired concept_explanation
#: module, inlined here so no figure depends on it).
DEFAULT_IMAGE_ID = "c7737e81ede69577d2e97baf5546fa4c"

PROJ = "data/embeddings/synthclic_projected_embeddings.pkl"
VOCAB_CANON = "data/embeddings/vocab_canon/antonyms.pt"
VOCAB_CSV = "data/vocabularies/antonyms.csv"
HF_NAME = "marco-willi/synthclic"
REAL_SOURCE = "clic2020"

POS_COLOR = color("positive")  # synthetic moves toward the cue's positive phrase
NEG_COLOR = color("negative")  # synthetic moves toward the cue's negative phrase


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def load_cue_basis(vocab: str | Path = VOCAB_CANON) -> tuple[list[str], np.ndarray]:
    """Canonical antonym cue names + unit-norm difference directions (168, 768)."""
    v = torch.load(vocab, weights_only=False)
    return list(v["vocabulary"]), _unit(np.asarray(v["embeddings"], dtype=np.float64))


def load_polarity(vocab_csv: str | Path = VOCAB_CSV) -> dict[str, tuple[str, str]]:
    """``attribute_name -> (positive phrase, negative phrase)``; empty dict if unavailable."""
    try:
        df = pd.read_csv(vocab_csv)
    except Exception:
        return {}
    return {r["attribute_name"]: (r["positive"], r["negative"]) for _, r in df.iterrows()}


def pair_cue_deltas(
    *,
    gen: str,
    split: str = "test",
    proj_emb: str | Path = PROJ,
    vocab: str | Path = VOCAB_CANON,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Δ cue scores for every real↔``gen`` pair in ``split``.

    Returns ``(pairs, D, cue_names)`` where ``pairs`` has one row per content id (``image_id`` plus the
    embedding row indices) and ``D`` is the aligned ``(n_pairs, n_cues)`` matrix of Δ scores.
    """
    names, C = load_cue_basis(vocab)
    d = pickle.load(open(proj_emb, "rb"))
    df = d["df"].reset_index(drop=True)
    df["image_id"] = df["image_id"].astype(str)
    emb = _unit(np.asarray(d["embeddings"], dtype=np.float64))

    sub = df[df["split"] == split]
    real = sub[sub["source"] == REAL_SOURCE].set_index("image_id")
    synth = sub[sub["source"] == gen].set_index("image_id")
    ids = sorted(set(real.index) & set(synth.index))
    if not ids:
        raise ValueError(f"no {REAL_SOURCE}↔{gen} pairs in split={split!r}")

    # positional rows in the embedding matrix (embeddings align with df row order)
    row_of = {(s, i): k for k, (s, i) in enumerate(zip(df["source"], df["image_id"]))}
    r_idx = np.asarray([row_of[(REAL_SOURCE, i)] for i in ids])
    s_idx = np.asarray([row_of[(gen, i)] for i in ids])

    D = (emb[s_idx] - emb[r_idx]) @ C.T  # (n_pairs, n_cues)
    pairs = pd.DataFrame({"image_id": ids, "real_row": r_idx, "synth_row": s_idx})
    return pairs, D, names


def select_image_ids(
    pairs: pd.DataFrame,
    D: np.ndarray,
    *,
    n: int,
    mode: str = "representative",
    pinned: list[str] | None = None,
) -> list[str]:
    """Pick ``n`` content ids, keeping ``pinned`` ids first.

    ``representative`` ranks pairs by cosine of their Δ profile to the population mean Δ (typical
    shifts); ``extreme`` ranks by ‖Δ‖ (largest total movement).
    """
    pinned = [p for p in (pinned or []) if p in set(pairs["image_id"])]
    if mode == "extreme":
        score = np.linalg.norm(D, axis=1)
    elif mode == "representative":
        score = _unit(D) @ _unit(D.mean(axis=0))
    else:
        raise ValueError(f"unknown mode {mode!r} (representative|extreme)")

    ranked = [pairs["image_id"].iloc[i] for i in np.argsort(-score)]
    out = list(pinned)
    for iid in ranked:
        if len(out) >= n:
            break
        if iid not in out:
            out.append(iid)
    return out[:n]


#: default redundancy cutoff per metric — see ``redundancy_matrix`` for why they differ
DEDUP_DEFAULTS = {"delta_corr": 0.5, "cosine": 0.35}


def redundancy_matrix(
    D: np.ndarray, C: np.ndarray, metric: str = "delta_corr"
) -> np.ndarray | None:
    """Cue×cue redundancy used to suppress near-duplicate bars (``None`` disables deduplication).

    ``delta_corr`` correlates the cues' Δ columns across the pair population — the *empirical* answer
    to "do these two cues carry the same evidence here". ``cosine`` uses the fixed geometry of the cue
    directions instead, which is data-independent but a weaker separator: the capture-cue family
    ('dslr_cues', 'mirrorless_cues', 'cctv_cues', …) reaches |cos| 0.63 against an overall median of
    0.08, but |Δ-corr| 0.82 against a median of 0.13 — hence the different default thresholds.
    """
    if metric == "none":
        return None
    if metric == "delta_corr":
        R = np.corrcoef(D.T)
        return np.nan_to_num(R)
    if metric == "cosine":
        return C @ C.T
    raise ValueError(f"unknown redundancy metric {metric!r} (delta_corr|cosine|none)")


def select_top_cues(
    delta: np.ndarray,
    *,
    k: int,
    redundancy: np.ndarray | None = None,
    threshold: float = 0.5,
) -> tuple[list[int], list[tuple[int, int, float]]]:
    """Greedy top-``k`` cue indices by |Δ|, skipping cues redundant with an already-selected one.

    Walking |Δ| in descending order, a cue is accepted only if its redundancy against every accepted
    cue stays at or below ``threshold``; otherwise it is suppressed and attributed to the strongest
    cue that blocked it. Returns ``(chosen, suppressed)`` where ``suppressed`` holds
    ``(cue_idx, blocked_by_idx, redundancy)`` — kept so the figure can report what it hid rather than
    silently dropping cues that moved just as much.
    """
    chosen: list[int] = []
    suppressed: list[tuple[int, int, float]] = []
    for j in np.argsort(np.abs(delta))[::-1]:
        if len(chosen) >= k:
            break
        if redundancy is None or not chosen:
            chosen.append(int(j))
            continue
        r = np.abs(redundancy[j, chosen])
        b = int(np.argmax(r))
        if r[b] > threshold:
            suppressed.append((int(j), chosen[b], float(r[b])))
        else:
            chosen.append(int(j))
    return chosen, suppressed


def paired_cue_delta_figure(
    image_ids: list[str] | None = None,
    *,
    gen: str = "imagen3",
    split: str = "test",
    top_k: int = 8,
    n_images: int = 3,
    select: str = "representative",
    pin_concept_example: bool = True,
    label_style: str = "name",
    dedup: str = "delta_corr",
    dedup_threshold: float | None = None,
    out_dir: str | Path = "outputs/e8/figures",
    also_dirs: list[str | Path] | None = None,
    proj_emb: str | Path = PROJ,
    vocab: str | Path = VOCAB_CANON,
    vocab_csv: str | Path = VOCAB_CSV,
    hf_name: str = HF_NAME,
    cache_dir: str = "data/hf_cache",
    stem: str | None = None,
    panel_width: float = 3.0,
    dpi: int = 200,
) -> dict:
    """Build + save the per-pair cue-change grid.

    Rows = content ids; columns = real image | synthetic image | top-``top_k`` Δ cue scores. Saves PNG
    and PDF, plus a tidy CSV of every plotted Δ and a second CSV of the cues ``dedup`` suppressed.
    ``label_style='phrase'`` labels the bars with the antonym phrase the synthetic image moved toward
    (Fig 1 convention) instead of the cue name.

    ``dedup`` keeps the bars mutually distinct: without it a single collinear cue family (the capture
    cues) can occupy 7 of 8 slots. See ``redundancy_matrix``; ``dedup='none'`` restores the plain
    top-``top_k``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from datasets import load_dataset

    pairs, D, names = pair_cue_deltas(gen=gen, split=split, proj_emb=proj_emb, vocab=vocab)
    polarity = load_polarity(vocab_csv)
    _, C = load_cue_basis(vocab)
    R = redundancy_matrix(D, C, dedup)
    thr = DEDUP_DEFAULTS.get(dedup, 1.0) if dedup_threshold is None else dedup_threshold

    if image_ids:
        missing = [i for i in image_ids if i not in set(pairs["image_id"])]
        if missing:
            raise ValueError(f"image_id(s) not paired with {gen} in split={split}: {missing}")
        ids = list(image_ids)
    else:
        ids = select_image_ids(
            pairs,
            D,
            n=n_images,
            mode=select,
            pinned=[DEFAULT_IMAGE_ID] if pin_concept_example else None,
        )

    pos_of = {iid: k for k, iid in enumerate(pairs["image_id"])}

    # image pixels: read metadata columns only (decoding every row is very slow)
    ds = load_dataset(hf_name, cache_dir=cache_dir)[split]
    meta = ds.select_columns(["source", "image_id"])
    hf_row = {(r["source"], str(r["image_id"])): i for i, r in enumerate(meta)}

    n = len(ids)
    fig, axes = plt.subplots(
        n,
        3,
        figsize=(3.6 * panel_width, 1.15 * panel_width * n),
        gridspec_kw={"width_ratios": [1, 1, 1.6]},
    )
    if n == 1:
        axes = axes[None, :]

    records, hidden = [], []
    for r, iid in enumerate(ids):
        delta = D[pos_of[iid]]
        chosen, dropped = select_top_cues(delta, k=top_k, redundancy=R, threshold=thr)
        top = np.asarray(chosen[::-1], dtype=int)  # smallest |Δ| at the bottom of the bars
        for j, blocker, red in dropped:
            hidden.append(
                {
                    "image_id": iid,
                    "generator": gen,
                    "cue": names[j],
                    "delta": float(delta[j]),
                    "redundant_with": names[blocker],
                    "redundancy": red,
                }
            )

        for c, src in ((0, REAL_SOURCE), (1, gen)):
            ax = axes[r, c]
            k = hf_row.get((src, iid))
            if k is not None:
                ax.imshow(np.asarray(ds[k]["image"].convert("RGB")))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title("real (CLIC2020)" if c == 0 else f"synthetic ({gen})", fontsize=10)
        axes[r, 0].set_ylabel(iid[:8], fontsize=8)

        axb = axes[r, 2]
        colors = [POS_COLOR if delta[j] > 0 else NEG_COLOR for j in top]
        axb.barh(np.arange(len(top)), delta[top], color=colors, edgecolor="0.3")
        axb.set_yticks(np.arange(len(top)))
        labels = []
        for j in top:
            pos_phrase, neg_phrase = polarity.get(names[j], ("", ""))
            if label_style == "phrase" and pos_phrase:
                labels.append(pos_phrase if delta[j] > 0 else neg_phrase)
            else:
                labels.append(names[j])
        axb.set_yticklabels(labels, fontsize=8)
        axb.axvline(0, color="0.4", lw=0.8)
        axb.tick_params(axis="x", labelsize=7)
        if r == n - 1:
            axb.set_xlabel("Δ cue score  (synthetic − real)", fontsize=8)
        if r == 0:
            note = "" if R is None else f", |{dedup}| ≤ {thr:g}"
            axb.set_title(f"top-{top_k} cue changes{note}", fontsize=10)

        for j in top[::-1]:
            records.append(
                {
                    "image_id": iid,
                    "generator": gen,
                    "split": split,
                    "cue": names[j],
                    "delta": float(delta[j]),
                    "toward": polarity.get(names[j], ("positive", "negative"))[
                        0 if delta[j] > 0 else 1
                    ],
                }
            )

    fig.suptitle(
        f"Content-controlled cue changes, real → synthetic ({gen}, {split} split)",
        fontsize=11,
        y=0.997,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))

    stem = stem or f"paired_cue_delta_{gen.replace('.', '').replace('-', '')}"
    table = pd.DataFrame(records)
    suppressed = pd.DataFrame(hidden)
    saved = []
    for dd in [out_dir, *(also_dirs or [])]:
        Path(dd).mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            p = Path(dd) / f"{stem}.{ext}"
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            saved.append(str(p))
        p = Path(dd) / f"{stem}.csv"
        table.to_csv(p, index=False)
        saved.append(str(p))
        if not suppressed.empty:
            p = Path(dd) / f"{stem}_suppressed.csv"
            suppressed.to_csv(p, index=False)
            saved.append(str(p))
    plt.close(fig)

    return {
        "image_ids": ids,
        "gen": gen,
        "split": split,
        "n_pairs": len(pairs),
        "dedup": dedup,
        "dedup_threshold": thr,
        "table": table,
        "suppressed": suppressed,
        "saved": saved,
    }
