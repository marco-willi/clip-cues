"""Data access for the E9 vocabulary-optimization loop.

Loads the cached SynthCLIC embedding frames (1024-d pooler = detector space, 768-d projected =
cue space), vocabulary embeddings (``{embeddings, vocabulary}`` .pt files), and the paired
real<->synthetic difference vectors (raw ``S - R`` on a chosen frame, the E8 convention from
``scripts/analyze/analyze_paired_shift.py``).

Splits: all fitting/selection uses ``train``; ``validation`` is the loop's reporting split; the
``test`` split stays untouched until the final vocabulary is frozen.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

POOLER_PKL = "data/embeddings/synthclic_clip_large_patch14.pkl"
PROJECTED_PKL = "data/embeddings/synthclic_projected_embeddings.pkl"

# Named vocabularies known to the loop (all in the 768-d cross-modal space).
VOCAB_REGISTRY = {
    "antonyms": "data/embeddings/antonyms_diff_embeddings.pt",
    "antonyms_poles": "data/embeddings/antonyms_embeddings.pt",
    "textspan": "data/embeddings/textspan_embeddings.pt",
}

EVAL_SPLIT = "validation"  # loop reporting split (test frozen until the final vocabulary)


def unit(v: np.ndarray) -> np.ndarray:
    """Row-normalize to unit L2 norm."""
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


@dataclass
class Frame:
    """One embedding frame (df metadata + embedding matrix)."""

    df: pd.DataFrame
    emb: np.ndarray

    def split(self, name: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """(embeddings, labels, sub-df) for one split."""
        m = (self.df["split"] == name).to_numpy()
        sub = self.df[m].reset_index(drop=True)
        return self.emb[m], sub["label"].to_numpy().astype(int), sub


def load_frame(pkl: str) -> Frame:
    d = pickle.load(open(pkl, "rb"))
    return Frame(d["df"].reset_index(drop=True), d["embeddings"].astype(np.float64))


def load_vocab(name_or_path: str) -> tuple[np.ndarray, list[str]]:
    """Load a vocabulary as (unit-row [K,768] array, names). Accepts registry names or .pt paths."""
    path = VOCAB_REGISTRY.get(name_or_path, name_or_path)
    a = torch.load(path, map_location="cpu", weights_only=False)
    emb = unit(np.asarray(a["embeddings"], dtype=np.float64))
    return emb, list(a["vocabulary"])


@dataclass
class Pairs:
    """Paired real<->synthetic differences ``S - R`` per (image_id, generator)."""

    diffs: np.ndarray  # (n_pairs, dim)
    gens: np.ndarray  # (n_pairs,) generator of the synthetic image
    image_ids: np.ndarray  # (n_pairs,)


def load_pairs(frame: Frame, split: str) -> Pairs:
    """Raw S - R diffs for every (image_id, synthetic-generator) pair in a split (E8 convention)."""
    m = (frame.df["split"] == split).to_numpy()
    sub = frame.df[m].reset_index(drop=True)
    E = frame.emb[m]
    diffs, gens, ids = [], [], []
    for iid, grp in sub.groupby("image_id"):
        r = grp[grp.label == 0]
        if len(r) != 1:
            continue
        R = E[r.index[0]]
        for idx, row in grp[grp.label == 1].iterrows():
            diffs.append(E[idx] - R)
            gens.append(row["source"])
            ids.append(iid)
    return Pairs(np.array(diffs), np.array(gens), np.array(ids))


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
