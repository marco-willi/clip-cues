"""The three input spaces of the consolidation, loaded on a common scale.

F1/F3/F4 train the same head under the same recipe on 1024-d pooler, 768-d projected and 168-d cue
features. For that comparison to isolate *the representation*, the recipe's fixed
``weight_decay = 0.01`` must mean the same thing in each space — which requires comparable feature
scale. It is not comparable as cached (mean row norms 32.95 / 18.83 / ~1), and the consequences are
large and measured: see :func:`clip_cues_research.finalexp.features.match_scale`.

So every space is loaded rescaled by **one global scalar** to the pooler train split's mean row
norm. This preserves the geometry exactly and introduces no per-dimension statistics (so no
standardization and no coefficient back-transformation), while making "same weight decay" an
honest statement.

The scale statistic is always measured on the **train** split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from clip_cues_research.finalexp import data as D
from clip_cues_research.finalexp.features import match_scale, mean_row_norm, unit

# SynthBuster+ test is closed under docs/revision_state/EXTERNAL_VALIDATION_PROTOCOL.md
# (one read, executed 2026-07-19). Reads are refused in code, not merely discouraged.
CLOSED_SPLITS = {"synthbuster-plus": {"test"}}


@dataclass
class Space:
    """One scale-matched feature space for one dataset."""

    name: str
    dataset: str
    x: np.ndarray
    df: pd.DataFrame
    scale_factor: float
    target_row_norm: float
    inputs: list[str]

    def split(self, name: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        if name in CLOSED_SPLITS.get(self.dataset, set()):
            raise AssertionError(
                f"REFUSING to read {self.dataset}/{name}: SynthBuster+ test is closed under "
                f"docs/revision_state/EXTERNAL_VALIDATION_PROTOCOL.md (one read, 2026-07-19)."
            )
        m = (self.df["split"] == name).to_numpy()
        return (
            self.x[m],
            self.df.loc[m, "label"].to_numpy().astype(int),
            (self.df[m].reset_index(drop=True)),
        )

    def as_dict(self) -> dict:
        return {
            "space": self.name,
            "dataset": self.dataset,
            "dim": int(self.x.shape[1]),
            "scale_factor": round(self.scale_factor, 6),
            "target_row_norm": round(self.target_row_norm, 4),
            "inputs": self.inputs,
        }


def pooler_target(dataset: str) -> float:
    """The common scale: the pooler train split's mean row norm."""
    frame = D.get_frame(f"pooler/{dataset}", expected_space=D.SPACE_POOLER)
    tr = (frame.df["split"] == "train").to_numpy()
    return mean_row_norm(frame.emb[tr])


def load(dataset: str, kind: str, vocab: str = "antonyms", rescale: bool = True) -> Space:
    """Load one scale-matched space.

    Args:
        dataset: ``synthclic`` / ``cnnspot`` / ``synthbuster-plus``.
        kind: ``pooler`` (1024-d, D_h) · ``projected`` (768-d, D_e) · ``cue`` (named-cue scores).
        vocab: cue vocabulary, for ``kind="cue"``.
        rescale: match the pooler train scale (default). ``False`` reproduces the raw cached
            scale — kept so the scaling effect itself can be re-measured.
    """
    target = pooler_target(dataset)

    if kind == "pooler":
        frame = D.get_frame(f"pooler/{dataset}", expected_space=D.SPACE_POOLER)
        x, df, inputs = frame.emb.astype(np.float64), frame.df, [f"pooler/{dataset}"]
    elif kind == "projected":
        # Raw e = Wp h, NOT unit-normalized: normalizing would change the scale as well as the
        # projection, and D_h vs D_e must differ only by the projection.
        frame = D.get_frame(f"projected/{dataset}", expected_space=D.SPACE_CANON)
        x, df, inputs = frame.emb.astype(np.float64), frame.df, [f"projected/{dataset}"]
    elif kind == "cue":
        # c_j = <e/||e||, v_j> is definitionally a cosine, so the unit normalization here is part
        # of the feature definition; the global rescale afterwards restores comparability.
        frame = D.get_frame(f"projected/{dataset}", expected_space=D.SPACE_CANON)
        x = D.get_npz(f"cue_scores/{dataset}__{vocab}")["scores"].astype(np.float64)
        df, inputs = frame.df, [f"cue_scores/{dataset}__{vocab}", f"vocab/{vocab}"]
    else:
        raise ValueError(f"unknown space {kind!r}")

    if len(x) != len(df):
        raise AssertionError(f"{kind}/{dataset}: feature/metadata length mismatch")

    factor = 1.0
    if rescale:
        tr = (df["split"] == "train").to_numpy()
        before = mean_row_norm(x[tr])
        x = match_scale(x, target, reference=x[tr])
        factor = target / before

    return Space(
        name=kind if kind != "cue" else f"cue:{vocab}",
        dataset=dataset,
        x=x,
        df=df.reset_index(drop=True),
        scale_factor=factor,
        target_row_norm=target,
        inputs=inputs + [f"pooler/{dataset}"],
    )


__all__ = ["Space", "load", "pooler_target", "unit"]
