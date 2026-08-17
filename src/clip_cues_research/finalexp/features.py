"""Feature spaces for F1-F7: 1024-d pooler, 768-d projected, 168-d cue scores.

The three input spaces of the consolidation. F1/F3/F4 train the *same* head under the *same* recipe
on these three, so the only difference between them is the representation.

**The 768-d space is derived, not separately extracted** — ``e = Wp h`` — for three reasons:

1. It is the project's frozen convention: EXTERNAL_VALIDATION_PROTOCOL.md mandates the
   "both-sides-derived rule" (``pooler @ visual_projection^T``) with a recorded sanity delta
   <= 0.003, and E11b fitted its cross-dataset probes this way.
2. It is exact: HF's ``CLIPVisionModelWithProjection`` computes
   ``image_embeds = visual_projection(pooler_output)``.
3. It is the scientifically correct choice here: ``D_h`` and ``D_e`` then see literally the same
   image representation and differ *only* by the projection, which is the claim under test.
"""

from __future__ import annotations

import numpy as np


def unit(v: np.ndarray) -> np.ndarray:
    """Row-normalize to unit L2 norm."""
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def project(pooler: np.ndarray, wp: np.ndarray) -> np.ndarray:
    """1024-d pooler -> 768-d shared image-text space: ``e = Wp h``.

    Args:
        pooler: ``(n, 1024)`` frozen ``pooler_output``.
        wp: ``(768, 1024)`` CLIP ``visual_projection`` weight.

    Returns:
        ``(n, 768)`` un-normalized projected embeddings.
    """
    if pooler.shape[1] != wp.shape[1]:
        raise ValueError(f"pooler dim {pooler.shape[1]} != Wp input dim {wp.shape[1]}")
    return np.asarray(pooler, dtype=np.float64) @ np.asarray(wp, dtype=np.float64).T


def cue_scores(projected: np.ndarray, vocab: np.ndarray) -> np.ndarray:
    """Named-cue features ``c_j = <e/||e||, v_j>`` for a unit-row cue basis.

    Args:
        projected: ``(n, 768)`` projected embeddings (normalized internally).
        vocab: ``(k, 768)`` unit-row cue directions.

    Returns:
        ``(n, k)`` cue scores — the restricted-information feature space of F4.
    """
    return unit(np.asarray(projected, dtype=np.float64)) @ np.asarray(vocab, dtype=np.float64).T


def mean_row_norm(x: np.ndarray) -> float:
    """Mean L2 row norm — the scale statistic the matched recipe turns out to be sensitive to."""
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64), axis=1).mean())


def match_scale(
    x: np.ndarray, target_row_norm: float, reference: np.ndarray | None = None
) -> np.ndarray:
    """Rescale a feature space by a **single global scalar** so its mean row norm hits a target.

    Why this is necessary, and why it is not "standardization" (measured 2026-08-08):

    The matched recipe fixes ``weight_decay = 0.01`` (coupled L2). That is only "the same
    regularization" if the feature spaces have comparable scale — and they do not. On SynthCLIC the
    mean row norms are pooler **32.95**, projected ``Wp h`` **18.83**, unit-normalized **1.00**. On
    unit-normalized 768-d features the same recipe reaches AUROC 0.725 versus 0.888 on the raw
    projected ones, and *more epochs do not close it* (0.744 at 212 epochs): the penalty simply
    dominates the likelihood term. The give-away is that F4's 168-cue **restricted** probe scored
    0.800 — above the 768-d "unrestricted" probe it is a strict subspace of, which is impossible
    unless the latter is under-trained.

    So holding ``wd`` fixed across differently-scaled inputs silently varies the effective
    regularization, which would make the F3/F4 comparisons measure optimization artefacts rather
    than representational capacity. One global scalar per space fixes that while preserving the
    geometry exactly: it is a rescale, **not** a per-dimension standardization (no per-feature means
    or variances are used, so no back-transformation machinery reappears) and it changes nothing
    about what the space can express.

    Args:
        x: features to rescale.
        target_row_norm: the reference mean row norm (the pooler train split's).
        reference: rows used to measure the current scale — pass the **train** split so the
            statistic never sees validation or test data.
    """
    current = mean_row_norm(reference if reference is not None else x)
    if current <= 0:
        raise ValueError("cannot rescale a feature space with zero mean row norm")
    return np.asarray(x, dtype=np.float64) * (target_row_norm / current)


def per_row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row cosine between two aligned matrices (for the derived-vs-cached cross-check)."""
    a64, b64 = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    num = (a64 * b64).sum(1)
    den = np.linalg.norm(a64, axis=1) * np.linalg.norm(b64, axis=1)
    return num / np.clip(den, 1e-12, None)
