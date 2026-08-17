"""Stability and agreement metrics for F1/F2/F7.

**The primary direction metric is the data-metric (Sigma) cosine, not the raw cosine.** Two results
in the interpretation record show a raw-weight cosine gives the wrong verdict about whether two
detectors compute the same thing:

- N21/E11a: ``cos(P768t, P1024-distilled) = 0.07`` in raw weights, yet their validation scores
  correlate **0.938** and both select the same top axes at bootstrap frequency 1.00. E11a's
  conclusion: *interpret boundaries where the images lie, not in weight space*.
- N24/E12: the raw cosine understates the best cue's alignment ~2.4x (0.189 vs 0.459) and only
  rank-agrees at rho ~ 0.65-0.72.

So a naive raw seed-cosine of ~0.5 in F1 could be read as "the canonical direction is unstable"
when the models are functionally identical. Every function here reports both; callers must state
which one they quote.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


def raw_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Plain cosine between two direction vectors. Secondary metric — never quote alone."""
    a64, b64 = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    den = np.linalg.norm(a64) * np.linalg.norm(b64)
    return float(a64 @ b64 / max(den, 1e-12))


def sigma_cosine(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    """Data-metric cosine: the cosine in the metric induced by the data covariance ``Sigma``.

    ``<a, b>_Sigma = a' Sigma b``, which is exactly the correlation between the two scorers'
    centered scores on ``x``. This is the metric under which "same decision function" means what a
    reader assumes it means.

    Args:
        a, b: direction vectors in the feature space of ``x``.
        x: ``(n, d)`` feature matrix defining the data metric.
    """
    xc = np.asarray(x, dtype=np.float64)
    xc = xc - xc.mean(0)
    za, zb = xc @ np.asarray(a, dtype=np.float64), xc @ np.asarray(b, dtype=np.float64)
    den = np.linalg.norm(za) * np.linalg.norm(zb)
    return float(za @ zb / max(den, 1e-12))


def whitened_cosine(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    """N19's "whitened cosine" — **the same quantity** as :func:`sigma_cosine`.

    Kept as a named alias because the interpretation record (N19, N21) uses this term, but it is
    not an independent check: applying ``Sigma^{1/2}`` to both directions and taking the ordinary
    cosine gives ``a'Sigma b / sqrt(a'Sigma a . b'Sigma b)``, which is exactly the data-metric
    cosine and exactly the correlation of the two scorers' centered scores. Verified numerically
    (``tests/test_finalexp.py::test_whitened_cosine_is_sigma_cosine``); reporting both as separate
    columns would double-count one piece of evidence.
    """
    return sigma_cosine(a, b, x)


def direction_agreement(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> dict[str, float]:
    """The two *distinct* direction metrics for one pair.

    ``sigma_cosine`` (primary, = N19's "whitened cosine") and ``raw_cosine`` (secondary, reported
    for continuity with weight-space statements in the earlier record).
    """
    return {
        "sigma_cosine": round(sigma_cosine(a, b, x), 6),
        "raw_cosine": round(raw_cosine(a, b), 6),
    }


def score_agreement(za: np.ndarray, zb: np.ndarray) -> dict[str, float]:
    """Score-level agreement between two scorers on the same images."""
    za64, zb64 = np.asarray(za, dtype=np.float64), np.asarray(zb, dtype=np.float64)
    pear = float(np.corrcoef(za64, zb64)[0, 1])
    spear = float(spearmanr(za64, zb64).statistic)
    agree = float(((za64 > 0) == (zb64 > 0)).mean())
    return {
        "logit_pearson": round(pear, 6),
        "logit_spearman": round(spear, 6),
        "decision_agreement": round(agree, 6),
    }


def extreme_overlap(za: np.ndarray, zb: np.ndarray, n: int = 50) -> dict[str, float]:
    """Jaccard overlap of the top-n and bottom-n ranked images — the montage's stability.

    F5 ranks images by a detector's logit, so this answers "would the figure look the same?".
    """
    ra, rb = np.argsort(za), np.argsort(zb)
    out = {}
    for label, sa, sb in (
        ("top", set(ra[-n:]), set(rb[-n:])),
        ("bottom", set(ra[:n]), set(rb[:n])),
    ):
        out[f"{label}{n}_jaccard"] = round(len(sa & sb) / max(len(sa | sb), 1), 6)
    out["n"] = n
    return out


def profile_agreement(pa: np.ndarray, pb: np.ndarray) -> float:
    """Spearman between two per-cue association profiles (the E12/S2 quantity)."""
    return round(float(spearmanr(np.asarray(pa), np.asarray(pb)).statistic), 6)


def matched_axis_cosines(axes_a: np.ndarray, axes_b: np.ndarray) -> dict:
    """Compare two sets of factorized axes under optimal (Hungarian) matching on |cosine|.

    Without matching, axis permutation between seeds trivially manufactures "instability" — the
    headline claim of F2 would then be an artifact of arbitrary ordering rather than a property of
    the factorization. Sign is ignored (an axis and its negation are the same direction).
    """
    a = np.asarray(axes_a, dtype=np.float64)
    b = np.asarray(axes_b, dtype=np.float64)
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    sim = np.abs(a @ b.T)
    rows, cols = linear_sum_assignment(-sim)
    matched = sim[rows, cols]
    return {
        "matched_abs_cosines": [round(float(v), 6) for v in matched],
        "mean": round(float(matched.mean()), 6),
        "min": round(float(matched.min()), 6),
        "max": round(float(matched.max()), 6),
    }


def summarize_pairs(values: list[float]) -> dict[str, float]:
    """Mean / min / max over seed pairs. Ranges matter more than means with only 5 seeds."""
    v = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(v.mean()), 6),
        "min": round(float(v.min()), 6),
        "max": round(float(v.max()), 6),
        "n_pairs": int(len(v)),
    }
