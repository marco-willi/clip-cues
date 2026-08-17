"""Cue-association profiles — the E12 estimators, as the single shared implementation.

A scorer's *profile* is its correlation with every named cue: ``r_j = corr(z, c_j)``. Two detectors
with the same profile describe the same thing in cue terms, whatever their weights look like.

These functions were lifted verbatim (behaviour-preserving) out of
``scripts/interpret/run_score_alignment.py`` so E12 and F1-F7 share one implementation; that script now
imports them, and a regression test pins its published ``summary.json`` numbers.

Estimands, per E12: **within-class** ``r`` is the identifying one — pooled ``r`` is inflated by
class separation, so any label-predictive cue looks "aligned". Uncertainty is a cluster bootstrap by
source photo (``image_id``), because SynthCLIC's real image and all its synthetic variants share an
id and are not independent.
"""

from __future__ import annotations

import numpy as np

RIDGE = 1e-3  # on the correlation-matrix diagonal, for the partial-correlation inverse
N_BOOT = 1000


def col_corr(z: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Pearson r between z and every column of C."""
    zc = z - z.mean()
    Cc = C - C.mean(0)
    den = np.linalg.norm(zc) * np.linalg.norm(Cc, axis=0)
    return (zc @ Cc) / np.clip(den, 1e-12, None)


def partial_corr(z: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Partial r between z and each cue, controlling for all other cues (precision-matrix form)."""
    M = np.column_stack([z, C])
    R = np.corrcoef(M, rowvar=False)
    R[np.diag_indices_from(R)] += RIDGE
    Th = np.linalg.pinv(R)
    d = np.sqrt(np.clip(np.diag(Th), 1e-12, None))
    return -Th[0, 1:] / (d[0] * d[1:])


def boot_ci(z: np.ndarray, C: np.ndarray, ids: np.ndarray, n: int = N_BOOT, seed: int = 0):
    """Cluster-bootstrap by source photo: ``(lo, hi, two-sided p)`` per cue."""
    rng = np.random.default_rng(seed)
    uids = np.unique(ids)
    idx_of = {u: np.where(ids == u)[0] for u in uids}
    draws = np.empty((n, C.shape[1]))
    for i in range(n):
        rows = np.concatenate([idx_of[u] for u in rng.choice(uids, len(uids), replace=True)])
        draws[i] = col_corr(z[rows], C[rows])
    lo, hi = np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)
    frac_neg = (draws <= 0).mean(0)
    p = 2 * np.minimum(frac_neg, 1 - frac_neg)
    return lo, hi, np.clip(p, 1.0 / n, 1.0)


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    q[order] = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def within_class_corr(z: np.ndarray, C: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Per-cue correlation computed inside each class, plus the macro average.

    The identifying estimand: it asks whether the scorer tracks a cue *among images of the same
    class*, rather than merely tracking the class boundary.
    """
    out = {}
    for name, mask in (("real", y == 0), ("synthetic", y == 1)):
        out[name] = col_corr(z[mask], C[mask]) if mask.sum() > 2 else np.full(C.shape[1], np.nan)
    out["macro"] = np.nanmean([out["real"], out["synthetic"]], axis=0)
    return out


def cue_profile(
    z: np.ndarray,
    C: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray | None = None,
    *,
    with_ci: bool = False,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Full per-cue profile for one scorer.

    Args:
        z: scorer logits, ``(n,)``.
        C: cue scores, ``(n, k)``.
        y: 0/1 labels, ``(n,)``.
        ids: cluster ids (source photo) — required when ``with_ci``.
        with_ci: also run the cluster bootstrap and BH-FDR (the expensive part).

    Returns:
        Arrays keyed ``pooled``, ``within_real``, ``within_synthetic``, ``within_macro``,
        ``partial`` and, when requested, ``ci_lo``, ``ci_hi``, ``p``, ``q``.
    """
    wc = within_class_corr(z, C, y)
    out = {
        "pooled": col_corr(z, C),
        "within_real": wc["real"],
        "within_synthetic": wc["synthetic"],
        "within_macro": wc["macro"],
        "partial": partial_corr(z, C),
    }
    if with_ci:
        if ids is None:
            raise ValueError("cluster ids are required for bootstrap CIs")
        lo, hi, p = boot_ci(z, C, ids, n=n_boot, seed=seed)
        out |= {"ci_lo": lo, "ci_hi": hi, "p": p, "q": bh_fdr(p)}
    return out


def random_direction_floor(x: np.ndarray, C: np.ndarray, n_dirs: int = 128, seed: int = 0) -> float:
    """95th percentile of |pooled r| for random unit directions — the "is this cue real?" floor.

    E12 found only 25/128 optimized cues clear this floor, so the floor is what keeps a diffuse
    profile from being over-read.
    """
    rng = np.random.default_rng(seed)
    x64 = np.asarray(x, dtype=np.float64)
    vals = []
    for _ in range(n_dirs):
        w = rng.normal(size=x64.shape[1])
        vals.append(np.abs(col_corr(x64 @ (w / np.linalg.norm(w)), C)))
    return float(np.percentile(np.concatenate(vals), 95))
