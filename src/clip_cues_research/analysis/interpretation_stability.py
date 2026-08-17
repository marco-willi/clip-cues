"""E8: stability-of-interpretability metrics for the two interpretable heads.

Reviewer 1 asks for more analysis of *generalization*; our revision so far measures generalization of
detection mAP, not whether the *interpretations themselves* are stable/reproducible. The paper makes two
unquantified stability claims this module is built to test:

  * orthogonal head — the ``k=8`` directions of ``W_L1`` are "roughly orthogonal" and *hypothesized* to
    be driven by identical random init (docs/initial_submission.tex §5.1). If true, the directions are an
    artifact of init, not a property of the data.
  * concept model — "more sensitive than simple linear heads to vocabulary and sparsity hyperparameters"
    (§6 discussion). We currently only quantify the *count* of active concepts vs beta (E2), never the
    *identity/ranking* of the selected concepts.

This module is **pure functions over already-fitted artifacts** — collections of direction matrices
(``W_L1`` per fit), per-direction/per-concept importance vectors, and top-term/top-concept sets — so it
is unit-testable without any training. The harness (``scripts/run/run_interpretation_stability.py``) produces
the artifacts; this file only measures their agreement.

Two identifiability facts shape the metric design:
  * orthogonal-head directions are identifiable only up to **permutation, sign, and rotation** within the
    learned subspace, so direction-level metrics first **Hungarian-match** rows and **sign-align** them,
    and we additionally report a rotation-invariant **subspace** distance (principal angles).
  * concept importance is sign-meaningful (a concept votes real vs synthetic) but the *ranking* is the
    interpretable object, so concept metrics use top-K overlap + rank correlation + sign agreement.

Research-only (E8); never back-ported to ``clip_cues``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.linalg import subspace_angles
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


# ── direction-level (orthogonal head) ────────────────────────────────────────────────────────────
def _as_directions(mat: np.ndarray) -> np.ndarray:
    """Normalize an input ``W_L1`` to a (k, d) array of L2-normalized direction rows.

    ``ActivationOrthogonalityHead.layers[-1].weight`` is ``nn.Linear(in=d, out=k).weight`` of shape
    (k, d) — each row is one learned direction in CLIP's d-dim hidden-state space. We L2-normalize rows
    so cosine comparisons are well defined.
    """
    a = np.asarray(mat, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D (k, d) direction matrix, got shape {a.shape}")
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return a / norms


@dataclass(frozen=True)
class DirectionMatch:
    """Result of Hungarian-matching the rows of two direction matrices by ``|cosine|``.

    Attributes:
        col_for_row: ``col_for_row[i]`` is the index in ``B`` matched to row ``i`` of ``A``.
        abs_cosine: per-matched-pair ``|cosine|`` aligned to ``col_for_row`` (1.0 == identical axis).
        mean_abs_cosine: mean of ``abs_cosine`` — the scalar per-direction stability for this pair.
    """

    col_for_row: np.ndarray
    abs_cosine: np.ndarray
    mean_abs_cosine: float


def match_directions(a: np.ndarray, b: np.ndarray) -> DirectionMatch:
    """Permutation/sign-invariant matching of two sets of directions by ``|cosine|``.

    Directions carry no canonical order or sign, so we find the row permutation maximizing total
    ``|cosine|`` (Hungarian on the cost ``1 - |cos|``) and report the matched absolute cosines. ``a`` and
    ``b`` must have the same number of rows ``k``.
    """
    da, db = _as_directions(a), _as_directions(b)
    if da.shape != db.shape:
        raise ValueError(f"direction matrices must have equal shape, got {da.shape} vs {db.shape}")
    abs_cos = np.abs(da @ db.T)  # (k, k), |cosine| between every pair of rows
    row_ind, col_ind = linear_sum_assignment(1.0 - abs_cos)
    matched = abs_cos[row_ind, col_ind]
    col_for_row = np.empty(da.shape[0], dtype=int)
    col_for_row[row_ind] = col_ind
    aligned = np.empty(da.shape[0], dtype=float)
    aligned[row_ind] = matched
    return DirectionMatch(
        col_for_row=col_for_row, abs_cosine=aligned, mean_abs_cosine=float(matched.mean())
    )


def subspace_chordal_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Rotation-invariant distance between the two row-subspaces, in ``[0, sqrt(k)]`` (0 == identical).

    ``sqrt(sum sin^2(principal_angles))``. Unlike ``match_directions`` this ignores how the subspace is
    factored into axes, so it stays small even if individual directions rotate within a stable subspace —
    the key disambiguation for the paper's init hypothesis (axes can be init-driven while the *subspace*
    is data-driven).
    """
    da, db = _as_directions(a), _as_directions(b)
    angles = subspace_angles(da.T, db.T)  # expects (d, k) column matrices
    return float(np.sqrt(np.sum(np.sin(angles) ** 2)))


def _pairwise(values, fn) -> dict[str, float]:
    """Mean/std/min over ``fn`` applied to every unordered pair in ``values`` (>=2 items)."""
    items = list(values)
    if len(items) < 2:
        raise ValueError("need at least two fitted artifacts to measure stability")
    scores = [fn(x, y) for x, y in itertools.combinations(items, 2)]
    arr = np.asarray(scores, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "n_pairs": len(scores),
    }


def direction_stability(direction_matrices) -> dict[str, dict[str, float]]:
    """Aggregate orthogonal-head direction stability across a collection of fits.

    Args:
        direction_matrices: iterable of ``W_L1`` arrays (each (k, d)), one per fit (seed/backbone/domain).

    Returns:
        ``{"matched_abs_cosine": {...}, "subspace_chordal_distance": {...}}`` where each value is the
        mean/std/min over all fit pairs. High ``matched_abs_cosine`` ⇒ stable individual axes; low
        ``subspace_chordal_distance`` ⇒ stable subspace even if axes rotate.
    """
    mats = list(direction_matrices)
    return {
        "matched_abs_cosine": _pairwise(mats, lambda x, y: match_directions(x, y).mean_abs_cosine),
        "subspace_chordal_distance": _pairwise(mats, subspace_chordal_distance),
    }


# ── importance-vector level (both heads) ─────────────────────────────────────────────────────────
def top_k_indices(importance: np.ndarray, k: int) -> set[int]:
    """Indices of the ``k`` most important entries, ranked by **absolute** value (sign-agnostic)."""
    a = np.abs(np.asarray(importance, dtype=float))
    k = min(k, a.shape[0])
    return set(np.argsort(-a)[:k].tolist())


def jaccard(a: set[int], b: set[int]) -> float:
    """Jaccard overlap of two index sets; ``1.0`` for two empty sets."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def top_k_jaccard(importances, k: int) -> dict[str, float]:
    """Mean/std/min pairwise Jaccard of the top-``k`` (by ``|importance|``) across fits.

    Use for "do the same directions/concepts stay in the top-K" across seeds/beta/backbone/domain.
    """
    sets = [top_k_indices(v, k) for v in importances]
    return _pairwise(sets, jaccard) | {"k": k}


def importance_rank_correlation(importances) -> dict[str, float]:
    """Mean/std/min pairwise Spearman correlation of ``|importance|`` across fits.

    Uses absolute importance so the ranking compares *how strongly* each direction/concept drives the
    decision, independent of which class it votes for.
    """
    vecs = [np.abs(np.asarray(v, dtype=float)) for v in importances]

    def _rho(x, y):
        rho = spearmanr(x, y).statistic
        return 0.0 if np.isnan(rho) else float(rho)

    return _pairwise(vecs, _rho)


def sign_agreement(importances) -> np.ndarray:
    """Per-entry fraction of fits agreeing with the majority sign (length = #directions/#concepts).

    For each concept/direction, the share of fits whose signed importance matches the modal sign across
    fits. ``1.0`` ⇒ the concept always votes the same class; values near ``0.5`` ⇒ its class role flips
    between fits (an instability worth flagging per R1#7). Zeros count toward whichever sign is in the
    majority via ``np.sign`` (0 → 0, treated as its own bucket only if it dominates).
    """
    mat = np.vstack(
        [np.sign(np.asarray(v, dtype=float)) for v in importances]
    )  # (n_fits, n_entries)
    n_fits = mat.shape[0]
    out = np.empty(mat.shape[1], dtype=float)
    for j in range(mat.shape[1]):
        signs, counts = np.unique(mat[:, j], return_counts=True)
        out[j] = counts.max() / n_fits
    return out


# ── top-term sets (orthogonal head vocabulary interpretation) ────────────────────────────────────
def top_term_jaccard(term_sets) -> dict[str, float]:
    """Mean/std/min pairwise Jaccard of top-vocabulary-term sets for one direction across fits.

    Args:
        term_sets: iterable of sets/lists of vocabulary terms (strings) — the top-N antonym/TextSpan
            terms aligned to a given direction, one set per fit.
    """
    sets = [set(s) for s in term_sets]
    return _pairwise(sets, jaccard)


# ── domain-dependent importance (shared by both heads, used for Tier-B transfer) ─────────────────
def class_mean_difference_importance(contributions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Signed per-feature importance = mean contribution(synthetic) - mean(real), on one domain.

    Both heads expose a per-image, per-feature *logit contribution* (orthogonal head:
    ``activation_i * w_logit_i``; concept model: the ``per_concept_logit_contribution`` forward output).
    The class-mean difference is the paper's "difference in logit contribution" diagnostic — how strongly
    (and toward which class) each direction/concept drives the decision **on this domain's images**. It is
    domain-dependent (unlike the static ``W_classifier``), which is what makes it the right object for the
    SynthCLIC-vs-CF transfer comparison.

    Args:
        contributions: (n_images, n_features) per-image logit contributions on the domain.
        labels: (n_images,) binary labels (1 = synthetic, 0 = real).

    Returns:
        (n_features,) signed importance; ``|value|`` is the discriminative strength on this domain.
    """
    c = np.asarray(contributions, dtype=float)
    y = np.asarray(labels).reshape(-1)
    if c.ndim != 2 or c.shape[0] != y.shape[0]:
        raise ValueError(f"contributions {c.shape} incompatible with labels {y.shape}")
    syn, real = c[y == 1], c[y == 0]
    mean_syn = syn.mean(axis=0) if len(syn) else np.zeros(c.shape[1])
    mean_real = real.mean(axis=0) if len(real) else np.zeros(c.shape[1])
    return mean_syn - mean_real


# ── transfer (Tier B: SynthCLIC vs CommunityForensics) ───────────────────────────────────────────
def diagnostic_agreement(importance_a: np.ndarray, importance_b: np.ndarray) -> float:
    """Spearman of ``|importance|`` between two evaluation domains (e.g. SynthCLIC vs CF).

    Single scalar for the Tier-B transfer table: does the *same* direction/concept stay the most
    discriminative when the SAME fitted head is diagnosed on a different benchmark? Low ⇒ the
    interpretation is dataset-specific.
    """
    x, y = (
        np.abs(np.asarray(importance_a, dtype=float)),
        np.abs(np.asarray(importance_b, dtype=float)),
    )
    rho = spearmanr(x, y).statistic
    return 0.0 if np.isnan(rho) else float(rho)


def selection_survival(importance_a: np.ndarray, importance_b: np.ndarray, k: int) -> float:
    """Jaccard of the top-``k`` (by ``|importance|``) between two domains — the Tier-B survival metric."""
    return jaccard(top_k_indices(importance_a, k), top_k_indices(importance_b, k))


def transfer_table(source_importance: np.ndarray, target_importance_by_group, k: int):
    """Per-group Tier-B transfer of a single fitted head's interpretation, source domain vs target.

    For each group (e.g. CommunityForensics architecture: GAN / LatDiff / PixDiff / Commercial), compares
    the importance the *same* head assigns when diagnosed on the source domain (SynthCLIC) vs on that
    group's images, via ``diagnostic_agreement`` (rank) and ``selection_survival`` (top-K overlap). This
    is the artifact that links per-architecture interpretation survival to the E7 per-architecture AP —
    do the discriminative directions/concepts go silent exactly where detection collapses?

    Args:
        source_importance: importance vector on the source domain (length = #directions/#concepts).
        target_importance_by_group: mapping ``group -> importance vector on that group's target images``.
        k: top-K for the survival Jaccard.

    Returns:
        ``pandas.DataFrame`` [group, diagnostic_agreement, selection_survival] sorted by group.
    """
    import pandas as pd

    rows = []
    for group, imp in target_importance_by_group.items():
        rows.append(
            {
                "group": group,
                "diagnostic_agreement": diagnostic_agreement(source_importance, imp),
                "selection_survival": selection_survival(source_importance, imp, k),
            }
        )
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
