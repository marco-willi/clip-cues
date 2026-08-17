"""E8 Step-1 acceptance tests: the stability metrics behave on controlled synthetic inputs.

The invariants below are the contract the harness relies on:
  * identical artifacts ⇒ maximal stability (matched |cos| = 1, chordal distance = 0, Jaccard = 1,
    Spearman = 1, sign agreement = 1);
  * permutation + sign flips of the SAME directions ⇒ still maximal (the identifiability we designed for);
  * unrelated/random artifacts ⇒ clearly lower;
  * a stable *subspace* with rotated *axes* ⇒ small chordal distance but reduced matched |cos|
    (the disambiguation that tests the paper's init hypothesis).
"""

from __future__ import annotations

import numpy as np
import pytest

from clip_cues_research.analysis.interpretation_stability import (
    class_mean_difference_importance,
    diagnostic_agreement,
    direction_stability,
    importance_rank_correlation,
    jaccard,
    match_directions,
    selection_survival,
    sign_agreement,
    subspace_chordal_distance,
    top_k_indices,
    top_k_jaccard,
    top_term_jaccard,
    transfer_table,
)


def _orthonormal(k: int, d: int, seed: int) -> np.ndarray:
    """k orthonormal direction rows in R^d (like an orthogonally-init W_L1)."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d, k)))
    return q[:, :k].T  # (k, d)


# ── direction matching / subspace ────────────────────────────────────────────────────────────────
def test_match_identical_directions_is_one():
    w = _orthonormal(8, 64, seed=0)
    assert match_directions(w, w).mean_abs_cosine == pytest.approx(1.0, abs=1e-9)
    assert subspace_chordal_distance(w, w) == pytest.approx(0.0, abs=1e-7)


def test_match_is_permutation_and_sign_invariant():
    w = _orthonormal(8, 64, seed=1)
    perm = np.random.default_rng(2).permutation(8)
    signs = np.array([1, -1, 1, 1, -1, -1, 1, -1])[:, None]
    w_shuffled = w[perm] * signs
    m = match_directions(w, w_shuffled)
    assert m.mean_abs_cosine == pytest.approx(1.0, abs=1e-9)
    # same subspace regardless of axis permutation/sign
    assert subspace_chordal_distance(w, w_shuffled) == pytest.approx(0.0, abs=1e-7)


def test_random_directions_are_less_stable_than_identical():
    a = _orthonormal(8, 64, seed=3)
    b = _orthonormal(8, 64, seed=4)
    assert match_directions(a, b).mean_abs_cosine < 0.9
    assert subspace_chordal_distance(a, b) > 0.1


def test_rotated_axes_keep_subspace_but_lower_matched_cosine():
    """A within-subspace rotation: chordal distance stays ~0 but individual axes no longer align."""
    base = _orthonormal(8, 64, seed=5)
    # rotate the 8 axes among themselves (stays in the same row-subspace)
    rot, _ = np.linalg.qr(np.random.default_rng(6).standard_normal((8, 8)))
    rotated = rot @ base
    assert subspace_chordal_distance(base, rotated) == pytest.approx(0.0, abs=1e-7)
    assert match_directions(base, rotated).mean_abs_cosine < 0.99


def test_direction_stability_aggregates_pairs():
    mats = [_orthonormal(8, 64, seed=s) for s in range(4)]
    out = direction_stability(mats)
    assert out["matched_abs_cosine"]["n_pairs"] == 6  # C(4,2)
    assert 0.0 <= out["matched_abs_cosine"]["mean"] <= 1.0
    assert out["subspace_chordal_distance"]["mean"] >= 0.0


# ── importance vectors ─────────────────────────────────────────────────────────────────────────
def test_top_k_indices_uses_absolute_value():
    imp = np.array([0.1, -5.0, 0.2, 3.0])
    assert top_k_indices(imp, 2) == {1, 3}


def test_jaccard_basic():
    assert jaccard({1, 2, 3}, {1, 2, 3}) == 1.0
    assert jaccard({1, 2}, {3, 4}) == 0.0
    assert jaccard(set(), set()) == 1.0
    assert jaccard({1, 2, 3, 4}, {3, 4}) == pytest.approx(0.5)


def test_top_k_jaccard_identical_is_one():
    imp = np.array([5.0, 1.0, 4.0, 0.1, 3.0])
    out = top_k_jaccard([imp, imp.copy(), imp.copy()], k=3)
    assert out["mean"] == pytest.approx(1.0)
    assert out["k"] == 3


def test_rank_correlation_identical_and_reversed():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert importance_rank_correlation([a, a.copy()])["mean"] == pytest.approx(1.0)
    # |importance| of a and of -reversed differ in ranking ⇒ strong negative correlation
    rev = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert importance_rank_correlation([a, rev])["mean"] == pytest.approx(-1.0)


def test_sign_agreement():
    # concept 0 always +, concept 1 flips 2:1, concept 2 always -
    fits = [np.array([1.0, 1.0, -1.0]), np.array([2.0, -1.0, -3.0]), np.array([0.5, 1.0, -0.2])]
    agree = sign_agreement(fits)
    assert agree[0] == pytest.approx(1.0)
    assert agree[1] == pytest.approx(2 / 3)
    assert agree[2] == pytest.approx(1.0)


# ── transfer (Tier B) ────────────────────────────────────────────────────────────────────────────
def test_diagnostic_agreement_and_survival():
    a = np.array([0.1, 0.9, 0.2, 0.8, 0.05])
    assert diagnostic_agreement(a, a.copy()) == pytest.approx(1.0)
    assert selection_survival(a, a.copy(), k=2) == 1.0
    b = np.array([0.9, 0.1, 0.8, 0.2, 0.5])  # different ranking
    assert selection_survival(a, b, k=2) < 1.0


def test_top_term_jaccard():
    s1 = {"sharp detail", "minimalist", "high saturation"}
    s2 = {"sharp detail", "minimalist", "low contrast"}
    out = top_term_jaccard([s1, s2])
    assert out["mean"] == pytest.approx(2 / 4)


def test_class_mean_difference_importance():
    # feature 0 separates classes (+2 for synthetic), feature 1 is noise-ish, feature 2 favors real
    contribs = np.array(
        [
            [2.0, 0.1, -1.0],  # synthetic
            [2.0, -0.1, -1.0],  # synthetic
            [0.0, 0.0, 1.0],  # real
            [0.0, 0.2, 1.0],  # real
        ]
    )
    labels = np.array([1, 1, 0, 0])
    imp = class_mean_difference_importance(contribs, labels)
    assert imp[0] == pytest.approx(2.0)  # strong synthetic driver
    assert imp[2] == pytest.approx(-2.0)  # strong real driver
    assert abs(imp[1]) < 0.2  # weak


def test_transfer_table_per_group():
    src = np.array([0.1, 0.9, 0.2, 0.8, 0.05])
    targets = {
        "LatDiff": src.copy(),  # interpretation fully survives
        "GAN": np.array([0.9, 0.1, 0.8, 0.2, 0.5]),  # reshuffled ⇒ low survival
    }
    df = transfer_table(src, targets, k=2)
    assert list(df["group"]) == ["GAN", "LatDiff"]  # sorted
    lat = df[df["group"] == "LatDiff"].iloc[0]
    gan = df[df["group"] == "GAN"].iloc[0]
    assert lat["diagnostic_agreement"] == pytest.approx(1.0)
    assert lat["selection_survival"] == 1.0
    assert gan["selection_survival"] < lat["selection_survival"]


def test_needs_two_artifacts():
    with pytest.raises(ValueError):
        importance_rank_correlation([np.array([1.0, 2.0])])
