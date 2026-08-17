"""Trainer, stability metrics and profile estimators for the F1-F7 consolidation (Step 3)."""

from __future__ import annotations

import numpy as np
import pytest

from clip_cues_research.finalexp import profiles, stability
from clip_cues_research.finalexp.trainer import RECIPE, make_ortho_head, train_head


@pytest.fixture(scope="module")
def toy():
    """A small linearly-separable problem, deterministic."""
    rng = np.random.default_rng(0)
    n, d = 400, 24
    w = rng.normal(size=d)
    x = rng.normal(size=(n, d))
    y = (x @ w + 0.3 * rng.normal(size=n) > 0).astype(int)
    return x[:300], y[:300], x[300:], y[300:]


# ── trainer ──────────────────────────────────────────────────────────────────────────────────
def test_training_is_reproducible_for_a_fixed_seed(toy):
    xtr, ytr, xva, yva = toy
    a = train_head(xtr, ytr, xva, yva, seed=123)
    b = train_head(xtr, ytr, xva, yva, seed=123)
    np.testing.assert_allclose(a.weight, b.weight, rtol=0, atol=0)
    assert a.best_val_ce == b.best_val_ce


def test_different_seeds_give_different_weights(toy):
    xtr, ytr, xva, yva = toy
    a = train_head(xtr, ytr, xva, yva, seed=123)
    b = train_head(xtr, ytr, xva, yva, seed=124)
    assert not np.allclose(a.weight, b.weight)


def test_early_stopping_respects_the_recipe(toy):
    xtr, ytr, xva, yva = toy
    h = train_head(xtr, ytr, xva, yva, seed=123)
    assert h.epochs_run <= RECIPE.epochs
    assert h.best_epoch >= 0
    # the restored checkpoint is the best one seen, not the last
    assert h.best_val_ce == min(e["val_ce"] for e in h.history)


def test_k8_effective_direction_reproduces_the_head_logits(toy):
    """w_eff = w2 @ W1 must reproduce the k=8 head exactly (it is linear when non_linear=False).

    This is what licenses F2's distinction between the stable *effective direction* and the
    unstable *individual axes*.
    """
    import torch

    xtr, ytr, xva, yva = toy
    h = train_head(
        xtr,
        ytr,
        xva,
        yva,
        seed=123,
        head_factory=lambda d: make_ortho_head(d, k=8),
        head_type="ortho_k8",
    )
    head = make_ortho_head(xtr.shape[1], k=8)
    head.load_state_dict({k: torch.as_tensor(v) for k, v in h.state_dict.items()})
    head.eval()
    probe = np.random.default_rng(1).normal(size=(32, xtr.shape[1]))
    with torch.no_grad():
        ref = head(torch.as_tensor(probe, dtype=torch.float32))["logits"].view(-1).numpy()
    np.testing.assert_allclose(h.logits(probe), ref, rtol=0, atol=1e-4)


def test_k8_exposes_individual_axes(toy):
    xtr, ytr, xva, yva = toy
    h = train_head(
        xtr,
        ytr,
        xva,
        yva,
        seed=123,
        head_factory=lambda d: make_ortho_head(d, k=8),
        head_type="ortho_k8",
    )
    assert h.axes is not None and h.axes.shape == (8, xtr.shape[1])


# ── stability metrics ────────────────────────────────────────────────────────────────────────
def test_cosines_of_a_vector_with_itself_are_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 10))
    w = rng.normal(size=10)
    assert stability.raw_cosine(w, w) == pytest.approx(1.0)
    assert stability.sigma_cosine(w, w, x) == pytest.approx(1.0)
    assert stability.whitened_cosine(w, w, x) == pytest.approx(1.0, abs=1e-6)


def test_sigma_cosine_equals_score_correlation():
    """The Sigma-metric cosine *is* the correlation of the two scorers' centered scores (N24)."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(500, 12)) @ rng.normal(size=(12, 12))  # correlated features
    a, b = rng.normal(size=12), rng.normal(size=12)
    za, zb = x @ a, x @ b
    assert stability.sigma_cosine(a, b, x) == pytest.approx(np.corrcoef(za, zb)[0, 1], abs=1e-10)


def test_raw_and_sigma_cosine_can_disagree_sharply():
    """Guards the plan's headline caution: raw cosine can call identical scorers 'unstable'.

    Reproduces the N21 pattern (raw cos 0.07, score correlation 0.94) on synthetic data, so the
    choice of primary metric is pinned by a test rather than by a comment.
    """
    rng = np.random.default_rng(3)
    d = 40
    scale = np.concatenate([np.full(4, 12.0), np.full(d - 4, 0.05)])  # anisotropic data
    x = rng.normal(size=(800, d)) * scale
    a = np.concatenate([[1.0, 0, 0, 0], rng.normal(size=d - 4) * 3])
    b = np.concatenate([[1.0, 0, 0, 0], rng.normal(size=d - 4) * 3])
    assert stability.raw_cosine(a, b) < 0.5
    assert stability.sigma_cosine(a, b, x) > 0.9


def test_matched_axis_cosines_are_permutation_invariant():
    """Hungarian matching must not report instability caused purely by axis ordering."""
    rng = np.random.default_rng(4)
    axes = rng.normal(size=(8, 30))
    shuffled = axes[rng.permutation(8)] * rng.choice([-1.0, 1.0], size=(8, 1))  # reorder + flip
    out = stability.matched_axis_cosines(axes, shuffled)
    assert out["min"] == pytest.approx(1.0, abs=1e-8)


def test_extreme_overlap_is_one_for_identical_rankings():
    z = np.random.default_rng(5).normal(size=300)
    out = stability.extreme_overlap(z, z.copy(), n=50)
    assert out["top50_jaccard"] == 1.0 and out["bottom50_jaccard"] == 1.0


# ── profile estimators ───────────────────────────────────────────────────────────────────────
def test_col_corr_matches_numpy_corrcoef():
    rng = np.random.default_rng(6)
    z, C = rng.normal(size=200), rng.normal(size=(200, 7))
    expected = [np.corrcoef(z, C[:, j])[0, 1] for j in range(7)]
    np.testing.assert_allclose(profiles.col_corr(z, C), expected, atol=1e-12)


def test_bh_fdr_is_monotone_and_bounded():
    p = np.array([0.001, 0.01, 0.02, 0.2, 0.9])
    q = profiles.bh_fdr(p)
    assert (q >= p - 1e-12).all() and (q <= 1).all()
    assert (np.diff(q) >= -1e-12).all()


def test_within_class_corr_is_not_the_pooled_corr():
    """Pooled r is inflated by class separation — the reason E12 identifies on within-class r."""
    rng = np.random.default_rng(7)
    y = np.repeat([0, 1], 200)
    cue = rng.normal(size=400) + 3.0 * y  # cue tracks the class, not the score within a class
    z = 3.0 * y + rng.normal(size=400) * 0.1
    C = cue[:, None]
    pooled = profiles.col_corr(z, C)[0]
    within = profiles.within_class_corr(z, C, y)["macro"][0]
    assert pooled > 0.85 and abs(within) < 0.3


def test_cue_profile_returns_all_fields_with_ci():
    rng = np.random.default_rng(8)
    n = 120
    z, C, y = rng.normal(size=n), rng.normal(size=(n, 5)), rng.integers(0, 2, n)
    ids = np.repeat(np.arange(n // 4), 4)  # clustered by "source photo"
    out = profiles.cue_profile(z, C, y, ids, with_ci=True, n_boot=50)
    for key in ("pooled", "within_macro", "partial", "ci_lo", "ci_hi", "p", "q"):
        assert out[key].shape == (5,)


def test_whitened_cosine_is_sigma_cosine():
    """N19's "whitened cosine" is not an independent check — it IS the data-metric cosine.

    Pinned so the two are never reported as separate columns of evidence.
    """
    rng = np.random.default_rng(11)
    x = rng.normal(size=(300, 15)) @ rng.normal(size=(15, 15))
    a, b = rng.normal(size=15), rng.normal(size=15)
    assert stability.whitened_cosine(a, b, x) == pytest.approx(stability.sigma_cosine(a, b, x))
    assert set(stability.direction_agreement(a, b, x)) == {"sigma_cosine", "raw_cosine"}
