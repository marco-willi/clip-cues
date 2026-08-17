"""Unit tests for the E11 boundary module (toy-vector checks for normals, distillation, lasso)."""

import numpy as np

from clip_cues_research.vocab_opt.boundary import (
    distill_to_shared,
    fit_probe,
    knee_row,
    lasso_path_decompose,
    raw_normal,
    unitv,
)


def test_raw_normal_reproduces_probe_scores():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6)) * np.array([1.0, 5.0, 0.2, 1.0, 3.0, 1.0])
    y = (X[:, 1] - X[:, 4] > 0).astype(int)
    sc, lr = fit_probe(X, y, C=1.0)
    w, b = raw_normal(sc, lr)
    assert np.allclose(X @ w + b, lr.decision_function(sc.transform(X)), atol=1e-8)


def test_distill_recovers_row_space_direction():
    rng = np.random.default_rng(1)
    Wp = np.linalg.qr(rng.normal(size=(10, 4)))[0].T  # (4, 10) projection
    u_true = rng.normal(size=4)
    w = Wp.T @ u_true  # normal entirely in the row space
    H = rng.normal(size=(300, 10))
    u, rep = distill_to_shared(H, w, Wp, lambdas=(1e-8,))
    assert rep["preserved_logit_variance_r2"] > 0.999
    assert abs(unitv(u) @ unitv(u_true)) > 0.999


def test_lasso_recovers_sparse_combination():
    rng = np.random.default_rng(2)
    T = np.linalg.qr(rng.normal(size=(8, 8)))[0]  # orthonormal axes
    w = 2.0 * T[0] - 1.0 * T[3]  # true sparse signed combination
    V = rng.normal(size=(500, 8))
    yval = (V @ unitv(w) > 0).astype(int)
    rows = lasso_path_decompose(V, V, yval, w, T, alphas=np.logspace(-1, -4, 10))
    knee = knee_row(rows, tol=0.01)
    sup = set(np.flatnonzero(knee["coef"]))
    assert {0, 3} <= sup and knee["val_score_r2"] > 0.98
    assert knee["coef"][0] > 0 > knee["coef"][3]
    assert knee["cos_coverage"] > 0.98 and knee["val_auroc"] > 0.99
