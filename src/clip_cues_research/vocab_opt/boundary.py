"""E11 boundary-normal analysis: decompose a linear detector's normal onto signed cue axes.

Instead of decomposing image embeddings (E9) or retraining cue probes, this module interprets
the classifier itself: the raw-space effective normal ``w`` of a standardized logistic probe,
optionally distilled from pooler (1024-d) space into the shared 768-d space, is approximated as
a sparse signed combination of canonical text axes ``T`` (rows = unit diff directions
normalize(pos) - normalize(neg)).

Data-weighted objective (the one that matters — fidelity where the images lie):

    min_a  || V w - V T^T a ||^2 + lam ||a||_1

so per-image  z_i ~ sum_j a_j (v_i . t_j) + c  is an additive, faithful contribution per axis.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def unitv(w: np.ndarray) -> np.ndarray:
    return w / np.linalg.norm(w)


def fit_probe(
    Xtr: np.ndarray, ytr: np.ndarray, C: float
) -> tuple[StandardScaler, LogisticRegression]:
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(Xtr), ytr)
    return sc, lr


def raw_normal(sc: StandardScaler, lr: LogisticRegression) -> tuple[np.ndarray, float]:
    """Effective (w, b) in the raw feature space of a standardized logistic probe."""
    w = lr.coef_.ravel() / sc.scale_
    b = float(lr.intercept_[0] - (lr.coef_.ravel() * sc.mean_ / sc.scale_).sum())
    return w, b


def distill_to_shared(
    H: np.ndarray, w_eff: np.ndarray, Wp: np.ndarray, lambdas=(1e-6, 1e-4, 1e-2, 1.0, 100.0)
) -> tuple[np.ndarray, dict]:
    """Data-aware ridge distillation of a pooler-space normal into the shared 768-d space.

    Solves min_u ||H w_eff - (H Wp^T) u||^2 + lam ||u||^2 on rows H; returns the u with the
    best preserved-variance R^2 (centered) and a report over the lambda grid.
    """
    G, t = H @ Wp.T, H @ w_eff
    report = {}
    best = None
    for lam in lambdas:
        r = Ridge(alpha=lam, fit_intercept=True).fit(G, t)
        r2 = float(r.score(G, t))
        report[lam] = r2
        if best is None or r2 > best[1]:
            best = (r.coef_.copy(), r2, lam)
    u, r2, lam = best
    return u, {"r2_by_lambda": report, "best_lambda": lam, "preserved_logit_variance_r2": r2}


def lasso_path_decompose(
    Vtr: np.ndarray,
    Vval: np.ndarray,
    yval: np.ndarray | None,
    w: np.ndarray,
    T: np.ndarray,
    alphas: np.ndarray,
) -> list[dict]:
    """Data-weighted lasso of the target scores V@w onto axis similarities V@T^T over a path.

    Returns one row per alpha with: nnz, vector-space cosine coverage cos(T^T a, w),
    val score-space R^2, val label AUROC of the decomposition scores, residual norm fraction,
    and the coefficient vector.
    """
    w = unitv(w)
    Ztr, ztr = Vtr @ T.T, Vtr @ w
    Zval, zval = Vval @ T.T, Vval @ w
    rows = []
    for a in alphas:
        m = Lasso(alpha=a, fit_intercept=True, max_iter=50000).fit(Ztr, ztr)
        coef = m.coef_
        nnz = int((coef != 0).sum())
        w_hat = T.T @ coef
        pred_val = Zval @ coef + m.intercept_
        ss = ((zval - zval.mean()) ** 2).sum()
        r2 = float(1 - ((pred_val - zval) ** 2).sum() / ss)
        auroc = (
            float(roc_auc_score(yval, pred_val))
            if yval is not None and 0 < yval.sum() < len(yval)
            else np.nan
        )
        rows.append(
            {
                "alpha": float(a),
                "nnz": nnz,
                "cos_coverage": float(w @ unitv(w_hat)) if nnz else 0.0,
                "val_score_r2": r2,
                "val_auroc": auroc,
                "residual_norm_frac": float(np.linalg.norm(w - w_hat) / np.linalg.norm(w)),
                "coef": coef.copy(),
            }
        )
    return rows


def knee_row(rows: list[dict], tol: float = 0.01) -> dict:
    """Smallest-support row whose val score-R^2 is within tol of the path maximum."""
    best = max(r["val_score_r2"] for r in rows)
    ok = [r for r in rows if r["val_score_r2"] >= best - tol and r["nnz"] > 0]
    return min(ok, key=lambda r: r["nnz"])


def support_stability(
    Vtr: np.ndarray,
    ids: np.ndarray,
    w: np.ndarray,
    T: np.ndarray,
    alpha: float,
    n_boot: int = 50,
    seed: int = 0,
) -> np.ndarray:
    """Cluster-bootstrap (by image_id) selection frequency of each axis at a fixed alpha."""
    w = unitv(w)
    rng = np.random.default_rng(seed)
    uids = np.unique(ids)
    idx_of = {u: np.where(ids == u)[0] for u in uids}
    Z, z = Vtr @ T.T, Vtr @ w
    freq = np.zeros(T.shape[0])
    for _ in range(n_boot):
        rows = np.concatenate([idx_of[u] for u in rng.choice(uids, len(uids), replace=True)])
        m = Lasso(alpha=alpha, fit_intercept=True, max_iter=50000).fit(Z[rows], z[rows])
        freq += m.coef_ != 0
    return freq / n_boot
