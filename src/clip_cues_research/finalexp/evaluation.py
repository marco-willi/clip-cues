"""Shared detection evaluation for F1-F7 — one metric convention across every experiment.

Convention A (per-generator mean AP) everywhere, with the dataset-appropriate real-pairing rule,
plus pooled AUROC alongside for transparency. Sources are attached **by position** because
``image_id`` is not unique in SynthCLIC/SynthBuster: a merge on it would explode the frame and
silently collapse per-generator mAP to pooled AP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from clip_cues_research.analysis.metrics import detection_metrics, pairing_for_dataset


def predictions_frame(z: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """Scored predictions aligned positionally with a split's metadata frame."""
    if len(z) != len(df):
        raise AssertionError(f"score/metadata length mismatch: {len(z)} vs {len(df)}")
    return pd.DataFrame(
        {
            "image_id": df["image_id"].astype(str).values,
            "label": df["label"].to_numpy().astype(int),
            "score": 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64))),
            "source": df["source"].values,
        }
    )


def score_metrics(z: np.ndarray, df: pd.DataFrame, dataset: str) -> dict:
    """Convention-A mAP + pooled AP + pooled AUROC for one set of scores."""
    pred = predictions_frame(z, df)
    pairing = pairing_for_dataset(dataset)
    bundle = detection_metrics(pred, real_pairing=pairing)
    y = pred["label"].to_numpy()
    auroc = float(roc_auc_score(y, pred["score"])) if 0 < int(y.sum()) < len(y) else float("nan")
    return {
        "mAP": float(bundle["mAP"]),
        "pooled_ap": float(bundle["pooled_ap"]),
        "auroc": auroc,
        "n": int(len(pred)),
        "n_generators": int(bundle["n_generators"]),
        "real_pairing": pairing,
    }


def evaluate_head(head, x: np.ndarray, df: pd.DataFrame, dataset: str) -> tuple[dict, np.ndarray]:
    """``(metrics, logits)`` for a :class:`TrainedHead` on one split."""
    z = head.logits(x)
    return score_metrics(z, df, dataset), z


def cluster_bootstrap_auroc_delta(
    z_a: np.ndarray,
    z_b: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Paired cluster-bootstrap CI for ``AUROC(a) - AUROC(b)`` on the same images.

    Clusters are source photos (``image_id``): SynthCLIC's real image and all of its synthetic
    variants share an id and are not independent, so an image-level bootstrap would understate the
    interval. 2,000 draws, seed 0, percentile CIs — the E9/E12/protocol convention.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    uids = np.unique(clusters)
    idx_of = {u: np.where(clusters == u)[0] for u in uids}

    obs_a, obs_b = roc_auc_score(y, z_a), roc_auc_score(y, z_b)
    draws = np.empty(n_boot)
    kept = 0
    for _ in range(n_boot):
        rows = np.concatenate([idx_of[u] for u in rng.choice(uids, len(uids), replace=True)])
        yy = y[rows]
        if 0 < yy.sum() < len(yy):
            draws[kept] = roc_auc_score(yy, z_a[rows]) - roc_auc_score(yy, z_b[rows])
            kept += 1
    draws = draws[:kept]
    return {
        "auroc_a": round(float(obs_a), 6),
        "auroc_b": round(float(obs_b), 6),
        "delta": round(float(obs_a - obs_b), 6),
        "ci_lo": round(float(np.percentile(draws, 2.5)), 6),
        "ci_hi": round(float(np.percentile(draws, 97.5)), 6),
        "n_boot": int(kept),
        "n_clusters": int(len(uids)),
    }


def excess_auroc_recovery(auroc_restricted: float, auroc_full: float) -> float:
    """``(AUROC_restricted - 0.5) / (AUROC_full - 0.5)`` — the E9/N5c recovery ratio."""
    denom = auroc_full - 0.5
    return float("nan") if abs(denom) < 1e-12 else round((auroc_restricted - 0.5) / denom, 6)
