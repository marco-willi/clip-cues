"""E7 CommunityForensics evaluation core (revision-only).

Score a CommunityForensics split with any detector exposing ``list[PIL] -> probs`` and persist
**full-metadata** predictions (parquet) + overall metrics, so the per-architecture / per-generator /
per-real-source breakdowns (export step) need no re-inference. Detectors plug in via a
``score_batch`` callable, so CLIP heads (``model.predict_batch``) and forensic CNNs
(``predict_probs`` + ``eval_transform``) share one code path with guaranteed metadata alignment
(metadata is read from the same row indices that produced each score).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from pyarrow import Table
from pyarrow.parquet import ParquetWriter
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from clip_cues_research.results import save_run_results

EXPERIMENT = "e7_community_eval"
_META = ["image_id", "label", "source", "architecture", "real_source", "subset"]


def _metrics_from_arrays(y: np.ndarray, s: np.ndarray) -> dict:
    pred = (s >= 0.5).astype(int)
    out = {"n": int(len(y)), "n_fake": int(y.sum()), "n_real": int((y == 0).sum())}
    if 0 < y.sum() < len(y):
        out["overall_ap"] = float(average_precision_score(y, s))
        out["auroc"] = float(roc_auc_score(y, s))
        out["mAP"] = out["overall_ap"]
    out["accuracy"] = float(accuracy_score(y, pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
    return out


def _batch_to_frame(batch: dict, probs: np.ndarray, detector: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": [str(x) for x in batch["image_id"]],
            "label": [int(x) for x in batch["label"]],
            "score": [float(x) for x in probs],
            "source": [str(x) for x in batch["source"]],
            "architecture": [str(x) for x in batch["architecture"]],
            "real_source": [str(x) for x in batch["real_source"]],
            "subset": [str(x) for x in batch["subset"]],
            "detector": detector,
        }
    )


def score_cf_split(
    split,
    score_batch: Callable[[list[Image.Image]], Sequence[float]],
    *,
    detector: str,
    batch_size: int = 32,
    max_samples: int | None = None,
) -> pd.DataFrame:
    """Score a (map-style) CommunityForensics split and return a per-image DataFrame.

    Columns: image_id, label, score (P(fake)), source, architecture, real_source, subset, detector.
    Rows are produced in split order; metadata is sliced from the same indices as each scored batch.
    """
    n = len(split) if max_samples is None else min(max_samples, len(split))
    cols: dict[str, list] = {k: [] for k in (_META + ["score"])}
    for i in range(0, n, batch_size):
        batch = split[i : min(i + batch_size, n)]
        probs = np.asarray(score_batch([img.convert("RGB") for img in batch["image"]])).reshape(-1)
        cols["score"].extend(float(p) for p in probs)
        cols["image_id"].extend(str(x) for x in batch["image_id"])
        cols["label"].extend(int(x) for x in batch["label"])
        cols["source"].extend(str(x) for x in batch["source"])
        cols["architecture"].extend(str(x) for x in batch["architecture"])
        cols["real_source"].extend(str(x) for x in batch["real_source"])
        cols["subset"].extend(str(x) for x in batch["subset"])
    df = pd.DataFrame(cols)
    df["detector"] = detector
    return df


def cf_metrics(df: pd.DataFrame) -> dict:
    """Overall metrics for one detector (per-generator/architecture breakdowns happen at export)."""
    return _metrics_from_arrays(df["label"].to_numpy(), df["score"].to_numpy())


def score_cf_split_to_parquet(
    split,
    score_batch: Callable[[list[Image.Image]], Sequence[float]],
    *,
    detector: str,
    out_path: str | Path,
    batch_size: int = 32,
    max_samples: int | None = None,
    progress_every: int = 50,
) -> dict:
    """Stream CF predictions to parquet while scoring, then return overall metrics.

    This avoids holding the full metadata table in memory and emits periodic progress lines so
    long-running CF evals don't look stalled in detached logs.
    """
    n = len(split) if max_samples is None else min(max_samples, len(split))
    total_batches = max(1, math.ceil(n / batch_size))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_all: list[int] = []
    s_all: list[float] = []
    writer: ParquetWriter | None = None
    t0 = time.monotonic()

    try:
        for batch_idx, i in enumerate(range(0, n, batch_size), start=1):
            end = min(i + batch_size, n)
            batch = split[i:end]
            probs = np.asarray(score_batch([img.convert("RGB") for img in batch["image"]])).reshape(
                -1
            )
            batch_df = _batch_to_frame(batch, probs, detector)
            table = Table.from_pandas(batch_df, preserve_index=False)
            if writer is None:
                writer = ParquetWriter(str(out_path), table.schema, compression="snappy")
            writer.write_table(table)
            y_all.extend(batch_df["label"].tolist())
            s_all.extend(batch_df["score"].tolist())

            if progress_every > 0 and (
                batch_idx % progress_every == 0 or batch_idx == total_batches
            ):
                elapsed = time.monotonic() - t0
                done = end
                print(
                    f"  {detector}: batch {batch_idx}/{total_batches} ({done}/{n}) "
                    f"in {elapsed:.1f}s",
                    flush=True,
                )
    finally:
        if writer is not None:
            writer.close()

    return _metrics_from_arrays(np.asarray(y_all), np.asarray(s_all))


def save_cf_eval(
    detector: str, df: pd.DataFrame, metrics: dict, run_id: str, *, base: str = "results"
) -> Path:
    """Persist metrics (`results/e7_community_eval/<detector>/<run_id>/metrics.json`) and the
    full-metadata predictions (`results/e7_community_eval/predictions/<detector>__<run_id>.parquet`)."""
    run_dir = save_run_results(EXPERIMENT, detector, metrics, run_id=run_id, base=base)
    pred_dir = Path(base) / EXPERIMENT / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(pred_dir / f"{detector}__{run_id}.parquet", index=False)
    return run_dir
