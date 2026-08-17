"""Step 1 acceptance test: the canonical per-generator mAP reproduces the paper's metric.

The oracle below is a verbatim port of the original
``archive/detection_via_clip/analyse.py::calculate_metrics`` (lines 392-428): for each synthetic
source, AP over {that source's fakes ∪ ALL reals}. We assert the new
``clip_cues_research.analysis.metrics`` helpers match it to ≤1e-9, and that they differ from the
pooled-AP convention (so a regression back to pooled AP would fail loudly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from clip_cues_research.analysis.metrics import (
    detection_metrics,
    pairing_for_dataset,
    per_generator_ap,
    per_generator_map,
)


def reference_calculate_metrics(df: pd.DataFrame) -> list[dict]:
    """Verbatim logic from archive/detection_via_clip/analyse.py::calculate_metrics (shared reals)."""
    results = []
    synthetic_sources = df.query("label == 1").source.unique()
    for synthetic_source in synthetic_sources:
        df_synthetic = df.query(f"source == '{synthetic_source}'")
        df_real = df.query("label == 0")
        df_eval = pd.concat([df_real, df_synthetic])
        ap = average_precision_score(y_true=df_eval.label, y_score=df_eval.label_prob)
        results.append({"source": synthetic_source, "average_precision": float(ap)})
    return results


def reference_calculate_metrics_cnnspot(df: pd.DataFrame) -> list[dict]:
    """Verbatim logic from analyse.py::calculate_metrics_for_cnnspot (source-matched reals)."""
    results = []
    synthetic_sources = df.query("label == 1").source.unique()
    for synthetic_source in synthetic_sources:
        df_eval = df.query(f"source == '{synthetic_source}'")  # both real+fake of that source
        ap = average_precision_score(y_true=df_eval.label, y_score=df_eval.label_prob)
        results.append({"source": synthetic_source, "average_precision": float(ap)})
    return results


def make_cnnspot_frame(seed: int = 1) -> pd.DataFrame:
    """CNNSpot-style frame: each source carries its own real+fake; plus real-only sources."""
    rng = np.random.default_rng(seed)
    rows = []
    for gen, shift, nf, nr in [
        ("progan", 2.5, 50, 60),
        ("stylegan", 1.5, 40, 55),
        ("biggan", 0.5, 30, 35),
    ]:
        for sc in rng.normal(shift, 1.0, nf):
            rows.append({"label": 1, "score": float(sc), "source": gen})
        for sc in rng.normal(0.0, 1.0, nr):
            rows.append({"label": 0, "score": float(sc), "source": gen})
    for sc in rng.normal(0.0, 1.0, 40):  # real-only source, must be ignored by matched pairing
        rows.append({"label": 0, "score": float(sc), "source": "imagenet"})
    return pd.DataFrame(rows)


def make_frame(seed: int = 0, n_real: int = 120, n_per_gen: int = 40) -> pd.DataFrame:
    """Synthetic predictions with 3 generators of varying separability + 2 architectures."""
    rng = np.random.default_rng(seed)
    gens = {
        "easy_gen": (3.0, "diffusion"),  # well separated -> high AP
        "medium_gen": (1.0, "diffusion"),
        "hard_gen": (0.2, "gan"),  # barely separated -> low AP
    }
    rows = []
    real_scores = rng.normal(0.0, 1.0, n_real)
    for sc in real_scores:
        rows.append({"label": 0, "score": float(sc), "source": "real", "architecture": "real"})
    for gen, (shift, arch) in gens.items():
        fake_scores = rng.normal(shift, 1.0, n_per_gen)
        for sc in fake_scores:
            rows.append({"label": 1, "score": float(sc), "source": gen, "architecture": arch})
    return pd.DataFrame(rows)


def test_per_generator_ap_matches_original_oracle():
    df = make_frame()
    ours = per_generator_ap(df).set_index("generator")["ap"].to_dict()

    ref_df = df.rename(columns={"score": "label_prob"})
    oracle = {r["source"]: r["average_precision"] for r in reference_calculate_metrics(ref_df)}

    assert set(ours) == set(oracle)
    for gen, ap in oracle.items():
        assert ours[gen] == pytest.approx(ap, abs=1e-9)


def test_per_generator_map_is_mean_of_oracle_aps():
    df = make_frame()
    oracle = [
        r["average_precision"]
        for r in reference_calculate_metrics(df.rename(columns={"score": "label_prob"}))
    ]
    assert per_generator_map(df) == pytest.approx(float(np.mean(oracle)), abs=1e-9)


def test_convention_a_differs_from_pooled_ap():
    """Sanity guard: per-generator mAP must not silently equal pooled AP."""
    df = make_frame()
    bundle = detection_metrics(df)
    assert bundle["mAP"] != pytest.approx(bundle["pooled_ap"], abs=1e-3)
    assert bundle["n_generators"] == 3
    assert bundle["n_real"] == 120


def test_degenerate_generator_returns_nan_and_is_skipped():
    """A generator with no reals available yields nan AP; mAP skips it (no crash)."""
    only_fakes = pd.DataFrame(
        {"label": [1, 1, 1], "score": [0.1, 0.2, 0.3], "source": ["g", "g", "g"]}
    )
    tbl = per_generator_ap(only_fakes)
    assert np.isnan(tbl["ap"]).all()
    assert np.isnan(per_generator_map(only_fakes))


def test_cnnspot_matched_pairing_matches_oracle():
    """Source-matched pairing reproduces calculate_metrics_for_cnnspot and ignores real-only sources."""
    df = make_cnnspot_frame()
    ours = per_generator_ap(df, real_pairing="matched").set_index("generator")["ap"].to_dict()
    oracle = {
        r["source"]: r["average_precision"]
        for r in reference_calculate_metrics_cnnspot(df.rename(columns={"score": "label_prob"}))
    }
    assert set(ours) == set(oracle) == {"progan", "stylegan", "biggan"}
    for gen, ap in oracle.items():
        assert ours[gen] == pytest.approx(ap, abs=1e-9)
    # matched pairing uses only same-source reals (e.g. biggan: 35), never the imagenet real-only set
    n_real = per_generator_ap(df, real_pairing="matched").set_index("generator")["n_real"].to_dict()
    assert n_real["biggan"] == 35


def test_shared_vs_matched_differ_on_cnnspot_frame():
    df = make_cnnspot_frame()
    assert per_generator_map(df, real_pairing="shared") != pytest.approx(
        per_generator_map(df, real_pairing="matched"), abs=1e-3
    )


def test_pairing_for_dataset():
    assert pairing_for_dataset("cnnspot") == "matched"
    assert pairing_for_dataset("cnnspot_clip_base_patch16") == "matched"
    assert pairing_for_dataset("synthclic") == "shared"
    assert pairing_for_dataset("synthbuster-plus") == "shared"
