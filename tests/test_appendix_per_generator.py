"""TableB — the regenerated appendix per-generator table (`tab:clip:test_results_detail`).

Two layers: the metric helper's grouping contract (always runs), and the shipped artifacts
(skipped when the snapshot has not been built with `--with-appendix`).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from clip_cues_research.analysis.metrics import (
    pairing_for_dataset,
    per_generator_accuracy,
    per_generator_ap,
)

ARTIFACTS = Path("reproduction/experiments/final_consolidation/TableB-per-generator/artifacts")
DETAIL = ARTIFACTS / "per_generator_detail.csv"
ACCEPTANCE = ARTIFACTS / "acceptance.csv"
SUMMARY = ARTIFACTS / "summary.json"


def _load_exporter():
    """`scripts/export/` is not a package, so load the exporter by path."""
    spec = importlib.util.spec_from_file_location(
        "export_per_generator_table", "scripts/export/export_per_generator_table.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


X = _load_exporter()

needs_artifacts = pytest.mark.skipif(
    not DETAIL.exists(),
    reason="run `make finalexp-data WITH_CFEVAL=1 WITH_APPENDIX=1 && make finalexp-tableb` first",
)


@pytest.fixture
def toy() -> pd.DataFrame:
    """Two generators, each with its own reals — so `matched` and `shared` differ."""
    return pd.DataFrame(
        {
            "label": [1, 1, 0, 0, 1, 1, 0, 0],
            "score": [0.9, 0.8, 0.4, 0.1, 0.6, 0.3, 0.7, 0.2],
            "source": ["a"] * 4 + ["b"] * 4,
        }
    )


# ── the grouping contract ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("pairing", ["shared", "matched"])
def test_accuracy_and_ap_use_the_same_groups(toy, pairing):
    """ACC beside AP is only meaningful if both describe the same population."""
    ap = per_generator_ap(toy, real_pairing=pairing).sort_values("generator")
    acc = per_generator_accuracy(toy, real_pairing=pairing).sort_values("generator")
    pd.testing.assert_frame_equal(
        ap[["generator", "n_fake", "n_real"]].reset_index(drop=True),
        acc[["generator", "n_fake", "n_real"]].reset_index(drop=True),
    )


def test_accuracy_matched_pairing_is_hand_checkable(toy):
    """`a`: fakes 0.9/0.8 correct, reals 0.4/0.1 correct -> 4/4. `b`: 0.6 ok, 0.3 miss, 0.7 miss, 0.2 ok."""
    acc = per_generator_accuracy(toy, real_pairing="matched").set_index("generator")["acc"]
    assert acc["a"] == pytest.approx(1.0)
    assert acc["b"] == pytest.approx(0.5)


def test_accuracy_threshold_is_honoured(toy):
    """A threshold above every score classifies everything real: ACC == the group's real fraction."""
    acc = per_generator_accuracy(toy, real_pairing="matched", threshold=1.0)
    assert list(acc["acc"]) == [0.5, 0.5]


def test_accuracy_rejects_unknown_pairing(toy):
    with pytest.raises(ValueError, match="real_pairing"):
        per_generator_accuracy(toy, real_pairing="paired")


# ── the shipped table ────────────────────────────────────────────────────────────────────────
@needs_artifacts
def test_table_shape_matches_the_manuscript_block_structure():
    """21 CNNSpot + 13 SynthBuster+ + 4 SynthCLIC generators, each for 4 training sets."""
    d = pd.read_csv(DETAIL)
    counts = d.groupby("test_corpus")["generator"].nunique().to_dict()
    assert counts == {"cnnspot": 21, "synthbuster-plus": 13, "synthclic": 4}
    assert len(d) == 38 * 4
    assert set(d.train_set) == {"cnnspot", "synthbuster-plus", "synthclic", "combined"}
    # section 4.3: `seeingdark` is present, with synthetics — it is absent from cnnspot-small.
    sd = d[(d.test_corpus == "cnnspot") & (d.generator == "seeingdark")]
    assert len(sd) == 4 and (sd.n_fake > 0).all()


@needs_artifacts
def test_real_pairing_is_the_paper_convention_per_corpus():
    """CNNSpot source-matched reals; SynthCLIC/SB+ one shared real pool for every generator."""
    d = pd.read_csv(DETAIL)
    for corpus, group in d.groupby("test_corpus"):
        assert group["real_pairing"].unique().tolist() == [pairing_for_dataset(corpus)]
    shared = d[d.test_corpus == "synthclic"]
    assert shared["n_real"].nunique() == 1  # the one clic2020 pool
    matched = d[(d.test_corpus == "cnnspot") & (d.train_set == "synthclic")]
    assert matched["n_real"].nunique() > 1  # each generator brings its own reals


@needs_artifacts
def test_acceptance_reproduces_the_authoritative_matrix():
    """Eleven cells to 2 dp; the twelfth is the recorded frame mismatch, and nothing else fails."""
    a = pd.read_csv(ACCEPTANCE)
    assert len(a) == 12
    failures = a[~a.agrees_2dp]
    assert failures.train_set.tolist() == ["combined"]
    assert failures.test_corpus.tolist() == ["cnnspot"]
    assert bool(failures.known_frame_mismatch.iloc[0])
    assert (a[a.agrees_2dp].delta.abs() < 0.005).all()


@needs_artifacts
def test_column_means_are_reproducible_from_the_shipped_rows():
    """The acceptance table must be an aggregate of the table, not an independent number."""
    d, a = pd.read_csv(DETAIL), pd.read_csv(ACCEPTANCE)
    got = d.groupby(["test_corpus", "train_set"])["ap"].mean().round(4)
    for _, r in a.iterrows():
        assert got[(r.test_corpus, r.train_set)] == pytest.approx(r.column_mean_ap, abs=5e-5)


@needs_artifacts
def test_summary_records_the_declared_conventions():
    """A cell without a stated threshold and pairing rule is not interpretable."""
    s = json.loads(SUMMARY.read_text())
    assert s["recipe"]["weight_decay"] == 0.01 and s["recipe"]["label_smoothing"] == 0.1
    assert s["recipe"]["seed"] == 123
    assert "no augmentation" in s["recipe"]["features"]
    assert s["real_pairing"] == {
        "cnnspot": "matched",
        "synthbuster-plus": "shared",
        "synthclic": "shared",
    }
    assert s["eval_frames"]["cnnspot"] == "pooler/cnnspot_full"
    assert s["acceptance"]["passed"]
    assert s["combined_head_check"]["max_abs_score_delta"] < 1e-5


@needs_artifacts
def test_refit_under_the_f1_trainer_reproduces_the_reused_cells():
    """The reused E3 predictions are only valid if E3's estimator is F1's estimator."""
    r = pd.read_csv(ARTIFACTS / "recipe_equivalence.csv")
    assert len(r) == 8
    assert r.delta.abs().max() < 0.02


# ── display names (reproduction/config/mappings.yaml) ─────────────────────────────────────────────────────
@needs_artifacts
def test_every_generator_has_a_display_name_from_its_own_corpus_map():
    """Per-corpus lookup, and no silent fall-through to the raw id."""
    detail = pd.read_csv(ARTIFACTS / "per_generator_detail.csv")
    maps = X.load_mappings()
    labels = X.generator_labels(maps, detail)
    assert len(labels) == 38
    for (corpus, raw), shown in labels.items():
        assert shown == maps[f"{X._norm(corpus)}_source_map"][raw]
    # `FLUX.1-dev` is spelled differently per corpus — a global map would collapse the two.
    assert labels[("synthclic", "FLUX.1-dev")] != labels[("synthbuster-plus", "FLUX.1-dev")]


@needs_artifacts
def test_unmapped_generator_is_a_hard_error():
    detail = pd.read_csv(ARTIFACTS / "per_generator_detail.csv")
    maps = X.load_mappings()
    del maps["cnnspot_source_map"]["progan"]
    with pytest.raises(KeyError, match="cnnspot/progan"):
        X.generator_labels(maps, detail)


def test_model_labels_come_from_the_config():
    labels = X.model_labels(X.load_mappings())
    assert labels == {
        "synthclic": "SynthCLIC",
        "synthbuster-plus": "SynthBuster+",
        "cnnspot": "CNNSpot",
        "combined": "Combined",
    }


@needs_artifacts
def test_exported_tex_uses_display_names_and_keeps_the_numbers():
    """The rendered table shows mapped names; the CSV keeps the raw ids as join keys."""
    tex = ARTIFACTS / "clip_per_generator_detail.tex"
    if not tex.exists():
        pytest.skip("run `make finalexp-tableb` to regenerate")
    body = tex.read_text()
    for shown in ("WhichFaceIsReal", "SD1.4", "MJv5", "SeeingDark", "LDM-200-CFG"):
        assert f"& {shown} &" in body
    for raw in ("whichfaceisreal", "stable-diffusion-1-4", "midjourney-v5", "ldm\\_200\\_cfg"):
        assert raw not in body
    shipped = pd.read_csv(ARTIFACTS / "clip_per_generator_detail_exported.csv")
    detail = pd.read_csv(ARTIFACTS / "per_generator_detail.csv")
    assert list(shipped.generator) == list(detail.generator)  # raw ids preserved
    assert shipped.ap.equals(detail.ap) and shipped.acc.equals(detail.acc)
