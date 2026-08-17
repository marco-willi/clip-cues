"""Unit tests for the per-pair cue-delta figure module (toy vocab/embedding fixtures, no data files)."""

import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from clip_cues_research.figures.paired_cue_delta import (
    load_cue_basis,
    load_polarity,
    pair_cue_deltas,
    redundancy_matrix,
    select_image_ids,
    select_top_cues,
)


@pytest.fixture
def toy(tmp_path):
    """A 3-pair SynthCLIC-shaped projected-embedding pickle + a 4-cue vocabulary."""
    rng = np.random.default_rng(0)
    ids = ["aaa", "bbb", "ccc"]
    # interleave sources so the code cannot rely on real/synth being contiguous or ordered
    rows = [(i, s) for i in ids for s in ("clic2020", "imagen3", "FLUX.1-dev")]
    df = pd.DataFrame(
        {
            "image_id": [r[0] for r in rows],
            "source": [r[1] for r in rows],
            "label": [0 if r[1] == "clic2020" else 1 for r in rows],
            "split": "test",
            "ds_name": "synthclic",
        }
    )
    emb = rng.normal(size=(len(rows), 8))
    proj = tmp_path / "proj.pkl"
    pickle.dump({"df": df, "embeddings": emb}, open(proj, "wb"))

    names = ["cue_a", "cue_b", "cue_c", "cue_d"]
    vocab = tmp_path / "vocab.pt"
    torch.save(
        {"vocabulary": names, "embeddings": torch.from_numpy(rng.normal(size=(4, 8)))}, vocab
    )
    return {"proj": str(proj), "vocab": str(vocab), "ids": ids, "emb": emb, "df": df}


def test_deltas_match_manual_cosine_difference(toy):
    pairs, D, names = pair_cue_deltas(
        gen="imagen3", split="test", proj_emb=toy["proj"], vocab=toy["vocab"]
    )
    assert names == ["cue_a", "cue_b", "cue_c", "cue_d"]
    assert list(pairs["image_id"]) == sorted(toy["ids"])
    assert D.shape == (3, 4)

    cues = torch.load(toy["vocab"], weights_only=False)["embeddings"].numpy()
    cues = cues / np.linalg.norm(cues, axis=1, keepdims=True)
    df, emb = toy["df"], toy["emb"]
    for k, iid in enumerate(pairs["image_id"]):
        r = df.index[(df.image_id == iid) & (df.source == "clic2020")][0]
        s = df.index[(df.image_id == iid) & (df.source == "imagen3")][0]
        u = emb[s] / np.linalg.norm(emb[s]) - emb[r] / np.linalg.norm(emb[r])
        assert np.allclose(D[k], u @ cues.T, atol=1e-12)


def test_pairs_index_the_requested_generator(toy):
    """Rows must point at the chosen synthetic source, not just any label-1 row."""
    df = toy["df"]
    for gen in ("imagen3", "FLUX.1-dev"):
        pairs, _, _ = pair_cue_deltas(
            gen=gen, split="test", proj_emb=toy["proj"], vocab=toy["vocab"]
        )
        for _, p in pairs.iterrows():
            assert df.loc[p.real_row, "source"] == "clic2020"
            assert df.loc[p.synth_row, "source"] == gen
            assert df.loc[p.real_row, "image_id"] == df.loc[p.synth_row, "image_id"] == p.image_id


def test_unknown_generator_raises(toy):
    with pytest.raises(ValueError, match="no clic2020"):
        pair_cue_deltas(gen="nope", split="test", proj_emb=toy["proj"], vocab=toy["vocab"])


def test_select_representative_ranks_by_alignment_to_the_mean_shift():
    pairs = pd.DataFrame({"image_id": ["a", "b", "c"]})
    # mean shift is [0.667, 0]: b lies on it, c is off-axis, a is nearly orthogonal to it
    D = np.array([[0.1, 1.0], [1.0, 0.0], [0.9, -1.0]])
    assert np.allclose(D.mean(axis=0), [2.0 / 3.0, 0.0])
    assert select_image_ids(pairs, D, n=1, mode="representative") == ["b"]
    assert select_image_ids(pairs, D, n=3, mode="representative") == ["b", "c", "a"]


def test_select_extreme_ranks_by_total_movement():
    pairs = pd.DataFrame({"image_id": ["a", "b", "c"]})
    D = np.array([[0.1, 0.0], [3.0, 4.0], [1.0, 0.0]])
    assert select_image_ids(pairs, D, n=2, mode="extreme") == ["b", "c"]


def test_pinned_id_comes_first_and_is_not_duplicated():
    pairs = pd.DataFrame({"image_id": ["a", "b", "c"]})
    D = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    out = select_image_ids(pairs, D, n=2, mode="extreme", pinned=["c"])
    assert out[0] == "c" and len(out) == len(set(out)) == 2
    # a pinned id that has no pair is dropped rather than crashing the figure
    assert select_image_ids(pairs, D, n=1, mode="extreme", pinned=["zz"])[0] in {"a", "b", "c"}


def test_unknown_selection_mode_raises():
    with pytest.raises(ValueError, match="unknown mode"):
        select_image_ids(pd.DataFrame({"image_id": ["a"]}), np.zeros((1, 2)), n=1, mode="wat")


def test_top_cues_without_dedup_is_plain_ranking_by_abs_delta():
    delta = np.array([0.5, -0.9, 0.1, 0.7])
    chosen, dropped = select_top_cues(delta, k=3, redundancy=None)
    assert chosen == [1, 3, 0] and dropped == []


def test_dedup_suppresses_the_redundant_cluster_and_backfills():
    """Cues 0-2 are a collinear cluster: only the strongest should survive, and 3/4 fill the slots."""
    delta = np.array([0.9, 0.85, 0.8, 0.4, 0.3])
    R = np.eye(5)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        R[i, j] = R[j, i] = 0.9
    chosen, dropped = select_top_cues(delta, k=3, redundancy=R, threshold=0.5)
    assert chosen == [0, 3, 4]
    assert [(names, blocker) for names, blocker, _ in dropped] == [(1, 0), (2, 0)]
    assert all(r == pytest.approx(0.9) for _, _, r in dropped)


def test_dedup_attributes_a_drop_to_its_strongest_blocker():
    delta = np.array([1.0, 0.9, 0.8])
    R = np.eye(3)
    R[0, 2] = R[2, 0] = 0.6
    R[1, 2] = R[2, 1] = 0.95  # cue 1 is the stronger blocker for cue 2
    chosen, dropped = select_top_cues(delta, k=3, redundancy=R, threshold=0.5)
    assert chosen == [0, 1]
    assert dropped == [(2, 1, pytest.approx(0.95))]


def test_dedup_threshold_is_inclusive_at_the_boundary():
    delta = np.array([1.0, 0.9])
    R = np.array([[1.0, 0.5], [0.5, 1.0]])
    assert select_top_cues(delta, k=2, redundancy=R, threshold=0.5)[0] == [0, 1]
    assert select_top_cues(delta, k=2, redundancy=R, threshold=0.49)[0] == [0]


def test_dedup_uses_absolute_redundancy_so_anticorrelated_cues_also_collapse():
    delta = np.array([1.0, 0.9])
    R = np.array([[1.0, -0.9], [-0.9, 1.0]])
    assert select_top_cues(delta, k=2, redundancy=R, threshold=0.5)[0] == [0]


def test_redundancy_matrix_modes(toy):
    _, D, names = pair_cue_deltas(
        gen="imagen3", split="test", **{"proj_emb": toy["proj"], "vocab": toy["vocab"]}
    )
    _, C = load_cue_basis(toy["vocab"])
    assert redundancy_matrix(D, C, "none") is None
    assert np.allclose(redundancy_matrix(D, C, "cosine"), C @ C.T)
    assert np.allclose(np.diag(redundancy_matrix(D, C, "delta_corr")), 1.0)
    with pytest.raises(ValueError, match="unknown redundancy metric"):
        redundancy_matrix(D, C, "wat")


def test_dedup_actually_breaks_up_the_capture_cue_cluster():
    """Regression for the real failure mode: 7 of 8 slots taken by collinear '*_cues' cues."""
    names, C = load_cue_basis()
    pairs, D, _ = pair_cue_deltas(gen="FLUX.1-dev", split="test")
    delta = D[list(pairs["image_id"]).index("01d8472427bf120a4574ee3dbb3f1234")]

    plain, _ = select_top_cues(delta, k=8, redundancy=None)
    deduped, dropped = select_top_cues(
        delta, k=8, redundancy=redundancy_matrix(D, C, "delta_corr"), threshold=0.5
    )
    n_capture = lambda idx: sum(names[j].endswith("_cues") for j in idx)  # noqa: E731
    assert n_capture(plain) >= 6, "fixture no longer exhibits the clustering this guards against"
    assert n_capture(deduped) <= 2
    assert len(deduped) == 8 and len(set(deduped)) == 8
    assert dropped, "the suppressed cues must be reported, not silently dropped"


def test_polarity_phrases_cover_the_canonical_vocabulary():
    """The real vocabulary CSV must label every canonical cue, else bars fall back to bare names."""
    pol = load_polarity()
    names = list(
        torch.load("data/embeddings/vocab_canon/antonyms.pt", weights_only=False)["vocabulary"]
    )
    assert set(names) <= set(pol)


def test_missing_polarity_csv_degrades_gracefully(tmp_path):
    assert load_polarity(tmp_path / "does_not_exist.csv") == {}
