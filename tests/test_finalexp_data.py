"""Snapshot integrity + the anti-mixup guards (F1-F7, Step 1/2).

These tests exist because `data/` contains artifacts that are *silently substitutable*: three
SynthCLIC pooler pickles with identical shape/splits/id-order, and a retracted W-squared vocabulary
that is shape-identical to the canonical one. Shape assertions cannot catch a wrong path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from clip_cues_research.finalexp import data as D

pytestmark = pytest.mark.skipif(
    not D.MANIFEST.exists(),
    reason="snapshot not built (run scripts/finalexp/build_data_snapshot.py)",
)


def test_manifest_loads_and_ids_are_unique():
    recs = D.load_manifest()
    assert recs, "manifest has no artifacts"
    doc = json.loads(D.MANIFEST.read_text())
    ids = [a["id"] for a in doc["artifacts"]]
    assert len(ids) == len(set(ids)), "duplicate artifact ids in the manifest"


def test_every_artifact_verifies_on_load():
    """resolve() re-hashes the file; a mismatch must raise rather than silently return."""
    for aid in D.load_manifest():
        assert D.resolve_any(aid).exists()


def test_checksum_mismatch_is_fatal(tmp_path, monkeypatch):
    """The core guard: a swapped file must fail, even with a plausible name and shape.

    This also pins the in-process verification cache: because the cached stamp includes the
    manifest's *expected* sha, a changed expectation must force a re-check rather than inherit an
    earlier "verified" verdict from another test in the same session. Caching on mtime+size alone
    silently passed this test.
    """
    recs = D.load_manifest()
    rec = recs["vocab/antonyms"]
    if not rec.full_path.exists():
        pytest.skip("built snapshot not present (this checks the built format; see the release twin)")
    doc = json.loads(D.MANIFEST.read_text())
    for a in doc["artifacts"]:
        if a["id"] == "vocab/antonyms":
            a["sha256"] = "0" * 64  # simulate the file having been replaced
    fake = tmp_path / "manifest.json"
    fake.write_text(json.dumps(doc))
    monkeypatch.setattr(D, "MANIFEST", fake)
    with pytest.raises(ValueError, match="CHECKSUM MISMATCH"):
        D.resolve("vocab/antonyms")
    assert rec.space == D.SPACE_CANON


def test_retracted_w2_space_cannot_be_loaded(tmp_path, monkeypatch):
    """A W-squared vocabulary must be rejected by require_space, not merely undocumented.

    This is the 2026-07-17 double-projection bug encoded as a test: the retracted file has the same
    (168, 768) float32 shape as the canonical one, so only the declared space distinguishes them.
    """
    doc = json.loads(D.MANIFEST.read_text())
    for a in doc["artifacts"]:
        if a["id"] == "vocab/antonyms":
            a["space"] = D.SPACE_W2_LEGACY
    fake = tmp_path / "manifest.json"
    fake.write_text(json.dumps(doc))
    monkeypatch.setattr(D, "MANIFEST", fake)
    with pytest.raises(ValueError, match="RETRACTED"):
        D.require_space("vocab/antonyms", D.SPACE_CANON)


def test_canonical_vocab_is_not_the_retracted_file():
    """Concrete identity check against the known-bad neighbour."""
    legacy = Path("data/embeddings/antonyms_diff_embeddings.pt")
    if not legacy.exists():
        pytest.skip("legacy W-squared file not present")
    canon_sha = D.record("vocab/antonyms").sha256
    assert canon_sha != D.sha256(legacy), "snapshot holds the RETRACTED W-squared vocabulary"


def test_pooler_frames_have_expected_geometry():
    for ds, n in (("synthclic", 10815), ("cnnspot", 8000), ("synthbuster-plus", 13999)):
        f = D.get_frame(f"pooler/{ds}", expected_space=D.SPACE_POOLER)
        assert f.emb.shape == (n, 1024)
        assert len(f.df) == n
        assert {"image_id", "label", "split", "source"} <= set(f.df.columns)


def test_derived_projected_matches_pooler_rows():
    """Derived features must stay positionally aligned with their source frame.

    `image_id` is non-unique in SynthCLIC/SynthBuster, so every join in this package is positional;
    a row-order drift would silently mislabel every downstream result.
    """
    for ds in ("synthclic", "cnnspot", "synthbuster-plus"):
        p = D.get_frame(f"pooler/{ds}", expected_space=D.SPACE_POOLER)
        e = D.get_frame(f"projected/{ds}", expected_space=D.SPACE_CANON)
        assert e.emb.shape == (p.emb.shape[0], 768)
        assert (p.df["image_id"].astype(str).values == e.df["image_id"].astype(str).values).all()
        assert (p.df["split"].values == e.df["split"].values).all()
        assert (p.df["label"].values == e.df["label"].values).all()


def test_cue_scores_align_and_are_bounded():
    scores = D.get_npz("cue_scores/synthclic__antonyms")
    frame = D.get_frame("projected/synthclic", expected_space=D.SPACE_CANON)
    assert scores["scores"].shape == (len(frame.df), 168)
    assert len(scores["cues"]) == 168
    # cue scores are inner products of a unit vector with unit cue directions
    assert np.abs(scores["scores"]).max() <= 2.0


def test_derived_artifacts_survive_a_manifest_rebuild():
    """Rebuilding the snapshot must MERGE, not replace.

    `build_data_snapshot.py` owns only the copied sources; `prepare_features.py` and
    `export_f5_rankings.py` register their outputs into the same manifest. A rebuild that rewrote
    the artifact list silently de-registered them — the files stayed on disk but every
    `get_*("projected/…")` started failing with "Unknown snapshot id". This pins the merge.
    """
    recs = D.load_manifest()
    expected = {f"projected/{ds}" for ds in ("synthclic", "cnnspot", "synthbuster-plus")}
    expected |= {
        f"cue_scores/{ds}__antonyms" for ds in ("synthclic", "cnnspot", "synthbuster-plus")
    }
    missing = sorted(expected - set(recs))
    assert not missing, f"derived artifacts lost from the manifest: {missing}"
    for aid in expected:
        assert recs[aid].raw.get("derived_from"), f"{aid} has no derived_from provenance"


def test_derived_vs_cached_crosscheck_recorded():
    """The equivalence check must exist and carry its numbers, whatever they say."""
    p = D.SNAPSHOT / "reference" / "derived_vs_cached_crosscheck.json"
    assert p.exists(), "run scripts/finalexp/prepare_features.py"
    checks = json.loads(p.read_text())["checks"]
    assert checks
    for c in checks:
        assert c["cosine_median"] > 0.99, f"{c['dataset']}: derived/cached cosine too low"
        assert "auroc_delta" in c and "within_benchmark" in c


def test_f1_regression_anchor_present():
    anchor = D.get_json("reference/e3_seed123")
    assert anchor["backbone"] == "clip_large_patch14"
    assert anchor["train_dataset"] == anchor["eval_dataset"] == "synthclic"
    assert abs(anchor["mAP"] - 0.9239) < 1e-3
    assert abs(anchor["auroc"] - 0.9227) < 1e-3


# ── the enforcement guard ────────────────────────────────────────────────────────────────────
FORBIDDEN = re.compile(r"""["'](?:\./)?(?:data/(?:embeddings|checkpoints|vocabularies))/""")
ALLOWED_FILES = {"data.py", "build_data_snapshot.py",
    # writes the legacy data/embeddings/ paths from the verified release — that is its job
    "fetch_snapshot.py",
}


def test_finalexp_code_never_uses_literal_input_paths():
    """F-experiment code must reach inputs only through the manifest.

    A literal `data/embeddings/...` path anywhere else would bypass checksum verification and the
    space assertion — exactly the failure mode the snapshot exists to prevent. `data.py` defines
    the access layer and `build_data_snapshot.py` declares the copy sources, so both are exempt.
    """
    roots = [Path("scripts/finalexp"), Path("src/clip_cues_research/finalexp")]
    offenders = []
    for root in roots:
        for path in root.rglob("*.py"):
            if path.name in ALLOWED_FILES:
                continue
            for i, line in enumerate(path.read_text().splitlines(), 1):
                code = line.split("#", 1)[0]
                if FORBIDDEN.search(code):
                    offenders.append(f"{path}:{i}: {line.strip()}")
    assert not offenders, "literal input paths bypass the manifest:\n" + "\n".join(offenders)


def test_release_checksum_mismatch_is_fatal(tmp_path, monkeypatch):
    """The same guard for a *fetched* snapshot, which only has the released .npz form.

    A download is verified against ``release_sha256`` rather than the built artifact's hash — the
    two necessarily differ for converted files — so the release path needs its own proof that a
    corrupted file is refused instead of silently loaded.
    """
    if not D.RELEASE_MANIFEST.exists():
        pytest.skip("no release materialised here")
    doc = json.loads(D.RELEASE_MANIFEST.read_text())
    aid = next(a for a in doc["artifacts"] if a.startswith("pooler/"))
    if (D.SNAPSHOT / D.record(aid).path).exists():
        pytest.skip("built format present; resolve_any prefers it")
    doc["artifacts"][aid]["release_sha256"] = "0" * 64
    fake = tmp_path / "release_manifest.json"
    fake.write_text(json.dumps(doc))
    monkeypatch.setattr(D, "RELEASE_MANIFEST", fake)
    with pytest.raises(ValueError, match="CHECKSUM MISMATCH"):
        D.resolve_any(aid)
