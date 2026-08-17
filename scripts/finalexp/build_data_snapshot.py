#!/usr/bin/env python
"""Step 1: build the frozen, checksummed input snapshot for F1-F7 in ``reproduction/experiments/data/``.

Copies every artifact F1-F7 consumes out of `data/` and `results/`, records each in
``manifest.json`` (sha256, declared embedding space, shape, split counts, fingerprint, provenance,
used_by), and writes the human-readable ``MANIFEST.md`` plus ``EXCLUDED.md`` — the list of
plausible-but-wrong neighbours that were deliberately NOT copied, with reasons.

Rationale (PLAN_FINAL_CONSOLIDATION.md §Context 5): `data/` holds three SynthCLIC pooler pickles with
identical shape/splits/id-order and a retracted W-squared vocabulary shape-identical to the canonical
one. Nothing but a checksum distinguishes them.

Copies are real files, not symlinks: a snapshot that changes when `data/` changes cannot certify
what a run consumed.

    uv run python scripts/finalexp/build_data_snapshot.py             # default snapshot (~260 MB)
    uv run python scripts/finalexp/build_data_snapshot.py --with-cfeval   # + CF-Eval (206 MB)
    uv run python scripts/finalexp/build_data_snapshot.py --with-appendix # + TableB inputs (610 MB)
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from clip_cues_research.finalexp.data import (
    MANIFEST,
    SNAPSHOT,
    SPACE_CANON,
    SPACE_NA,
    SPACE_POOLER,
    sha256,
)
from clip_cues_research.finalexp.snapshot import describe, git_commit, render_markdown

SNAPSHOT_VERSION = "1.0.0"

# ── what goes in ─────────────────────────────────────────────────────────────────────────────
# (id, source, dest, kind, space, used_by, provenance note)
SOURCES: list[tuple[str, str, str, str, str, list[str], str]] = [
    (
        "pooler/synthclic",
        "data/embeddings/synthclic_clip_large_patch14.pkl",
        "embeddings/pooler_l14/synthclic.pkl",
        "pooler_embeddings",
        SPACE_POOLER,
        ["F1", "F2", "F3", "F4", "F5", "F6", "F7"],
        "scripts/extract/extract_embeddings.py, frozen CLIP ViT-L/14-336 pooler_output; the "
        "canonical SynthCLIC frame behind results/e3_xdataset and E9-E12",
    ),
    (
        "pooler/cnnspot",
        "data/embeddings/cnnspot_clip_large_patch14.pkl",
        "embeddings/pooler_l14/cnnspot.pkl",
        "pooler_embeddings",
        SPACE_POOLER,
        ["F5", "F6"],
        "scripts/extract/extract_embeddings.py; train split = ds_train_very_small (2,000), "
        "confirmed 2026-08-08 (config-audit.md §A)",
    ),
    (
        "pooler/synthbuster-plus",
        "data/embeddings/synthbuster-plus_clip_large_patch14.pkl",
        "embeddings/pooler_l14/synthbuster-plus.pkl",
        "pooler_embeddings",
        SPACE_POOLER,
        ["F6"],
        "scripts/extract/extract_embeddings.py; F6 uses train/val ONLY (SB+ test is closed under "
        "EXTERNAL_VALIDATION_PROTOCOL.md)",
    ),
    (
        "projection/wp_l14_336",
        "data/embeddings/clip_l14_336_visual_projection.npy",
        "projection/clip_l14_336_visual_projection.npy",
        "projection_matrix",
        SPACE_NA,
        ["F3", "F4", "F6"],
        "openai/clip-vit-large-patch14-336 visual_projection weight (768x1024); image_embeds = "
        "visual_projection(pooler_output) exactly",
    ),
    (
        "vocab/antonyms",
        "data/embeddings/vocab_canon/antonyms.pt",
        "vocabularies/antonyms.pt",
        "cue_vocabulary",
        SPACE_CANON,
        ["F1", "F2", "F3", "F4", "F6", "F7"],
        "scripts/interpret/embed_vocab.py, CANONICAL re-embed after the 2026-07-17 double-projection "
        "fix; the paper's published 168-cue antonym set",
    ),
    (
        "vocab/clipiqa_full",
        "data/embeddings/vocab_canon/clipiqa_full.pt",
        "vocabularies/clipiqa_full.pt",
        "cue_vocabulary",
        SPACE_CANON,
        ["Fig9a"],
        "the 16 CLIP-IQA attribute directions (Wang et al. 2022) behind appendix Figure 10; "
        "canonical text space, re-embedded after the 2026-07-17 double-projection fix",
    ),
    (
        "vocab/clipiqa_full_poles",
        "data/embeddings/vocab_canon/clipiqa_full_poles.pt",
        "vocabularies/clipiqa_full_poles.pt",
        "cue_vocabulary",
        SPACE_CANON,
        ["Fig9a"],
        "the 32 positive/negative pole embeddings of the same 16 CLIP-IQA attributes; "
        "figures/clipiqa.py reads both this and vocab/clipiqa_full",
    ),
    (
        "vocab_terms/antonyms",
        "data/vocabularies/antonyms.csv",
        "vocabularies/antonyms.csv",
        "vocabulary_terms",
        SPACE_NA,
        ["F1", "F2", "F3", "F4", "F6", "F7"],
        "term list + poles for vocab/antonyms (row order matches the .pt)",
    ),
    (
        "ckpt/linear_probe_synthclic",
        "data/checkpoints/linear_probe_synthclic.ckpt",
        "checkpoints/linear_probe_synthclic.ckpt",
        "checkpoint",
        SPACE_NA,
        ["F7"],
        "PUBLISHED k=1 probe, trained end-to-end WITH augmentation (RandomResizedCrop 0.5-1.0 -> "
        "512 + HFlip + JPEG 65-100); F7 bridge target only",
    ),
    (
        "ckpt/linear_probe_cnnspot",
        "data/checkpoints/linear_probe_cnnspot.ckpt",
        "checkpoints/linear_probe_cnnspot.ckpt",
        "checkpoint",
        SPACE_NA,
        ["F7"],
        "PUBLISHED CNNSpot k=1 probe (augmented); F7 bridge target only",
    ),
    (
        "ckpt/clip_orthogonal_synthclic",
        "data/checkpoints/clip_orthogonal_synthclic.ckpt",
        "checkpoints/clip_orthogonal_synthclic.ckpt",
        "checkpoint",
        SPACE_NA,
        ["F7"],
        "PUBLISHED k=8 ActivationOrthogonalityHead (augmented); F7 bridge target only",
    ),
    (
        "reference/e3_seed123",
        "results/e3_xdataset/clip_large_patch14__synthclic__to__synthclic/202606242002/metrics.json",
        "reference/e3_xdataset_synthclic_seed123_metrics.json",
        "reference_metrics",
        SPACE_NA,
        ["F1"],
        "scripts/run/run_linear_probe.py seed 123 under the matched recipe — F1's regression anchor "
        "(mAP 0.9239, AUROC 0.9227); the number in the manuscript's Table A",
    ),
    (
        "reference/projected_cached_synthclic",
        "data/embeddings/synthclic_projected_embeddings.pkl",
        "reference/projected_cached_synthclic.pkl",
        "projected_embeddings_reference",
        SPACE_CANON,
        ["F3-crosscheck"],
        "SEPARATELY EXTRACTED projected embeddings (CLIPVisionModelWithProjection). CROSS-CHECK "
        "ONLY — never fitted on; F3/F4/F6 use the derived features (both-sides-derived rule)",
    ),
    (
        "reference/projected_cached_cnnspot",
        "data/embeddings/cnnspot_projected_embeddings.pkl",
        "reference/projected_cached_cnnspot.pkl",
        "projected_embeddings_reference",
        SPACE_CANON,
        ["F3-crosscheck"],
        "SEPARATELY EXTRACTED projected embeddings; CROSS-CHECK ONLY",
    ),
]

# ── the appendix per-generator table (TableB), behind --with-appendix ────────────────────────
# Table 13 of the manuscript needs two populations no F-experiment touches — the *combined* train
# corpus and the *full* CNNSpot benchmark test set — plus the per-image predictions the E3
# cross-dataset matrix was aggregated from. They are heavy (610 MB) and used by exactly one table,
# so they stay out of the default snapshot.
E3_PRED = "results/e3_xdataset/predictions"

# (train, eval, run timestamp). The **second** E3 generation (2026-06-24 20:0x) is canonical — it
# is the one `reference/e3_seed123` points at — and it is also the generation that evaluates
# CNNSpot on the FULL benchmark test set. The combined rows only exist in the 2026-06-30/07-01
# generation; the later of its two identical re-saves is taken.
E3_CELLS: list[tuple[str, str, str]] = [
    ("cnnspot", "cnnspot", "202606242003"),
    ("cnnspot", "synthbuster-plus", "202606242003"),
    ("cnnspot", "synthclic", "202606242003"),
    ("synthbuster-plus", "cnnspot", "202606242002"),
    ("synthbuster-plus", "synthbuster-plus", "202606242002"),
    ("synthbuster-plus", "synthclic", "202606242002"),
    ("synthclic", "cnnspot", "202606242002"),
    ("synthclic", "synthbuster-plus", "202606242002"),
    ("synthclic", "synthclic", "202606242002"),
    ("combined", "synthbuster-plus", "202607012112"),
    ("combined", "synthclic", "202607012112"),
]

APPENDIX: list[tuple[str, str, str, str, str, list[str], str]] = [
    (
        "pooler/combined",
        "data/embeddings/combined_clip_large_patch14.pkl",
        "embeddings/pooler_l14/combined.pkl",
        "pooler_embeddings",
        SPACE_POOLER,
        ["TableB"],
        "scripts/extract/extract_embeddings.py; the union frame (32,814 = 8,000 CNNSpot + 13,999 "
        "SB+ + 10,815 SynthCLIC) with split assignments identical to the three per-corpus frames. "
        "Re-admitted 2026-08-09: Table 13's Combined column cannot be produced without it",
    ),
    (
        "pooler/cnnspot_full",
        "data/embeddings_box/cnnspot_clip_large_patch14.pkl",
        "embeddings/pooler_l14/cnnspot_full.pkl",
        "pooler_embeddings",
        SPACE_POOLER,
        ["TableB"],
        "scripts/extract/extract_embeddings.py on the FULL CNNSpot benchmark (112,310 imgs; test = "
        "108,310 / 21 generators, seeingdark included). NOT `pooler/cnnspot`, which is the 8,000-img "
        "cnnspot-small frame (test 4,000 / 20 generators) F5/F6 train on",
    ),
    (
        "ckpt/linear_probe_combined_matched",
        "data/checkpoints/linear_probe_combined.ckpt",
        "checkpoints/linear_probe_combined_matched.ckpt",
        "checkpoint",
        SPACE_NA,
        ["TableB"],
        "The MATCHED (no-augmentation) combined head saved by scripts/run/run_linear_probe.py "
        "--save-checkpoint on 2026-07-01; it reproduces the combined E3 prediction parquets to "
        "float32 (1.2e-07). NOT a published augmented checkpoint — it overwrote the published "
        "`linear_probe_combined.ckpt` in place, which is why it is registered under a distinct id",
    ),
] + [
    (
        f"e3pred/{train}__to__{ev}",
        f"{E3_PRED}/clip_large_patch14__{train}__to__{ev}__{ts}.parquet",
        f"e3_predictions/{train}__to__{ev}.parquet",
        "predictions",
        SPACE_NA,
        ["TableB"],
        f"scripts/run/run_linear_probe.py seed 123, matched recipe (Adam lr 1e-3, wd 0.01, ls 0.1, "
        f"bs 64, <=200 epochs, patience 5, frozen cached pooler, NO augmentation) trained on "
        f"{train}, scored on the {ev} test split; run {ts}. These per-image scores are what "
        f"`reproduction/revision_export/tables/e3_cross_matrix_mAP.csv` aggregates",
    )
    for train, ev, ts in E3_CELLS
]

CFEVAL = (
    "pooler/cf_eval",
    "data/embeddings/communityforensics_l14_eval.pkl",
    "embeddings/pooler_l14/cf_eval.pkl",
    "pooler_embeddings",
    SPACE_POOLER,
    ["Step11-tableA"],
    "scripts/extract/extract_cf_eval_embeddings.py (Lambda A10, T0 2026-06-28); 51,836 imgs / 21 "
    "generators. Used ONLY for the Table A CF-Eval cell",
)

# ── what stays out, and why ──────────────────────────────────────────────────────────────────
EXCLUDED: list[tuple[str, str]] = [
    (
        "data/embeddings/synthclic_l14_local.pkl",
        "A SEPARATE local re-extraction of the same SynthCLIC images: identical shape (10815, 1024), "
        "identical splits and image_id order, per-image cosine 0.999992 vs the canonical file — but "
        "**max abs difference 0.23**, so it is not the same artifact. Using it would move F1's "
        "regression anchor. Canonical = `pooler/synthclic`.",
    ),
    (
        "data/embeddings/synthclic_embeddings.pkl",
        "Same shape class, but **cannot be unpickled in this environment** (pandas block-manager "
        "incompatibility from a py3.10-written cache; the known embeddings-pickle-portability "
        "issue). Superseded by `pooler/synthclic`.",
    ),
    (
        "data/embeddings/antonyms_diff_embeddings.pt",
        "**RETRACTED — double-projected (W-squared) text space** (bug fixed 2026-07-17). Shape "
        "(168, 768) float32, i.e. *indistinguishable by shape* from the canonical "
        "`vocab/antonyms`. This is the single most dangerous file in the repo for this work.",
    ),
    (
        "data/embeddings/antonyms_embeddings.pt",
        "RETRACTED W-squared text space (336 poles). Canonical equivalent: vocab_canon/antonyms_poles.pt.",
    ),
    (
        "data/embeddings/textspan_embeddings.pt",
        "RETRACTED W-squared text space. Canonical equivalent: vocab_canon/textspan.pt (not needed by F1-F7).",
    ),
    (
        "data/embeddings/vocab_pool/*",
        "Pre-fix vocabulary pool (W-squared space throughout). Superseded by data/embeddings/vocab_canon/.",
    ),
    (
        "data/embeddings/combined_clip_large_patch14.pkl",
        "Combined-dataset frame; no F1–F7 experiment trains on the combined corpus. **Re-admitted "
        "2026-08-09 under `--with-appendix`** as `pooler/combined`: Table 13's Combined column (76 "
        "cells) cannot be produced without it. Still excluded from the default snapshot.",
    ),
    (
        "data/checkpoints/linear_probe_combined.ckpt (as a *published* checkpoint)",
        "The file at this path is **not** the published augmented combined probe — the 2026-07-01 "
        "combined E3 run overwrote it with the matched no-augmentation head (verified: it "
        "reproduces the combined E3 prediction parquets to 1.2e-07). It is registered under "
        "`ckpt/linear_probe_combined_matched` so nothing can quote it as an augmented bridge "
        "target alongside `ckpt/linear_probe_{synthclic,cnnspot}`, which genuinely are published.",
    ),
    (
        "results/e3_xdataset/predictions/*__202606240*.parquet (the FIRST E3 generation)",
        "The 2026-06-24 04:4x runs are the same trained heads as the canonical 20:0x generation "
        "(scores agree to 1.5e-07) but evaluate CNNSpot on **cnnspot-small** (4,000 imgs, 20 "
        "generators) instead of the full benchmark test set (108,310, 21). Same file-name pattern, "
        "different population — registered ids pin the canonical generation by sha256.",
    ),
    (
        "data/embeddings/*_clip_base_patch{16,32}.pkl",
        "B/16 and B/32 backbones — E3's additional-backbone question, out of scope for the "
        "consolidation (which is entirely about the L/14 canonical detector).",
    ),
    (
        "HF image data (data/hf_cache)",
        "Image pixels are far too large to snapshot. F5 instead registers its **ranking** "
        "(ranked_scores_{dataset}.csv, sha256'd) so the montage is reproducible from the ranking "
        "plus the HF dataset id + revision.",
    ),
    (
        "data/checkpoints/{cm_antonyms_*,clip_orthogonal_{cnnspot,combined,synthbuster}}.ckpt",
        "Concept models and the other orthogonal heads are not F1-F7 bridge targets.",
    ),
]


def carry_over_derived() -> list[dict]:
    """Preserve manifest records this script does not own.

    ``prepare_features.py`` (derived 768-d frames, cue scores) and ``export_f5_rankings.py``
    (rankings) register their outputs into the same manifest. A rebuild must **merge** rather than
    replace, or it silently de-registers them: the files stay on disk but every
    ``data.get_*("projected/…")`` starts failing with "Unknown snapshot id". Records whose file has
    since changed or vanished are dropped, so the merge cannot resurrect a stale entry.
    """
    if not MANIFEST.exists():
        return []
    kept = []
    for rec in json.loads(MANIFEST.read_text()).get("artifacts", []):
        if not rec.get("derived_from") and rec.get("source_path"):
            continue  # owned by SOURCES below; will be regenerated
        path = SNAPSHOT / rec["path"]
        if path.exists() and sha256(path) == rec["sha256"]:
            kept.append(rec)
            print(f"  keep {rec['id']:38s} (derived, carried over)")
        else:
            print(f"  DROP {rec['id']:38s} (file missing or changed — regenerate it)")
    return kept


MIRROR_MARKER = Path("data/embeddings/.fetched-mirror")


def refuse_if_mirrored() -> None:
    """Refuse to rebuild from files that `make finalexp-fetch` wrote.

    The fetch rewrites the legacy ``data/embeddings/`` paths from the released ``.npz``. Those
    copies hold the same numbers but different bytes, so rebuilding the snapshot from them changes
    every artifact hash and breaks the `derived_from` chain — silently, because the arrays are
    equal. Rebuild only from the original extraction outputs.
    """
    if MIRROR_MARKER.exists():
        raise SystemExit(
            f"Refusing to build: {MIRROR_MARKER} shows these sources were written by\n"
            f"  make finalexp-fetch\n"
            + "".join(f"    {line}\n" for line in MIRROR_MARKER.read_text().split() if line)
            + "Rebuilding from them would re-hash every artifact. Either use the fetched snapshot\n"
            "as-is (make finalexp-verify), or restore the original extraction outputs and delete\n"
            f"{MIRROR_MARKER}."
        )


def build(with_cfeval: bool, with_appendix: bool = False) -> dict:
    refuse_if_mirrored()
    sources = (
        list(SOURCES) + ([CFEVAL] if with_cfeval else []) + (APPENDIX if with_appendix else [])
    )
    artifacts = carry_over_derived()
    for aid, src, dest, kind, space, used_by, note in sources:
        src_p, dest_p = Path(src), SNAPSHOT / dest
        if not src_p.exists():
            raise FileNotFoundError(f"Source missing for {aid!r}: {src_p}")
        dest_p.parent.mkdir(parents=True, exist_ok=True)
        src_sha = sha256(src_p)
        if not (dest_p.exists() and sha256(dest_p) == src_sha):
            print(f"  copy {aid:38s} {src_p}  ->  {dest_p}")
            shutil.copy2(src_p, dest_p)
        else:
            print(f"  keep {aid:38s} (already identical)")
        dest_sha = sha256(dest_p)
        assert dest_sha == src_sha, f"copy differs from source for {aid}"
        artifacts.append(
            {
                "id": aid,
                "path": dest,
                "source_path": src,
                "source_sha256": src_sha,
                "sha256": dest_sha,
                "bytes": dest_p.stat().st_size,
                "kind": kind,
                "space": space,
                "used_by": used_by,
                "provenance": note,
                "copied_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **describe(dest_p, kind),
            }
        )
    return {
        "version": SNAPSHOT_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_commit(),
        "plan": ".claude/plans/PLAN_FINAL_CONSOLIDATION.md",
        "builder": "scripts/finalexp/build_data_snapshot.py",
        "artifacts": artifacts,
    }


def render_excluded() -> str:
    lines = [
        "# `reproduction/experiments/data/` — deliberately excluded artifacts",
        "",
        'The other half of "no mixup": every plausible-but-wrong neighbour of a snapshot artifact,',
        "listed with the reason it is **not** used. Exclusion here is a recorded decision, not an",
        "oversight. Generated by `scripts/finalexp/build_data_snapshot.py`.",
        "",
    ]
    for path, reason in EXCLUDED:
        lines += [f"### `{path}`", "", reason, ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--with-cfeval",
        action="store_true",
        help="also copy the 206 MB CF-Eval pooler frame (needed only for the Step 11 Table A fix)",
    )
    ap.add_argument(
        "--with-appendix",
        action="store_true",
        help="also copy the 610 MB TableB inputs: the combined + full-CNNSpot pooler frames, the "
        "matched combined head, and the 11 E3 per-image prediction parquets",
    )
    args = ap.parse_args()

    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    print(
        f"Building snapshot in {SNAPSHOT}/ "
        f"(with_cfeval={args.with_cfeval}, with_appendix={args.with_appendix})"
    )
    doc = build(args.with_cfeval, args.with_appendix)
    MANIFEST.write_text(json.dumps(doc, indent=2) + "\n")
    (SNAPSHOT / "MANIFEST.md").write_text(render_markdown(doc))
    (SNAPSHOT / "EXCLUDED.md").write_text(render_excluded())

    total = sum(a["bytes"] for a in doc["artifacts"])
    print(f"\n{len(doc['artifacts'])} artifacts, {total / 1e6:.1f} MB")
    print(f"  {MANIFEST}\n  {SNAPSHOT / 'MANIFEST.md'}\n  {SNAPSHOT / 'EXCLUDED.md'}")


if __name__ == "__main__":
    main()
