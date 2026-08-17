"""Fetch the released input snapshot and verify every artifact.

This is the entry point for reproducing the paper's numbers without a GPU: it downloads the frozen,
checksummed inputs that F1–F7 and the figures read, checks each file against
``release_manifest.json``, and puts them where the code expects them.

    make finalexp-fetch                      # from the artifacts repository on HuggingFace
    make finalexp-fetch FROM_DIR=<path>      # from a local release directory

Two layouts are materialised, on purpose:

``reproduction/experiments/data/``
    the snapshot proper, reached through ``clip_cues_research.finalexp.data.get_*``, which verifies
    the hash and the declared embedding space on every load.
``data/embeddings/``
    legacy paths that the figure and analysis modules read directly, predating the manifest
    loaders. These are rewritten locally from the verified release, so no downloaded file is ever
    unpickled — see ``--no-mirror`` to skip them.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from clip_cues_research.finalexp import release  # noqa: E402
from clip_cues_research.finalexp.data import sha256  # noqa: E402

SNAPSHOT = Path("reproduction/experiments/data")
DEFAULT_REPO = "marco-willi/clip-cues-artifacts"

#: Snapshot id -> legacy path read directly by figure/analysis code. Written from the verified
#: release, never downloaded in this form.
MIRROR = {
    "pooler/synthclic": "data/embeddings/synthclic_clip_large_patch14.pkl",
    "pooler/synthbuster-plus": "data/embeddings/synthbuster-plus_clip_large_patch14.pkl",
    "pooler/cnnspot": "data/embeddings/cnnspot_clip_large_patch14.pkl",
    # These legacy paths hold the INDEPENDENTLY CACHED projections, not the derived ones. The two
    # are close but not equal (median cosine 0.9955, AUROC delta 0.0074), and the snapshot registers
    # them as separate artifacts precisely so the difference stays measurable — mirroring
    # `projected/*` here would overwrite the cached side with a copy of the derived side and make
    # the derived-vs-cached crosscheck vacuously perfect.
    "reference/projected_cached_synthclic": "data/embeddings/synthclic_projected_embeddings.pkl",
    "reference/projected_cached_cnnspot": "data/embeddings/cnnspot_projected_embeddings.pkl",
    "projection/wp_l14_336": "data/embeddings/clip_l14_336_visual_projection.npy",
    "vocab/antonyms": "data/embeddings/vocab_canon/antonyms.pt",
}


def stage_from_dir(src: Path, dest: Path) -> None:
    """Copy a local release tree into place."""
    for p in sorted(src.rglob("*")):
        if p.is_file():
            out = dest / p.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)


def stage_from_hub(repo: str, dest: Path, revision: str | None) -> None:
    """Download the release from a HuggingFace dataset repository."""
    from huggingface_hub import snapshot_download

    got = snapshot_download(repo_id=repo, repo_type="dataset", revision=revision)
    stage_from_dir(Path(got), dest)


def verify(dest: Path) -> list[str]:
    """Check every released artifact against its own hash. Returns the ids that failed."""
    rel = json.loads((dest / "release_manifest.json").read_text())
    bad = []
    for aid, rec in rel["artifacts"].items():
        p = dest / rec["release_path"]
        if not p.exists():
            print(f"  MISSING {aid}")
            bad.append(aid)
            continue
        if sha256(p) != rec["release_sha256"]:
            print(f"  CORRUPT {aid} ({p})")
            bad.append(aid)
            continue
        print(f"  ok   {aid:40s} {rec['release_bytes'] / 1e6:8.2f} MB")
    return bad


def mirror(dest: Path) -> list[str]:
    """Write the legacy ``data/embeddings/`` paths from the verified release."""
    rel = json.loads((dest / "release_manifest.json").read_text())["artifacts"]
    written: list[str] = []
    for aid, out_path in MIRROR.items():
        rec = rel.get(aid)
        if rec is None:
            continue
        src, out = dest / rec["release_path"], Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not rec["converted"]:
            shutil.copy2(src, out)
        elif out.suffix == ".pkl":
            with open(out, "wb") as f:
                pickle.dump(release.npz_to_frame(src), f, protocol=5)
        elif out.suffix == ".pt":
            import torch

            torch.save(release.npz_to_vocab(src), out)
        print(f"  mirror {aid:40s} -> {out}")
        written.append(str(out))
    return written


MIRROR_MARKER = Path("data/embeddings/.fetched-mirror")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=DEFAULT_REPO, help="HuggingFace dataset repo id")
    ap.add_argument("--revision", default=None, help="repo revision (branch, tag or commit)")
    ap.add_argument("--from-dir", default=None, help="stage from a local release directory instead")
    ap.add_argument(
        "--no-mirror", action="store_true", help="skip the legacy data/embeddings paths"
    )
    args = ap.parse_args()

    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    if args.from_dir:
        print(f"Staging from {args.from_dir}")
        stage_from_dir(Path(args.from_dir), SNAPSHOT)
    else:
        print(f"Downloading {args.repo} (revision={args.revision or 'main'})")
        stage_from_hub(args.repo, SNAPSHOT, args.revision)

    print("\nVerifying:")
    bad = verify(SNAPSHOT)
    if bad:
        raise SystemExit(f"\n{len(bad)} artifact(s) failed verification: {bad}")

    if not args.no_mirror:
        print("\nMirroring legacy paths:")
        written = mirror(SNAPSHOT)
        # Mark them. These files are rewritten locally, so they are byte-different from the
        # originals the snapshot was built from; build_data_snapshot.py must refuse to treat them
        # as build sources, or every downstream hash silently changes.
        MIRROR_MARKER.parent.mkdir(parents=True, exist_ok=True)
        MIRROR_MARKER.write_text("\n".join(written) + "\n")

    print("\nSnapshot ready. Next:  make finalexp-all")


if __name__ == "__main__":
    main()
