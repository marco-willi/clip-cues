"""Convert the built input snapshot into its distributable form.

Reads ``reproduction/experiments/data/`` (built by ``build_data_snapshot.py``) and writes a release tree whose
array artifacts are object-free ``.npz`` instead of pandas pickles / torch ``.pt`` — portable across
Python versions, and safe to download because nothing is unpickled. See
:mod:`clip_cues_research.finalexp.release` for the rationale and the exact layout.

The release carries ``release_manifest.json``, which pairs each artifact's **built** sha256 (the
provenance anchor cited by every ``run_meta.json``) with its **released** sha256 (what a download is
checked against). ``manifest.json`` itself is copied verbatim and never rewritten.

    uv run python scripts/finalexp/export_snapshot_release.py --out dist/snapshot-release

Upload the resulting directory to the artifacts repository; ``fetch_snapshot.py`` is the other half.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from clip_cues_research.finalexp import release  # noqa: E402
from clip_cues_research.finalexp.data import sha256  # noqa: E402

SNAPSHOT = Path("reproduction/experiments/data")
SIDECARS = [
    "manifest.json",
    "MANIFEST.md",
    "EXCLUDED.md",
    # not a registered artifact, but the record that the derived projected frames agree
    # with the cached ones — tests/test_finalexp_data.py requires it.
    "reference/derived_vs_cached_crosscheck.json",
]


def convert(rec: dict, src: Path, dest: Path) -> None:
    """Write the released form of one artifact."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    kind = rec["kind"]
    if kind in release.FRAME_KINDS:
        with open(src, "rb") as f:
            release.frame_to_npz(pickle.load(f), dest)
    elif kind in release.VOCAB_KINDS:
        import torch

        obj = torch.load(src, map_location="cpu", weights_only=False)
        release.vocab_to_npz(obj, dest)
    else:
        shutil.copy2(src, dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="dist/snapshot-release", help="output directory")
    args = ap.parse_args()

    out = Path(args.out)
    manifest = json.loads((SNAPSHOT / "manifest.json").read_text())
    out.mkdir(parents=True, exist_ok=True)

    entries: dict[str, dict] = {}
    total = 0
    for rec in manifest["artifacts"]:
        src = SNAPSHOT / rec["path"]
        if not src.exists():
            raise FileNotFoundError(
                f"{rec['id']}: {src} — build the snapshot first (make finalexp-data)"
            )
        rel_path = release.release_name(rec["path"])
        dest = out / rel_path
        convert(rec, src, dest)
        digest, size = sha256(dest), dest.stat().st_size
        total += size
        entries[rec["id"]] = {
            "release_path": rel_path,
            "release_sha256": digest,
            "release_bytes": size,
            "built_path": rec["path"],
            "sha256": rec["sha256"],
            "converted": rel_path != rec["path"],
        }
        flag = "convert" if rel_path != rec["path"] else "copy   "
        print(f"  {flag} {rec['id']:40s} {size / 1e6:8.2f} MB  {rel_path}")

    for name in SIDECARS:
        p = SNAPSHOT / name
        if p.exists():
            (out / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out / name)

    (out / "release_manifest.json").write_text(
        json.dumps(
            {
                "snapshot_version": manifest["version"],
                "built_at": manifest["built_at"],
                "released_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "format": "npz-object-free",
                "note": (
                    "sha256 is the hash of the artifact as built (the provenance anchor cited by "
                    "run_meta.json); release_sha256 is the hash of the distributed file. They "
                    "differ exactly where 'converted' is true."
                ),
                "artifacts": entries,
            },
            indent=2,
        )
        + "\n"
    )
    n_conv = sum(1 for e in entries.values() if e["converted"])
    print(f"\n{len(entries)} artifacts ({n_conv} converted), {total / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
