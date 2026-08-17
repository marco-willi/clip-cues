#!/usr/bin/env python
"""Verify the ``reproduction/experiments/data/`` snapshot. Exit 0 iff everything checks out.

Run before every F-experiment (``make finalexp-all`` does this and aborts on failure) and again on
a Lambda box after syncing, so the manifest certifies the remote copy too.

Checks, in order of severity:

  FAIL  a manifest artifact is missing on disk
  FAIL  a file's sha256 differs from the manifest (modified, or the wrong file is in place)
  FAIL  a declared shape / dtype / split-count / vocabulary size does not match the file
  FAIL  a structural fingerprint does not match (catches a swapped-then-renamed artifact)
  WARN  the SOURCE file in data/ has changed since the copy (the snapshot is deliberately frozen,
        so drift may be intentional — reported with the differing ids, never a hard failure)

    uv run python scripts/finalexp/verify_data_snapshot.py
"""

from __future__ import annotations

import json
from pathlib import Path

from clip_cues_research.finalexp.data import (
    MANIFEST,
    RELEASE_MANIFEST,
    SNAPSHOT,
    sha256,
)
from clip_cues_research.finalexp.snapshot import describe


def main() -> int:
    if not MANIFEST.exists():
        print(f"FAIL: no manifest at {MANIFEST}. Run scripts/finalexp/build_data_snapshot.py")
        return 1

    doc = json.loads(MANIFEST.read_text())
    failures: list[str] = []
    warnings: list[str] = []

    print(
        f"Verifying snapshot v{doc['version']} ({len(doc['artifacts'])} artifacts) in {SNAPSHOT}/"
    )

    # A snapshot obtained with `make finalexp-fetch` holds the released, object-free .npz form
    # instead of the built pickles. Those files cannot have the built artifact's hash, so they are
    # checked against release_sha256 — their own hash — from the release manifest.
    released = (
        json.loads(RELEASE_MANIFEST.read_text())["artifacts"] if RELEASE_MANIFEST.exists() else {}
    )

    for a in doc["artifacts"]:
        aid, path = a["id"], SNAPSHOT / a["path"]
        expected, form = a["sha256"], "built"

        if not path.exists() and aid in released:
            rel = released[aid]
            path, expected, form = SNAPSHOT / rel["release_path"], rel["release_sha256"], "released"

        if not path.exists():
            failures.append(
                f"{aid}: missing file {path}"
                + ("" if released else "  (fetch it: make finalexp-fetch)")
            )
            continue

        actual = sha256(path)
        if actual != expected:
            failures.append(
                f"{aid}: sha256 mismatch ({form})\n      expected {expected}\n      on disk  {actual}"
            )
            continue

        # Structural re-description: shape/dtype/splits/vocab/fingerprint must still agree.
        try:
            desc = describe(path, a["kind"])
        except Exception as exc:
            failures.append(f"{aid}: could not re-describe ({type(exc).__name__}: {exc})")
            continue
        for key in ("shape", "dtype", "n_rows", "n_vocab", "split_counts", "fingerprint"):
            if key in a and key in desc and a[key] != desc[key]:
                failures.append(
                    f"{aid}: {key} mismatch — manifest {a[key]!r} vs file {desc[key]!r}"
                )

        # Source drift is informational: the snapshot is frozen on purpose. Derived artifacts have
        # no external source — their inputs are pinned by `derived_from` instead, and those inputs
        # are themselves manifest entries verified in this same loop.
        # Skipped for a released artifact: the build-time source does not exist on a machine that
        # fetched the snapshot, and `make finalexp-fetch` rewrites the legacy mirror paths itself,
        # so comparing against them would warn on every single verify.
        if a.get("source_path") and form == "built":
            src = Path(a["source_path"])
            if not src.exists():
                warnings.append(f"{aid}: source {src} no longer exists (snapshot still valid)")
            elif sha256(src) != a["source_sha256"]:
                warnings.append(f"{aid}: source {src} has changed since the snapshot was taken")
        for dep_id, dep_sha in (a.get("derived_from") or {}).items():
            dep = next((r for r in doc["artifacts"] if r["id"] == dep_id), None)
            if dep is None:
                failures.append(f"{aid}: derived_from references unknown id {dep_id!r}")
            elif dep["sha256"] != dep_sha:
                failures.append(
                    f"{aid}: was derived from {dep_id} @ {dep_sha[:16]}, but that artifact is now "
                    f"{dep['sha256'][:16]} — regenerate with scripts/finalexp/prepare_features.py"
                )

        print(f"  ok   {aid:38s} {a['sha256'][:16]}  {a['bytes'] / 1e6:8.2f} MB")

    if warnings:
        print(f"\n{len(warnings)} warning(s) — source drift, not a failure:")
        for w in warnings:
            print(f"  WARN {w}")

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  FAIL {f}")
        return 1

    total = sum(a["bytes"] for a in doc["artifacts"])
    print(f"\nOK — {len(doc['artifacts'])} artifacts verified, {total / 1e6:.1f} MB total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
