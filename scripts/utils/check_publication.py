"""Publication gates — what must be true before this repository is pushed public.

These are not style checks. Each one corresponds to a way this project could ship something wrong or
something it has no right to distribute:

* a retracted artifact that is indistinguishable from the correct one by inspection;
* a checkpoint that is not the one the paper describes;
* third-party weights we may not redistribute;
* a figure without the caveat its caption is required to carry;
* pinned requirements that drifted from the lockfile;
* a blob large enough to make the repository painful to clone.

    uv run python scripts/utils/check_publication.py          # report
    uv run python scripts/utils/check_publication.py --strict # non-zero exit on any failure

Run it in CI on every push, and by hand before tagging a release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

MAX_BLOB_MB = 50

#: Filenames that must never appear anywhere in the repository. The first four are in the retracted
#: double-projected ("W-squared") text space and are shape-identical to the canonical cue basis; the
#: last two are third-party detector weights we do not have redistribution rights to.
FORBIDDEN_NAMES = [
    "antonyms_embeddings.pt",
    "antonyms_diff_embeddings.pt",
    "textspan_embeddings.pt",
    "vocab_pool",
    "blur_jpg_prob0.1.pth",
    "blur_jpg_prob0.5.pth",
]

#: The published checkpoints, by sha256. `linear_probe_combined` is listed explicitly because a
#: matched, no-augmentation head of the same name exists and must never replace it.
CHECKPOINT_SHAS = {
    "data/checkpoints/linear_probe_synthclic.ckpt": "c7a310eb0b14290d",
    "data/checkpoints/linear_probe_cnnspot.ckpt": "b8201afaaceda70b",
    "data/checkpoints/linear_probe_combined.ckpt": "8904774bf2bc8266",
    "data/checkpoints/clip_orthogonal_synthclic.ckpt": "0ddba778343c2018",
}

results: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    results.append((ok, name, detail))


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        capture_output=True, text=True, check=True,
    ).stdout
    # `git ls-files -c` still lists files deleted from the working tree; only existing ones
    # can be inspected.
    return [Path(f) for f in out.split("\0") if f and Path(f).exists()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_no_forbidden_artifacts(files: list[Path]) -> None:
    hits = [
        str(p) for p in files
        if any(bad in p.name or bad in str(p) for bad in FORBIDDEN_NAMES)
    ]
    check(not hits, "no retracted or third-party artifacts", "\n".join(hits))


def check_no_pre_fix_text_embeddings(files: list[Path]) -> None:
    """No `.pt` cue basis outside the snapshot, where every file is manifest-verified."""
    stray = [
        str(p) for p in files
        if p.suffix == ".pt"
        and "reproduction/experiments/data" not in str(p)
    ]
    check(not stray, "no unverified .pt cue bases", "\n".join(stray))


def check_checkpoints(files: list[Path]) -> None:
    bad = []
    for path, want in CHECKPOINT_SHAS.items():
        p = Path(path)
        if not p.exists():
            bad.append(f"{path}: missing")
        elif not sha256(p).startswith(want):
            bad.append(f"{path}: sha256 {sha256(p)[:16]} != published {want}")
    check(not bad, "published checkpoints unchanged", "\n".join(bad))


def check_blob_sizes(files: list[Path]) -> None:
    big = [
        f"{p} ({p.stat().st_size / 1e6:.1f} MB)"
        for p in files
        if p.exists() and p.stat().st_size > MAX_BLOB_MB * 1e6
    ]
    check(not big, f"no file over {MAX_BLOB_MB} MB", "\n".join(big))


def check_figure_captions() -> None:
    """Every shipped figure directory carries the generated caption its claim depends on."""
    root = Path("reproduction/experiments/figures")
    if not root.exists():
        return check(False, "figure captions present", f"{root} missing")
    missing = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "tables"):
        csvs = list(d.glob("*.csv"))
        caps = list(d.glob("*-caption.txt"))
        # fig2 is content-frozen and writes an image manifest instead of a plotted table
        if csvs and not caps and not d.name.startswith("fig2"):
            missing.append(str(d))
    check(not missing, "figure captions present", "\n".join(missing))


def check_no_secrets(files: list[Path]) -> None:
    pat = re.compile(r"(hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|/home/ubuntu)")
    hits = []
    for p in files:
        if p.suffix not in {".py", ".sh", ".md", ".yaml", ".yml", ".toml", ".json", ".cfg"}:
            continue
        if p.name == Path(__file__).name:
            continue  # this file spells out the patterns it looks for
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for m in pat.finditer(text):
            hits.append(f"{p}: {m.group(0)[:12]}…")
    check(not hits, "no credentials or host paths", "\n".join(hits))


def check_cache_root(files: list[Path]) -> None:
    """One HuggingFace cache root. A second one re-downloads ~150 GB."""
    pat = re.compile(r"""cache_dir\s*=\s*["'](?!data/hf_cache)(\./)?hf_cache""")
    hits = [
        f"{p}" for p in files
        if p.suffix in {".py", ".ipynb"} and pat.search(p.read_text(errors="ignore"))
    ]
    check(not hits, "single hf_cache root", "\n".join(hits))


def check_requirements_fresh() -> None:
    """The pinned requirements must not have drifted from uv.lock."""
    if not Path("uv.lock").exists():
        return check(False, "requirements match uv.lock", "no uv.lock")
    proc = subprocess.run(
        ["uv", "export", "--no-hashes", "--no-emit-project", "--extra", "all",
         "--no-default-groups", "--quiet"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return check(False, "requirements match uv.lock", proc.stderr.strip()[:200])
    fresh = {ln.strip() for ln in proc.stdout.splitlines() if "==" in ln}
    have = {ln.strip() for ln in Path("requirements.txt").read_text().splitlines() if "==" in ln}
    drift = sorted((fresh - have) | (have - fresh))
    check(not drift, "requirements match uv.lock", "\n".join(drift[:8]))


def check_wheel_scope() -> None:
    """The published wheel is the inference package only."""
    txt = Path("pyproject.toml").read_text()
    m = re.search(r"\[tool\.hatch\.build\.targets\.wheel\]\s*\npackages = \[([^\]]*)\]", txt)
    packages = m.group(1) if m else ""
    ok = "clip_cues_research" not in packages and "src/clip_cues" in packages
    check(ok, "wheel ships clip_cues only", packages.strip())


def check_snapshot_manifest() -> None:
    """Every manifest entry resolves, in either the built or the released format."""
    man = Path("reproduction/experiments/data/manifest.json")
    if not man.exists():
        return check(False, "snapshot manifest resolvable", "manifest.json missing")
    doc = json.loads(man.read_text())
    rel_path = man.parent / "release_manifest.json"
    released = json.loads(rel_path.read_text())["artifacts"] if rel_path.exists() else {}
    missing = [
        a["id"] for a in doc["artifacts"]
        if not (man.parent / a["path"]).exists()
        and not (a["id"] in released and (man.parent / released[a["id"]]["release_path"]).exists())
    ]
    check(not missing, "snapshot manifest resolvable", ", ".join(missing[:6]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true", help="exit non-zero if any gate fails")
    args = ap.parse_args()

    files = tracked_files()
    print(f"Checking {len(files)} tracked/untracked files\n")

    check_no_forbidden_artifacts(files)
    check_no_pre_fix_text_embeddings(files)
    check_checkpoints(files)
    check_blob_sizes(files)
    check_figure_captions()
    check_no_secrets(files)
    check_cache_root(files)
    check_requirements_fresh()
    check_wheel_scope()
    check_snapshot_manifest()

    failed = 0
    for ok, name, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            failed += 1
            for line in (detail or "").splitlines():
                print(f"          {line}")
    print(f"\n{len(results) - failed}/{len(results)} gates passed")
    return 1 if (failed and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
