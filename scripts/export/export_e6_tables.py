#!/usr/bin/env python
"""E6: aggregate the strong-baseline predictions into per-dataset metrics + a comparison table.

Reads the per-dataset parquet written by run_e6_strong_baseline.py
(`results/e6_strong_baseline/<model_tag>/predictions/<dataset>__<run_id>.parquet`, latest per dataset)
and writes:
    results/e6_strong_baseline/<model_tag>/tables/e6_main.csv          # per dataset: AP/AUROC/mAP-by-gen
    results/e6_strong_baseline/<model_tag>/tables/e6_by_generator.csv
    results/e6_strong_baseline/<model_tag>/tables/e6_by_architecture.csv  # CF-Eval only
    results/e6_strong_baseline/<model_tag>/summary.md

mAP-by-generator uses the paper's protocol: **balanced classes**, averaged per generator — each
generator's fakes vs an **equal number** of reals from its matched ``real_source`` (all-reals fallback
for single-source SynthCLIC/CNNSpot). Overall AP is all-fakes-vs-all-reals.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def canon_arch(a: str) -> str:
    import re

    key = re.sub(r"[^a-z]", "", str(a).lower())
    for k, v in {
        "gan": "GAN",
        "latdiff": "LatDiff",
        "pixdiff": "PixDiff",
        "commercial": "Commercial",
    }.items():
        if k in key:
            return v
    return str(a)


def _ap(y, s):
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def per_generator_ap(df: pd.DataFrame, n_seeds: int = 5, base_seed: int = 0) -> pd.DataFrame:
    """AP per fake generator under the paper's protocol (CommunityForensics + our Convention A):
    **balanced classes**, averaged per generator. Each generator's fakes are scored against an
    **equal number** of reals drawn from its **matched ``real_source``** (the corresponding real data);
    falls back to all reals for single-source datasets (SynthCLIC/CNNSpot). Balancing matters on
    CF-Eval, where reals span multiple sources — pairing a generator's ~1k fakes against *all* reals
    (1:N, cross-source) deflates AP (0.95 unbalanced vs 0.99 balanced ~ paper 0.987). Subsampling is
    averaged over ``n_seeds`` (fixed ``base_seed``) for a stable, reproducible number."""
    reals = df[df.label == 0]
    has_rs = "real_source" in df.columns and reals["real_source"].nunique() > 1
    rng = np.random.default_rng(base_seed)
    rows = []
    for gen, g in df[df.label == 1].groupby("source"):
        pool = (
            reals[reals["real_source"].isin(set(g["real_source"].dropna().unique()))]
            if has_rs
            else reals
        )
        if len(pool) == 0:
            pool = reals
        n = min(len(g), len(pool))  # balanced 1:1
        aps = []
        for _ in range(max(1, n_seeds)):
            negs = pool.sample(n=n, random_state=int(rng.integers(2**31)))
            sub = pd.concat([g, negs])
            aps.append(_ap(sub["label"].to_numpy(), sub["score"].to_numpy()))
        rows.append(
            {
                "generator": gen,
                "architecture": canon_arch(g["architecture"].iloc[0]),
                "n_fake": len(g),
                "n_real": n,
                "ap": float(np.nanmean(aps)),
            }
        )
    return pd.DataFrame(rows).sort_values("ap").reset_index(drop=True)


def latest(pred_dir: Path) -> dict[str, Path]:
    best = {}
    for p in sorted(glob.glob(str(pred_dir / "*.parquet"))):
        ds, _, rid = p.split("/")[-1].removesuffix(".parquet").rpartition("__")
        if ds and (ds not in best or rid > best[ds][0]):
            best[ds] = (rid, Path(p))
    return {d: p for d, (_, p) in best.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model-dir", type=Path, default=Path("results/e6_strong_baseline/commfor-model-384")
    )
    args = ap.parse_args()
    files = latest(args.model_dir / "predictions")
    if not files:
        raise SystemExit(
            f"No parquet in {args.model_dir}/predictions — run run_e6_strong_baseline.py first."
        )
    (args.model_dir / "tables").mkdir(parents=True, exist_ok=True)

    main_rows, gen_rows, arch_rows = [], [], []
    for ds, path in files.items():
        df = pd.read_parquet(path)
        y, s = df["label"].to_numpy(), df["score"].to_numpy()
        gens = per_generator_ap(df)
        gen_rows.append(gens.assign(dataset=ds))
        row = {
            "dataset": ds,
            "n": len(df),
            "n_fake": int(y.sum()),
            "overall_ap": _ap(y, s),
            "auroc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan,
            "mAP_by_generator": float(gens["ap"].mean()),
        }
        main_rows.append(row)
        if df["architecture"].nunique() > 1:  # CF-Eval has real architectures
            a = (
                gens.groupby("architecture")["ap"]
                .mean()
                .rename("mAP")
                .reset_index()
                .assign(dataset=ds)
            )
            arch_rows.append(a)

    main = pd.DataFrame(main_rows).sort_values("dataset")
    main.to_csv(args.model_dir / "tables" / "e6_main.csv", index=False)
    pd.concat(gen_rows).to_csv(args.model_dir / "tables" / "e6_by_generator.csv", index=False)
    if arch_rows:
        pd.concat(arch_rows).to_csv(
            args.model_dir / "tables" / "e6_by_architecture.csv", index=False
        )

    lines = [
        "# E6 — CommunityForensics out-of-the-box baseline",
        "",
        f"Model dir: `{args.model_dir}`",
        "",
        "## Per-dataset (overall AP / AUROC / mAP-by-generator)",
        "",
        "| dataset | n | overall AP | AUROC | mAP-by-gen |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in main.itertuples():
        lines.append(
            f"| {r.dataset} | {r.n} | {r.overall_ap:.3f} | {r.auroc:.3f} | {r.mAP_by_generator:.3f} |"
        )
    lines += [
        "",
        "_mAP-by-generator: balanced classes, averaged per generator — each generator's fakes vs an "
        "equal number of reals from its matched `real_source` (all-reals fallback for single-source). "
        "CommunityForensics-Eval is **in-distribution** for this detector — caption it as such; "
        "the out-of-distribution comparison is SynthCLIC / SynthBuster+ / CNNSpot._",
    ]
    (args.model_dir / "summary.md").write_text("\n".join(lines))
    print(f"Wrote tables + summary.md to {args.model_dir}/")
    print(main.to_string(index=False))


if __name__ == "__main__":
    main()
