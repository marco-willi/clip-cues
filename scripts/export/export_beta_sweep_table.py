#!/usr/bin/env python
"""E2: collect the beta-sensitivity sweep runs into a table for the response letter.

Pulls the finished runs of the E2 sweep from W&B (group ``e2_beta_sweep``), sorts by beta, and
writes a tidy CSV + markdown table of mAP and mean #activated concepts vs beta — the artifact
Reviewer 3 asks for.

Usage:
    python scripts/export/export_beta_sweep_table.py \
        --entity <wandb-entity> --project clip-cues --out-dir outputs/e2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COLUMNS = [
    "beta",
    "alpha",
    "val/mAP",
    "val/mean_active_concepts",
    "val/mean_gate_mass",
    "test/mAP",
    "test/mean_active_concepts",
    "num_concepts",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", required=True, help="W&B entity (user/team)")
    p.add_argument("--project", default="clip-cues")
    p.add_argument("--group", default="e2_beta_sweep")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/e2"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import wandb

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}", filters={"group": args.group})

    rows = []
    for run in runs:
        if run.state != "finished":
            continue
        s = run.summary
        rows.append({c: s.get(c) for c in COLUMNS})

    if not rows:
        raise SystemExit(f"No finished runs found in group '{args.group}'.")

    df = pd.DataFrame(rows).sort_values("beta").reset_index(drop=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.out_dir / "beta_sensitivity.csv"
    md_path = args.out_dir / "beta_sensitivity.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False, floatfmt=".4f"))

    print(df.to_string(index=False))
    print(f"\nWrote {csv_path}\nWrote {md_path}")


if __name__ == "__main__":
    main()
