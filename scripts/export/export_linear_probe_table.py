#!/usr/bin/env python
"""E3: collect the per-backbone linear-probe runs into a comparison table for the response letter.

Pulls the finished runs of the E3 experiment from W&B (group ``e3_clip_variants``) and writes a
tidy CSV + markdown table of mAP/AUROC per CLIP backbone — the artifact Reviewer 1 asks for.

Usage:
    python scripts/export/export_linear_probe_table.py --entity <wandb-entity> --out-dir outputs/e3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COLUMNS = ["backbone", "input_dim", "val/mAP", "val/auroc", "test/mAP", "test/auroc"]
# Order backbones largest -> smallest for the table.
BACKBONE_ORDER = {"clip_large_patch14": 0, "clip_base_patch16": 1, "clip_base_patch32": 2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", required=True, help="W&B entity (user/team)")
    p.add_argument("--project", default="clip-cues")
    p.add_argument("--group", default="e3_clip_variants")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/e3"))
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
        if s.get("backbone") is None:
            continue
        rows.append({c: s.get(c) for c in COLUMNS})

    if not rows:
        raise SystemExit(f"No finished runs found in group '{args.group}'.")

    df = pd.DataFrame(rows).drop_duplicates("backbone")
    df["_o"] = df["backbone"].map(lambda b: BACKBONE_ORDER.get(b, 99))
    df = df.sort_values("_o").drop(columns="_o").reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "backbone_comparison.csv").write_text(df.to_csv(index=False))
    (args.out_dir / "backbone_comparison.md").write_text(
        df.to_markdown(index=False, floatfmt=".4f")
    )
    print(df.to_string(index=False))
    print(f"\nWrote {args.out_dir}/backbone_comparison.csv and .md")


if __name__ == "__main__":
    main()
