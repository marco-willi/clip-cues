#!/usr/bin/env python
"""E1: assemble the cross-dataset generalization matrix from W&B.

Each forensic run (scripts/run/run_forensic_baseline.py) trains/zero-shots on one source dataset and
evaluates on several, writing ``matrix/<eval>/mAP`` (+ auroc) to its W&B summary and a
``train_label``. This script collects those runs (group ``forensics_xdataset``) into a
train(rows) × eval(cols) matrix of mAP and AUROC for the response letter.

Usage:
    python scripts/export/export_cross_dataset_matrix.py --entity <wandb-entity> --out-dir outputs/e1_cross
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# Preferred row/col ordering (others appended alphabetically).
ORDER = ["synthclic", "synthbuster-plus", "cnnspot-small", "cnnspot-progan-zeroshot"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", required=True, help="W&B entity (user/team)")
    p.add_argument("--project", default="clip-cues")
    p.add_argument("--group", default="forensics_xdataset")
    p.add_argument("--metric", default="mAP", choices=["mAP", "auroc"])
    p.add_argument("--out-dir", type=Path, default=Path("outputs/e1_cross"))
    return p.parse_args()


def _order_key(label: str):
    return (ORDER.index(label), label) if label in ORDER else (len(ORDER), label)


def main() -> None:
    args = parse_args()
    import wandb

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}", filters={"group": args.group})

    pat = re.compile(rf"^matrix/(.+)/{re.escape(args.metric)}$")
    cells: dict[str, dict[str, float]] = {}
    for run in runs:
        if run.state != "finished":
            continue
        train_label = run.summary.get("train_label") or run.config.get("train_label")
        if not train_label:
            continue
        row = cells.setdefault(train_label, {})
        for key, val in run.summary.items():
            m = pat.match(key)
            if m:
                row[m.group(1)] = float(val)

    if not cells:
        raise SystemExit(f"No matrix cells found in group '{args.group}'.")

    rows = sorted(cells, key=_order_key)
    cols = sorted({c for r in cells.values() for c in r}, key=_order_key)
    df = pd.DataFrame(
        [[cells[r].get(c, float("nan")) for c in cols] for r in rows],
        index=rows,
        columns=cols,
    )
    df.index.name = f"train \\ eval ({args.metric})"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"matrix_{args.metric}.csv"
    md_path = args.out_dir / f"matrix_{args.metric}.md"
    df.to_csv(csv_path)
    md_path.write_text(df.to_markdown(floatfmt=".4f"))
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote {csv_path}\nWrote {md_path}")


if __name__ == "__main__":
    main()
