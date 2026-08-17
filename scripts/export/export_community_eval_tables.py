#!/usr/bin/env python
"""E7 Option A: aggregate CommunityForensics-Eval predictions into response-letter tables.

Reads the per-detector full-metadata parquet files written by ``run_community_eval.py``
(``results/e7_community_eval/predictions/<detector>__<run_id>.parquet``; latest run_id per
detector) and emits:

    results/e7_community_eval/tables/community_eval_overall.csv
    results/e7_community_eval/tables/community_eval_by_generator.csv
    results/e7_community_eval/tables/community_eval_by_architecture.csv
    results/e7_community_eval/tables/community_eval_by_real_source.csv
    results/e7_community_eval/tables/community_eval_main.csv     # compact main table
    results/e7_community_eval/summary.md

Metric conventions (CommunityForensics-style):
    * AP within a (paired real+fake) group; **mAP-by-generator** = mean of per-``model_name`` APs.
    * mAP-by-architecture = mean of per-generator APs within each architecture.
    * AP-by-real_source = AP within each ``real_source`` group.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

EXPERIMENT = "e7_community_eval"

# Detector → (training data, cue type) for the main table.
DETECTOR_META = {
    "clip_orthogonal_synthclic": ("SynthCLIC", "CLIP / photographic-semantic"),
    "linear_probe_synthclic": ("SynthCLIC", "CLIP / linear probe"),
    "linear_probe_synthbuster": ("SynthBuster+", "CLIP / linear probe"),
    "linear_probe_cnnspot": ("CNNSpot", "CLIP / linear probe"),
    "cnnspot_synthclic": ("SynthCLIC", "low-level forensic"),
    "cnnspot_progan": ("ProGAN (CNNSpot)", "low-level forensic"),
}

# Normalise CommunityForensics `architecture` strings to canonical buckets (extend as values appear).
ARCH_CANON = {"gan": "GAN", "latdiff": "LatDiff", "pixdiff": "PixDiff", "commercial": "Commercial"}
MAIN_ARCH_COLS = ["GAN", "LatDiff", "PixDiff", "Commercial"]


def canon_arch(a: str) -> str:
    key = re.sub(r"[^a-z]", "", str(a).lower())
    for k, v in ARCH_CANON.items():
        if k in key:
            return v
    return str(a)


def _ap(y: np.ndarray, s: np.ndarray) -> float:
    """AP for a (paired) group; NaN if single-class."""
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def latest_parquets(pred_dir: Path) -> dict[str, Path]:
    """Map detector -> its latest-run_id parquet (filename ``<detector>__<run_id>.parquet``)."""
    best: dict[str, tuple[str, Path]] = {}
    for p in sorted(pred_dir.glob("*.parquet")):
        det, _, rid = p.stem.rpartition("__")
        if det and (det not in best or rid > best[det][0]):
            best[det] = (rid, p)
    return {d: p for d, (_, p) in best.items()}


def per_generator(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for src, g in df.groupby("source"):
        rows.append(
            {
                "generator": src,
                "architecture": canon_arch(g["architecture"].iloc[0]),
                "n": len(g),
                "ap": _ap(g["label"].to_numpy(), g["score"].to_numpy()),
                "acc": accuracy_score(g["label"], (g["score"] >= 0.5).astype(int)),
            }
        )
    return pd.DataFrame(rows).sort_values(["architecture", "generator"])


def detector_overall(df: pd.DataFrame) -> dict:
    y, s = df["label"].to_numpy(), df["score"].to_numpy()
    pred = (s >= 0.5).astype(int)
    gen = per_generator(df)
    return {
        "n": len(df),
        "overall_ap": _ap(y, s),
        "auroc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mAP_by_generator": float(gen["ap"].mean()),
        "mAcc_by_generator": float(gen["acc"].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results-dir", type=Path, default=Path("results") / EXPERIMENT)
    args = ap.parse_args()

    pred_dir = args.results_dir / "predictions"
    tables_dir = args.results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    files = latest_parquets(pred_dir)
    if not files:
        raise SystemExit(
            f"No prediction parquets in {pred_dir}/ — run scripts/run/run_community_eval.py first."
        )
    print(f"Aggregating {len(files)} detector(s): {sorted(files)}")

    overall_rows, gen_rows, arch_rows, src_rows, main_rows = [], [], [], [], []
    for det, path in files.items():
        df = pd.read_parquet(path)
        ov = detector_overall(df)
        overall_rows.append({"detector": det, **ov})

        gen = per_generator(df).assign(detector=det)
        gen_rows.append(gen)

        arch = (
            gen.groupby("architecture")["ap"]
            .mean()
            .rename("mAP")
            .reset_index()
            .assign(detector=det)
        )
        arch_rows.append(arch)

        for rs, g in df.groupby("real_source"):
            src_rows.append(
                {
                    "detector": det,
                    "real_source": rs,
                    "n": len(g),
                    "ap": _ap(g["label"].to_numpy(), g["score"].to_numpy()),
                }
            )

        train, cue = DETECTOR_META.get(det, ("?", "?"))
        arch_map = dict(zip(arch["architecture"], arch["mAP"]))
        main_rows.append(
            {
                "detector": det,
                "training_data": train,
                "cue_type": cue,
                "overall_ap": ov["overall_ap"],
                "mAP_by_gen": ov["mAP_by_generator"],
                "mAcc_by_gen": ov["mAcc_by_generator"],
                **{f"{a}_ap": arch_map.get(a, np.nan) for a in MAIN_ARCH_COLS},
            }
        )

    overall = pd.DataFrame(overall_rows).sort_values("mAP_by_generator", ascending=False)
    overall.to_csv(tables_dir / "community_eval_overall.csv", index=False)
    pd.concat(gen_rows).to_csv(tables_dir / "community_eval_by_generator.csv", index=False)
    pd.concat(arch_rows).to_csv(tables_dir / "community_eval_by_architecture.csv", index=False)
    pd.DataFrame(src_rows).to_csv(tables_dir / "community_eval_by_real_source.csv", index=False)
    main = pd.DataFrame(main_rows).sort_values("mAP_by_gen", ascending=False)
    main.to_csv(tables_dir / "community_eval_main.csv", index=False)

    _write_summary(args.results_dir / "summary.md", overall, main)
    print(f"Wrote 5 tables + summary.md to {args.results_dir}/")


def _df_md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table (no tabulate dependency)."""
    cols = list(df.columns)
    head = "| " + " | ".join(map(str, cols)) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _write_summary(path: Path, overall: pd.DataFrame, main: pd.DataFrame) -> None:
    best = overall.iloc[0]
    worst = overall.iloc[-1]
    clip = main[main["cue_type"].str.startswith("CLIP")]
    forensic = main[main["cue_type"].str.contains("forensic")]
    clip_map = float(clip["mAP_by_gen"].max()) if len(clip) else float("nan")
    forensic_map = float(forensic["mAP_by_gen"].max()) if len(forensic) else float("nan")
    if clip_map >= forensic_map:
        verdict = (
            f"CLIP transfers better (best CLIP mAP-by-gen {clip_map:.3f} ≥ best forensic "
            f"{forensic_map:.3f}) — consistent with 'forensic CNNs are dataset-locked; CLIP is "
            f"more robust but still imperfect.'"
        )
    else:
        verdict = (
            f"Forensic transfers better here (best forensic {forensic_map:.3f} > best CLIP "
            f"{clip_map:.3f}) — reframe as complementary forensic+CLIP evidence."
        )
    lines = [
        "# E7 — CommunityForensics-Eval (Option A) summary",
        "",
        f"- **Best transfer:** `{best['detector']}` (mAP-by-gen {best['mAP_by_generator']:.3f}, "
        f"overall AP {best['overall_ap']:.3f}).",
        f"- **Worst transfer:** `{worst['detector']}` (mAP-by-gen {worst['mAP_by_generator']:.3f}).",
        f"- **CLIP vs forensic:** {verdict}",
        "- **SynthCLIC conclusion:** "
        + (
            "supported — the SynthCLIC-trained CLIP detector retains the best generator-averaged "
            "transfer to a larger, generator-diverse benchmark."
            if best["detector"].startswith("clip")
            else "qualified — a forensic detector leads on this external benchmark; revisit the "
            "semantic-vs-forensic framing."
        ),
        "",
        "## Overall (sorted by mAP-by-generator)",
        "",
        _df_md(overall.round(4)),
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
