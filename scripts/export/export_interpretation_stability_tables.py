#!/usr/bin/env python
"""E8: assemble interpretation-stability runs into response-letter tables (CSV + markdown + LaTeX).

Reads the local artifacts written by ``scripts/run/run_interpretation_stability.py``
(``results/e8_interpretability_stability/{ortho,concept}/<run_id>/{stability.json,fits.csv}``) and
emits two tidy tables — one per head — that quantify the paper's two unquantified stability claims:

  * orthogonal head — per (dataset, regime): matched ``|cos|`` and subspace chordal distance across
    fits (the init-vs-data-shuffle test), plus top-K selection Jaccard, importance rank correlation,
    mean per-direction sign agreement, and the mean detection mAP for context.
  * concept model — per (dataset, beta): top-K concept Jaccard, importance rank correlation, mean
    per-concept sign agreement, mean #active concepts, and mean mAP.

Analysis-only, no GPU, no W&B. Figures are produced separately (project ``create-figures`` convention).

Usage:
    python scripts/export/export_interpretation_stability_tables.py \
        --base-dir results/e8_interpretability_stability --out-dir outputs/e8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt(mean: float, std: float | None = None) -> str:
    """Paper-styled ``mean ± std`` (or just ``mean``) to 3 decimals."""
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def _load_runs(base_dir: Path, mode: str) -> list[dict]:
    """Load (stability.json, fits.csv) for every run under ``base_dir/<mode>/``."""
    runs = []
    for run_dir in sorted((base_dir / mode).glob("*")):
        sj = run_dir / "stability.json"
        if not sj.exists():
            continue
        payload = json.loads(sj.read_text())
        fits = (
            pd.read_csv(run_dir / "fits.csv") if (run_dir / "fits.csv").exists() else pd.DataFrame()
        )
        runs.append({"dir": run_dir, "payload": payload, "fits": fits})
    return runs


def _ortho_table(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        p, fits = r["payload"], r["fits"]
        s = p["stability"]
        d = s["directions"]
        rows.append(
            {
                "dataset": p.get("dataset"),
                "regime": p.get("regime"),
                "n_fits": p.get("n_fits"),
                "matched_abs_cosine": _fmt(
                    d["matched_abs_cosine"]["mean"], d["matched_abs_cosine"]["std"]
                ),
                "subspace_chordal_dist": _fmt(
                    d["subspace_chordal_distance"]["mean"], d["subspace_chordal_distance"]["std"]
                ),
                "top_k_jaccard": _fmt(
                    s["importance_top_k_jaccard"]["mean"], s["importance_top_k_jaccard"]["std"]
                ),
                "rank_corr": _fmt(
                    s["importance_rank_correlation"]["mean"],
                    s["importance_rank_correlation"]["std"],
                ),
                "mean_sign_agree": _fmt(float(np.mean(s["sign_agreement_per_direction"]))),
                "mean_mAP": _fmt(float(fits["mAP"].mean())) if "mAP" in fits else "—",
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["dataset", "regime"]).reset_index(drop=True)


def _concept_table(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        p, fits = r["payload"], r["fits"]
        per_beta = p["stability"].get("per_beta_seed_stability", {})
        for beta_key, st in per_beta.items():
            beta = float(beta_key.replace("beta_", ""))
            sub = fits[np.isclose(fits["beta"], beta)] if "beta" in fits else fits
            sign = p["stability"].get("across_all_fits", {}).get("sign_agreement_per_concept")
            rows.append(
                {
                    "dataset": p.get("dataset"),
                    "beta": beta,
                    "top_k_jaccard": _fmt(st["top_k_jaccard"]["mean"], st["top_k_jaccard"]["std"]),
                    "rank_corr": _fmt(
                        st["rank_correlation"]["mean"], st["rank_correlation"]["std"]
                    ),
                    "mean_sign_agree": _fmt(float(np.mean(sign))) if sign else "—",
                    "mean_active": _fmt(float(sub["mean_active_concepts"].mean()))
                    if "mean_active_concepts" in sub
                    else "—",
                    "mean_mAP": _fmt(float(sub["mAP"].mean()))
                    if "mAP" in sub and len(sub)
                    else "—",
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["dataset", "beta"]).reset_index(drop=True)


def _to_markdown(df: pd.DataFrame) -> str:
    """Pipe-table markdown without the optional ``tabulate`` dependency."""
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body]) + "\n"


def _write(df: pd.DataFrame, out_dir: Path, stem: str, caption: str, label: str) -> None:
    if df.empty:
        print(f"[skip] no runs for {stem}")
        return
    (out_dir / f"{stem}.csv").write_text(df.to_csv(index=False))
    (out_dir / f"{stem}.md").write_text(_to_markdown(df))
    latex = df.to_latex(
        index=False, escape=True, caption=caption, label=label, column_format="l" * df.shape[1]
    )
    (out_dir / f"{stem}.tex").write_text(latex)
    print(f"\n=== {stem} ===")
    print(df.to_string(index=False))
    print(f"wrote {out_dir / stem}.{{csv,md,tex}}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--base-dir", type=Path, default=Path("results/e8_interpretability_stability"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/e8"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ortho = _ortho_table(_load_runs(args.base_dir, "ortho"))
    concept = _concept_table(_load_runs(args.base_dir, "concept"))

    _write(
        ortho,
        args.out_dir,
        "ortho_stability",
        caption="Stability of the orthogonal head's learned directions across fits "
        "(matched $|\\cos|$ and subspace chordal distance; higher $|\\cos|$ / lower distance = "
        "more stable). The vary-init vs vary-shuffle contrast tests whether the directions are "
        "init-driven.",
        label="tab:e8_ortho_stability",
    )
    _write(
        concept,
        args.out_dir,
        "concept_stability",
        caption="Stability of the concept model's selected concepts across seeds, by sparsity weight "
        "$\\beta$ (top-$K$ Jaccard and importance rank correlation; higher = more stable).",
        label="tab:e8_concept_stability",
    )


if __name__ == "__main__":
    main()
