#!/usr/bin/env python
"""Rebuild reproduction/revision_export/ tables from LOCAL results under Convention A.

Unlike scripts/export/package_revision_export.py this does NOT pull from W&B and does NOT rmtree the
export (so experimental_details/ survive). It regenerates the tables whose underlying experiments
were re-run with the corrected config (E2→table_d, E3→table_c + e3 cross matrix, E5→table_e) and
rebuilds the combined Table A/B + KEY_NUMBERS from local results. Run after
scripts/utils/rerun_inconsistent.sh completes.

    uv run python scripts/export/rebuild_export_local.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pandas as pd

from clip_cues_research.figures.latex import paper_table

sys.path.insert(0, "scripts/export")
import package_revision_export as P  # noqa: E402

EXPORT = Path("revision_export")
T, S, X = EXPORT / "tables", EXPORT / "tables" / "source", EXPORT / "tex"


def _latest(files, key):
    """Pick the latest run (by run_id = parent dir name) per key(metrics_dict)."""
    best: dict = {}
    for f in files:
        m = json.load(open(f))
        k = key(m)
        if k is None:
            continue
        rid = Path(f).parent.name
        if k not in best or rid > best[k][0]:
            best[k] = (rid, m)
    return [best[k][1] for k in sorted(best, key=lambda x: (str(type(x)), x))]


def table_c() -> pd.DataFrame:
    """In-domain CLIP backbone comparison (train==eval), Convention A, from results/e3_xdataset."""
    files = glob.glob("results/e3_xdataset/*/*/metrics.json")
    rows = _latest(
        files,
        key=lambda m: (
            (m["backbone"], m["train_dataset"])
            if m.get("train_dataset") == m.get("eval_dataset")
            else None
        ),
    )
    df = pd.DataFrame(
        [
            {
                "backbone": m["backbone"],
                "dataset": m["train_dataset"],
                "test_mAP": round(m["mAP"], 4),
                "test_pooled_ap": round(m.get("pooled_ap", float("nan")), 4),
                "test_auroc": round(m["auroc"], 4),
            }
            for m in rows
        ]
    )
    order = {"clip_base_patch16": 0, "clip_base_patch32": 1, "clip_large_patch14": 2}
    return df.sort_values(
        ["dataset", "backbone"], key=lambda s: s.map(order).fillna(s)
    ).reset_index(drop=True)


def table_d() -> pd.DataFrame:
    """E2 beta sensitivity (Convention A) from results/e2_beta_sweep."""
    files = glob.glob("results/e2_beta_sweep/*/*/metrics.json")
    rows = _latest(files, key=lambda m: float(m["beta"]))
    df = pd.DataFrame(
        [
            {
                "beta": m["beta"],
                "alpha": m["alpha"],
                "val/mAP": m["val/mAP"],
                "val/mean_active_concepts": m["val/mean_active_concepts"],
                "val/mean_gate_mass": m["val/mean_gate_mass"],
                "test/mAP": m["test/mAP"],
                "test/pooled_ap": m.get("test/pooled_ap"),
                "test/mean_active_concepts": m["test/mean_active_concepts"],
                "num_concepts": m["num_concepts"],
            }
            for m in rows
        ]
    )
    return df.sort_values("beta").reset_index(drop=True)


def table_d_multiseed() -> pd.DataFrame:
    """E2 beta sensitivity aggregated over seeds (mean +/- std). Uses only runs that recorded a
    ``seed`` (the multi-seed batch); returns empty if none. Latest run per (beta, seed)."""
    import numpy as np

    by: dict = {}
    for f in glob.glob("results/e2_beta_sweep/*/*/metrics.json"):
        m = json.load(open(f))
        if "seed" not in m:
            continue
        beta, seed, rid = float(m["beta"]), m["seed"], Path(f).parent.name
        by.setdefault(beta, {})
        if seed not in by[beta] or rid > by[beta][seed][0]:
            by[beta][seed] = (rid, m)
    rows = []
    for beta in sorted(by):
        runs = [m for _, m in by[beta].values()]

        def ms(key, runs=runs):
            v = np.array([r[key] for r in runs], float)
            return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0

        mAP_m, mAP_s = ms("test/mAP")
        ac_m, ac_s = ms("test/mean_active_concepts")
        gm_m, gm_s = ms("test/mean_gate_mass")
        wc_m, wc_s = ms("test/max_w_classifier")
        rows.append(
            {
                "beta": beta,
                "n_seeds": len(runs),
                "mAP_mean": mAP_m,
                "mAP_std": mAP_s,
                "active_mean": ac_m,
                "active_std": ac_s,
                "gate_mass_mean": gm_m,
                "gate_mass_std": gm_s,
                "maxWclf_mean": wc_m,
                "maxWclf_std": wc_s,
            }
        )
    return pd.DataFrame(rows)


def e1_matrix_local() -> None:
    """Rebuild outputs/e1_cross/matrix_{mAP,auroc}.csv from results/e1_forensic (latest run per cell).

    The per-cell metrics.json has null train/eval keys; the (train,eval) pair lives in the parent
    dir name ``<train>__to__<eval>``. Feeds P.table_a / P.e1_forensic_matrix."""
    cells: dict[tuple[str, str], tuple[str, dict]] = {}
    for f in glob.glob("results/e1_forensic/*/*/metrics.json"):
        p = Path(f)
        cell = p.parent.parent.name
        if "__to__" not in cell:
            continue
        tr, ev = cell.split("__to__")
        rid = p.parent.name
        if (tr, ev) not in cells or rid > cells[(tr, ev)][0]:
            cells[(tr, ev)] = (rid, json.load(open(f)))
    trains = sorted({k[0] for k in cells})
    evals = sorted({k[1] for k in cells})
    out = Path("outputs/e1_cross")
    out.mkdir(parents=True, exist_ok=True)
    for metric in ("mAP", "auroc"):
        df = pd.DataFrame(index=trains, columns=evals, dtype=float)
        for (tr, ev), (_, m) in cells.items():
            df.loc[tr, ev] = m.get(metric)
        df.to_csv(out / f"matrix_{metric}.csv")


def e5_ablation_local() -> pd.DataFrame:
    """E5 ablation from the FAITHFUL end-to-end runs (train_clip_head format), latest per head.

    The 2026-06-24 box E5 wrote ``results/e5_orthogonality/<head>__clip_large_patch14__synthclic__to__
    synthclic/<run_id>/metrics.json`` with keys head/mAP/pooled_ap/auroc/weight_ortho_score/
    activation_ortho_score — a different layout than the old cached ablation P.e5_ablation reads."""
    latest: dict[str, tuple[str, dict]] = {}
    pat = "results/e5_orthogonality/*__clip_large_patch14__synthclic__to__synthclic/*/metrics.json"
    for f in glob.glob(pat):
        p = Path(f)
        head = p.parent.parent.name.split("__")[0]
        rid = p.parent.name
        if head not in latest or rid > latest[head][0]:
            latest[head] = (rid, json.load(open(f)))
    order = {"none": 0, "activation_ortho": 1, "weight_ortho": 2}
    rows = [
        {
            "variant": head,
            "test_mAP": round(m["mAP"], 4),
            "test_pooled_ap": round(m.get("pooled_ap", float("nan")), 4),
            "test_auroc": round(m["auroc"], 4),
            "weight_ortho_score": round(m["weight_ortho_score"], 4),
            "activation_ortho_score": round(m["activation_ortho_score"], 4),
        }
        for head, (_, m) in latest.items()
    ]
    return (
        pd.DataFrame(rows).sort_values("variant", key=lambda s: s.map(order)).reset_index(drop=True)
    )


def _fmt_beta(b) -> str:
    """LaTeX scientific-notation label for a beta value, e.g. 1e-4 -> $10^{-4}$, 3e-4 -> $3\\times10^{-4}$."""
    import math

    b = float(b)
    exp = math.floor(math.log10(b))
    mant = b / 10**exp
    return f"$10^{{{exp}}}$" if abs(mant - 1.0) < 1e-9 else f"${mant:g}\\times10^{{{exp}}}$"


def _paper_tex(
    df: pd.DataFrame, caption: str, label: str, position: str = "!ht", index: bool = False
) -> str:
    """Render a DataFrame as a paper-styled LaTeX table: booktabs, \\scriptsize, \\centering, 2-decimal
    floats, caption + label — matching the style of docs/initial_submission.tex (e.g. tab:clip:test_results).
    Per-generator mAP only is assumed (callers drop AUROC / pooled-AP columns before calling).

    Thin wrapper over the shared emitter so this bundle and the F-experiment tables in
    ``reproduction/experiments/figures/tables/`` cannot drift apart; the 2-decimal default is this bundle's."""
    return paper_table(df, caption, label, position=position, index=index)


def main() -> None:
    import shutil

    for d in (T, S, X):
        d.mkdir(parents=True, exist_ok=True)

    e1_matrix_local()  # regenerate E1 matrix CSVs from local results (feeds Table A/B)

    a, b, e = P.table_a(), P.table_b(), e5_ablation_local()
    c, d = table_c(), table_d()
    xmat = P.e3_cross_matrix().round(4)

    a.to_csv(T / "table_a_detector_comparison.csv", index=False)
    b.to_csv(T / "table_b_architecture_breakdown.csv", index=False)
    c.to_csv(T / "table_c_clip_backbones.csv", index=False)
    d.to_csv(T / "table_d_beta_sensitivity.csv", index=False)
    e.to_csv(T / "table_e_orthogonality_ablation.csv", index=False)
    xmat.to_csv(T / "e3_cross_matrix_mAP.csv")

    # Table F (E4 concept-domain correlation)
    f_src = Path("results/e4_cross_family/tables/domain_concept_correlation.csv")
    if f_src.exists():
        shutil.copy(f_src, T / "table_f_concept_domain_correlation.csv")

    # Refresh ALL full-detail source copies from the fresh local artifacts (keep the bundle consistent).
    E6 = "results/e6_strong_baseline/commfor-model-384"
    E7 = "results/e7_community_eval"
    src_map = {
        "outputs/e1_cross/matrix_mAP.csv": "e1_forensic_matrix_mAP.csv",
        "outputs/e1_cross/matrix_auroc.csv": "e1_forensic_matrix_auroc.csv",
        f"{E6}/tables/e6_main.csv": "e6_main.csv",
        f"{E6}/tables/e6_by_generator.csv": "e6_by_generator.csv",
        f"{E6}/summary.md": "e6_summary.md",
        f"{E7}/tables/community_eval_main.csv": "e7_community_eval_main.csv",
        f"{E7}/tables/community_eval_by_generator.csv": "e7_by_generator.csv",
        f"{E7}/summary.md": "e7_summary.md",
        "results/e4_cross_family/tables/per_generator_ap_clip.csv": "e4_per_generator_ap_clip.csv",
        "results/e4_cross_family/tables/concept_importance_by_domain.csv": "e4_concept_importance_by_domain.csv",
        "results/e4_cross_family/tables/cross_family_cnnspot_to_synthclic_per_generator.csv": "e4_cross_family_cnnspot_to_synthclic_per_generator.csv",
        "results/e4_cross_family/summary.md": "e4_summary.md",
    }
    for src, dst in src_map.items():
        if Path(src).exists():
            shutil.copy(src, S / dst)
    # E2/E3 full-detail source = the fresh main tables (no separate W&B export needed).
    d.to_csv(S / "e2_beta_sensitivity.csv", index=False)
    c.to_csv(S / "e3_backbone_comparison.csv", index=False)

    # ── Manuscript LaTeX tables (paper style: booktabs/scriptsize/caption; per-generator mAP only,
    #    2 decimals; named <experiment>_<readable>.tex). AUROC + pooled-AP columns are dropped. ──
    for old in X.glob("*.tex"):  # remove the old generic table_{a..f}.tex names
        old.unlink()
    NAME = {"cnnspot": "CNNSpot", "synthbuster-plus": "SynthBuster+", "synthclic": "SynthCLIC"}

    # detector comparison (E1/E3/E6/E7) — per-generator mAP per test set + CF-Eval mAP-by-generator
    a_tex = a[
        [
            "detector",
            "cue_family",
            "synthclic_mAP",
            "synthbusterplus_mAP",
            "cnnspot_mAP",
            "cfeval_mAP_by_gen",
        ]
    ].rename(
        columns={
            "cue_family": "Cue",
            "synthclic_mAP": "SynthCLIC",
            "synthbusterplus_mAP": "SynthBuster+",
            "cnnspot_mAP": "CNNSpot",
            "cfeval_mAP_by_gen": "CF-Eval",
        }
    )
    (X / "e1_e3_e6_e7_detector_comparison.tex").write_text(
        _paper_tex(
            a_tex,
            "Per-generator mean AP (mAP) of each detector across test sets. The CF-Eval column is "
            "mAP-by-generator on CommunityForensics-Eval.",
            "tab:rev:detector_comparison",
        )
    )

    # CF-Eval architecture breakdown (E6/E7) — per-architecture mAP
    b_tex = b[
        ["detector", "mAP_by_gen", "GAN", "LatDiff", "PixDiff", "Commercial", "Other"]
    ].rename(columns={"mAP_by_gen": "mAP"})
    (X / "e6_e7_cfeval_architecture.tex").write_text(
        _paper_tex(
            b_tex,
            "Per-architecture mean AP on CommunityForensics-Eval (21 generators across 5 families).",
            "tab:rev:cfeval_architecture",
        )
    )

    # CLIP backbones (E3) — in-domain per-generator mAP, pivot test-set x backbone
    c_piv = c.pivot(index="dataset", columns="backbone", values="test_mAP")
    c_piv = c_piv[
        [
            col
            for col in ["clip_base_patch16", "clip_base_patch32", "clip_large_patch14"]
            if col in c_piv.columns
        ]
    ].rename(
        columns={
            "clip_base_patch16": "ViT-B/16",
            "clip_base_patch32": "ViT-B/32",
            "clip_large_patch14": "ViT-L/14",
        }
    )
    c_piv.index = [NAME.get(i, i) for i in c_piv.index]
    c_piv.index.name = "Test Set (in-domain)"
    (X / "e3_clip_backbones.tex").write_text(
        _paper_tex(
            c_piv.reset_index(),
            "In-domain per-generator mAP across CLIP backbones.",
            "tab:rev:clip_backbones",
        )
    )

    # E3 cross-dataset matrix (paper tab:clip:test_results style: Test Set rows x Training Set cols)
    xm = xmat.rename(index=NAME, columns=NAME).T  # xmat rows=train, cols=eval -> transpose
    xm.index.name = "Test Set"
    (X / "e3_cross_dataset_mAP.tex").write_text(
        _paper_tex(
            xm.reset_index(),
            "Per-generator mean AP for CLIP detectors trained on different sets (columns) and evaluated "
            "on different test sets (rows).",
            "tab:rev:e3_cross",
        )
    )

    # beta sensitivity (E2) — multi-seed (mean +/- std) when available, else single run.
    dm = table_d_multiseed()
    if not dm.empty and int(dm["n_seeds"].max()) > 1:
        dm.round(4).to_csv(T / "table_d_beta_sensitivity_multiseed.csv", index=False)
        n = int(dm["n_seeds"].min())
        et = pd.DataFrame(
            {
                r"$\beta$": [_fmt_beta(b) for b in dm["beta"]],
                "mAP": [f"{m:.2f} $\\pm$ {s:.2f}" for m, s in zip(dm.mAP_mean, dm.mAP_std)],
                r"\#active": [
                    f"{m:.1f} $\\pm$ {s:.1f}" for m, s in zip(dm.active_mean, dm.active_std)
                ],
                r"gate mass": [
                    f"{m:.2f} $\\pm$ {s:.2f}" for m, s in zip(dm.gate_mass_mean, dm.gate_mass_std)
                ],
                r"max$|W_{\mathrm{clf}}|$": [
                    f"{m:.0f} $\\pm$ {s:.0f}" for m, s in zip(dm.maxWclf_mean, dm.maxWclf_std)
                ],
            }
        )
        (X / "e2_beta_sensitivity.tex").write_text(
            _paper_tex(
                et,
                f"Concept-sparsity weight $\\beta$ vs detection on SynthCLIC (per-generator mAP), mean number "
                f"of active concepts, total gate mass, and max classifier weight (mean $\\pm$ std over {n} "
                f"seeds). As $\\beta$ grows the gates collapse but the classifier rescales "
                f"(max$|W_{{\\mathrm{{clf}}}}|$ grows), so mAP is preserved at near-zero active concepts.",
                "tab:rev:beta_sensitivity",
            )
        )
    else:
        d_tex = d[["beta", "test/mAP", "test/mean_active_concepts"]].copy()
        d_tex["beta"] = d_tex["beta"].map(_fmt_beta)
        d_tex = d_tex.rename(
            columns={
                "beta": r"$\beta$",
                "test/mAP": "mAP",
                "test/mean_active_concepts": r"\#active",
            }
        )
        (X / "e2_beta_sensitivity.tex").write_text(
            _paper_tex(
                d_tex,
                "Concept-sparsity weight $\\beta$ vs detection (per-generator mAP) and mean number of active "
                "concepts on SynthCLIC.",
                "tab:rev:beta_sensitivity",
            )
        )

    # orthogonality ablation (E5) — mAP + orthogonality scores (lower = more orthogonal)
    e_tex = e[["variant", "test_mAP", "weight_ortho_score", "activation_ortho_score"]].copy()
    e_tex["variant"] = (
        e_tex["variant"]
        .map({"none": "none", "activation_ortho": "activation", "weight_ortho": "weight"})
        .fillna(e_tex["variant"])
    )
    e_tex = e_tex.rename(
        columns={
            "variant": "Aux. loss",
            "test_mAP": "mAP",
            "weight_ortho_score": r"W-ortho $\downarrow$",
            "activation_ortho_score": r"Act-ortho $\downarrow$",
        }
    )
    (X / "e5_orthogonality_ablation.tex").write_text(
        _paper_tex(
            e_tex,
            "Orthogonality ablation on SynthCLIC: detection mAP is unchanged across variants; the "
            "activation loss decorrelates activations and already yields roughly-orthogonal weights.",
            "tab:rev:orthogonality",
        )
    )

    # concept-domain correlation (E4) — Spearman correlation matrix (no AP/AUROC)
    f_csv = T / "table_f_concept_domain_correlation.csv"
    if f_csv.exists():
        f_df = pd.read_csv(f_csv, index_col=0)
        f_df.index.name = "Train domain"
        (X / "e4_concept_domain_correlation.tex").write_text(
            _paper_tex(
                f_df.reset_index(),
                "Spearman correlation of per-concept detector importance (W\\_classifier) across "
                "training domains.",
                "tab:rev:concept_corr",
            )
        )

    # Canonical-figure replacements (Fig 1/5/6/7) — refresh from outputs/e8/figures/ (non-destructive).
    figs = P.copy_e8_figures(EXPORT / "figures")
    print(f"Copied {len(figs)} E8/canonical figures into {EXPORT / 'figures'}: {sorted(set(figs))}")

    # Table G — publication detector-performance table (per-generator mAP across all test sets).
    try:
        import subprocess

        subprocess.run(
            ["python", "scripts/export/export_detector_table.py"], check=True, capture_output=True
        )
        print("Rebuilt table_g_detector_mAP (per-generator mAP).")
    except Exception as e:  # noqa: BLE001
        print(f"(skipped table_g rebuild: {e})")

    # KEY_NUMBERS.md + MANIFEST.md (reuse the packager's writer with the fresh tables)
    P.write_docs(a, b, e)

    print("Rebuilt tables A–F + e3 matrix + KEY_NUMBERS/MANIFEST (Convention A, local results).")
    print("\nTable A:\n", a.to_string(index=False))
    print("\nTable D (beta):\n", d.to_string(index=False))
    print("\nTable E (ortho):\n", e.to_string(index=False))


if __name__ == "__main__":
    main()
