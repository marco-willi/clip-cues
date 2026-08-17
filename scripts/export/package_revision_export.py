#!/usr/bin/env python
"""Package all E1-E7 revision artifacts into a self-contained bundle for the write-up repo.

Idempotent: rebuilds `reproduction/revision_export/` from existing `outputs/`+`results/` artifacts (no experiments
re-run) and tars it. Produces the 6 manuscript tables (A-F) + their source tables, the E4 figures
(PNG+PDF), optional .tex stubs, a MANIFEST mapping each artifact -> manuscript table / reviewer point /
source / metric convention, and KEY_NUMBERS.md.

Defaults (per PLAN_PACKAGE_REVISION_DATA.md): common metric for the comparison = **overall mAP/AP**
(per-generator surfaced separately for E6/E7). The **canonical CLIP detector is the linear probe**
(k=1/logistic) end-to-end — cross-dataset rows from E3 AND the CF-Eval cell (T0:
`linear_probe_synthclic`, computed from cached CF-Eval embeddings via the E7 pipeline; see
config-audit.md §F). The k=8 orthogonal head (`clip_orthogonal_synthclic`) is kept as an appendix
ablation (E5: ortho ≈ linear at zero detection cost; CF-Eval mAP-by-gen 0.732 vs 0.734).

    python scripts/export/package_revision_export.py        # -> reproduction/revision_export/ + revision_export.tar.gz
"""

from __future__ import annotations

import glob
import json
import shutil
import tarfile
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(".")
EXPORT = Path("revision_export")
NA = "n/a"


# ── source loaders ──────────────────────────────────────────────────────────────────────────
def e1_forensic_matrix() -> pd.DataFrame:
    df = pd.read_csv("outputs/e1_cross/matrix_mAP.csv", index_col=0)
    df.columns = [c.strip() for c in df.columns]  # synthclic / synthbuster-plus / cnnspot
    return df


def e3_cross_matrix() -> pd.DataFrame:
    """Aggregate the L/14 CLIP linear-probe cross-dataset matrix from per-cell metrics.json.

    Two generations of E3 runs coexist per cell. They are the same trained heads (per-image scores
    agree to 1.5e-07) but the 2026-06-24 04:4x generation evaluates CNNSpot on **cnnspot-small**
    (4,000 imgs, 20 generators) while the 20:0x generation uses the **full CNNSpot benchmark test
    set** (108,310, 21). Selecting by glob order would let the CNNSpot column silently change
    population between rebuilds, so the **latest run_id wins**, explicitly — the same rule
    ``e5_ablation`` uses. See reproduction/experiments/final_consolidation/TableB-per-generator/README.md for the
    one cell where the two generations still leave the column inhomogeneous.
    """
    latest: dict[tuple[str, str], tuple[str, float]] = {}
    for f in glob.glob("results/e3_xdataset/*/*/metrics.json"):
        m = json.load(open(f))
        if m.get("backbone") != "clip_large_patch14":
            continue
        tr, ev, run_id = m.get("train_dataset"), m.get("eval_dataset"), Path(f).parent.name
        if tr and ev and (run_id > latest.get((tr, ev), ("", 0.0))[0]):
            latest[(tr, ev)] = (run_id, m.get("mAP"))
    cells: dict[str, dict[str, float]] = {}
    for (tr, ev), (_, value) in latest.items():
        cells.setdefault(tr, {})[ev] = value
    return pd.DataFrame(cells).T  # rows=train, cols=eval


def e6_main() -> pd.DataFrame:
    return pd.read_csv("results/e6_strong_baseline/commfor-model-384/tables/e6_main.csv").set_index(
        "dataset"
    )


def e7_main() -> pd.DataFrame:
    return pd.read_csv("results/e7_community_eval/tables/community_eval_main.csv").set_index(
        "detector"
    )


def e5_ablation() -> pd.DataFrame:
    # Two generations of runs coexist: pre-fix `<variant>/<run_id>/` (schema: variant, test/mAP)
    # and the faithful Convention-A re-run `<variant>__clip_large_patch14__synthclic__to__synthclic/
    # <run_id>/` (schema: mAP, backbone, train_dataset). Collapse to the CANONICAL variant name
    # (strip the `__...` suffix) and keep the latest run_id — which selects the faithful re-run —
    # reading each metric with a schema fallback (`test/mAP` old ↔ `mAP` new).
    latest: dict[str, tuple[str, str]] = {}
    for f in glob.glob("results/e5_orthogonality/*/*/metrics.json"):
        p = Path(f)
        variant, run_id = p.parent.parent.name.split("__")[0], p.parent.name
        if variant not in latest or run_id > latest[variant][0]:
            latest[variant] = (run_id, f)
    rows = []
    for variant, (_, f) in latest.items():
        m = json.load(open(f))
        rows.append(
            {
                "variant": variant,
                # test_mAP is Convention A (per-generator mean AP, the paper's metric); pooled AP kept
                # alongside for transparency (it is the quantity the old SimpleMetrics path reported).
                "test_mAP": m.get("test/mAP", m.get("mAP")),
                "test_pooled_ap": m.get("test/pooled_ap", m.get("pooled_ap")),
                "test_auroc": m.get("test/auroc", m.get("auroc")),
                "weight_ortho_score": m["weight_ortho_score"],
                "activation_ortho_score": m["activation_ortho_score"],
            }
        )
    order = {"none": 0, "activation_ortho": 1, "weight_ortho": 2}
    return (
        pd.DataFrame(rows).sort_values("variant", key=lambda s: s.map(order)).reset_index(drop=True)
    )


# ── Table A: combined detector comparison ───────────────────────────────────────────────────
def table_a() -> pd.DataFrame:
    e1, e3, e6, e7 = e1_forensic_matrix(), e3_cross_matrix(), e6_main(), e7_main()
    SC, SB, CN = "synthclic", "synthbuster-plus", "cnnspot"

    def g(df, r, c):  # safe lookup
        try:
            return round(float(df.loc[r, c]), 4)
        except Exception:
            return NA

    # Canonical CLIP detector = the linear probe (k=1/logistic) end-to-end (see config-audit.md §F).
    # The CF-Eval cell falls back to the k=8 orthogonal head only if the linear-probe CF-Eval row is
    # not yet present (i.e. before T0's eval_heads_on_cf_embeddings.py has been run).
    clip_cf = (
        "linear_probe_synthclic"
        if "linear_probe_synthclic" in e7.index
        else "clip_orthogonal_synthclic"
    )
    rows = [
        # cross-dataset cols = CLIP linear probe (E3); CF-Eval = CLIP linear probe (T0) — single head, consistent
        {
            "detector": "CLIP linear probe (SynthCLIC)",
            "exposure": "SynthCLIC",
            "cue_family": "semantic/photographic",
            "synthclic_mAP": g(e3, SC, SC),
            "synthbusterplus_mAP": g(e3, SC, SB),
            "cnnspot_mAP": g(e3, SC, CN),
            "cfeval_overall_ap": g(e7, clip_cf, "overall_ap"),
            "cfeval_mAP_by_gen": g(e7, clip_cf, "mAP_by_gen"),
        },
        {
            "detector": "CLIP linear probe (CNNSpot)",
            "exposure": "CNNSpot",
            "cue_family": "semantic/photographic",
            "synthclic_mAP": g(e3, CN, SC),
            "synthbusterplus_mAP": g(e3, CN, SB),
            "cnnspot_mAP": g(e3, CN, CN),
            "cfeval_overall_ap": NA,
            "cfeval_mAP_by_gen": NA,
        },
        {
            "detector": "Forensic CNN (SynthCLIC)",
            "exposure": "SynthCLIC",
            "cue_family": "low-level forensic",
            "synthclic_mAP": g(e1, SC, SC),
            "synthbusterplus_mAP": g(e1, SC, SB),
            "cnnspot_mAP": g(e1, SC, CN),
            "cfeval_overall_ap": g(e7, "cnnspot_synthclic", "overall_ap"),
            "cfeval_mAP_by_gen": g(e7, "cnnspot_synthclic", "mAP_by_gen"),
        },
        {
            "detector": "Forensic CNN (ProGAN/CNNSpot)",
            "exposure": "ProGAN",
            "cue_family": "low-level forensic",
            "synthclic_mAP": g(e1, "cnnspot-progan-zeroshot", SC),
            "synthbusterplus_mAP": g(e1, "cnnspot-progan-zeroshot", SB),
            "cnnspot_mAP": g(e1, "cnnspot-progan-zeroshot", CN),
            "cfeval_overall_ap": g(e7, "cnnspot_progan", "overall_ap"),
            "cfeval_mAP_by_gen": g(e7, "cnnspot_progan", "mAP_by_gen"),
        },
        {
            "detector": "CommunityForensics-384 (out-of-the-box)",
            "exposure": "broad",
            "cue_family": "broad-generator detector",
            "synthclic_mAP": g(e6, SC, "overall_ap"),
            "synthbusterplus_mAP": g(e6, "synthbuster_plus", "overall_ap"),
            "cnnspot_mAP": g(e6, CN, "overall_ap"),
            "cfeval_overall_ap": g(e6, "community_forensics_eval", "overall_ap"),
            "cfeval_mAP_by_gen": g(e6, "community_forensics_eval", "mAP_by_generator"),
        },
    ]
    return pd.DataFrame(rows)


# ── Table B: architecture breakdown on CF-Eval (CLIP/forensic from E7 + CommFor from E6) ──────
def table_b() -> pd.DataFrame:
    archs = ["GAN", "LatDiff", "PixDiff", "Commercial", "Other"]
    e7arch = pd.read_csv("results/e7_community_eval/tables/community_eval_by_architecture.csv")
    e6arch = pd.read_csv(
        "results/e6_strong_baseline/commfor-model-384/tables/e6_by_architecture.csv"
    )
    e7m, e6m = e7_main(), e6_main()
    # Canonical CLIP detector = linear probe (config-audit.md §F); fall back to the k=8 ortho head only
    # if the linear-probe CF-Eval row isn't present yet. The ortho per-architecture APs stay available in
    # results/e7_community_eval/tables/community_eval_by_architecture.csv (appendix; E5 head-invariance).
    clip_det = (
        "linear_probe_synthclic"
        if "linear_probe_synthclic" in e7m.index
        else "clip_orthogonal_synthclic"
    )
    clip_lbl = (
        "CLIP linear probe (SynthCLIC)"
        if clip_det == "linear_probe_synthclic"
        else "CLIP-orthogonal (SynthCLIC)"
    )
    label = {
        clip_det: clip_lbl,
        "cnnspot_synthclic": "Forensic CNN (SynthCLIC)",
        "cnnspot_progan": "Forensic CNN (ProGAN/CNNSpot)",
    }
    rows = []
    for det, name in label.items():
        piv = e7arch[e7arch.detector == det].set_index("architecture")["mAP"]
        rows.append(
            {
                "detector": name,
                "overall_ap": round(float(e7m.loc[det, "overall_ap"]), 4),
                "mAP_by_gen": round(float(e7m.loc[det, "mAP_by_gen"]), 4),
                **{a: round(float(piv.get(a, float("nan"))), 4) for a in archs},
            }
        )
    cf = e6arch.set_index("architecture")["mAP"]
    rows.append(
        {
            "detector": "CommunityForensics-384 (out-of-the-box)",
            "overall_ap": round(float(e6m.loc["community_forensics_eval", "overall_ap"]), 4),
            "mAP_by_gen": round(float(e6m.loc["community_forensics_eval", "mAP_by_generator"]), 4),
            **{a: round(float(cf.get(a, float("nan"))), 4) for a in archs},
        }
    )
    return pd.DataFrame(rows)


# ── bundle assembly ──────────────────────────────────────────────────────────────────────────
def copy_e8_figures(fdir: Path) -> list[str]:
    """Copy the canonical-figure replacements (Fig 1/5/6/7) from their durable home
    ``outputs/e8/figures/`` into the export ``figures/`` dir (PNG+PDF, already vector-correct).

    Generate the figures first with the ``scripts/plot/plot_*.py`` drivers (see MANIFEST). Globs so
    ``concept_explanation_<id>.{png,pdf}`` is picked up for any content id. Shared by the destructive
    packager (``build_and_copy``) and the safe local rebuild (``rebuild_export_local``)."""
    e8fig = Path("outputs/e8/figures")
    patterns = [
        "score_distributions",  # Fig 5 — deterministic linear-probe score densities
        "cue_profile",  # Fig 7 — bootstrap-stable cue profile
        "concept_explanation_*",  # Fig 1 — concept local explanation (per content id)
        "synthclic-logreg-direction",  # Fig 6 — deterministic extreme-sample montages
        "cnnspot-logreg-direction",
        # Appendix (full-tier) figures — analysis precomputed in outputs/e8/{clipiqa,head_decomp}/
        "clipiqa_axes",  # CLIP-IQA perceptual axes
        "head_decomp_heatmap",  # per-head layer×head direct-AUROC heatmap
        "head_decomp_direct_vs_causal",  # direct top-k vs causal ablation
        "head_concepts",  # head → nearest antonym concepts (Gandelsman naming)
        "paired_cue_shift",  # content-controlled cue shifts (antonyms; SynthCLIC pairing)
        "paired_cue_shift_textspan",  # TextSpan residual semantic signature
        "paired_head_shift",  # per-head content-controlled shift (cross-validated heads)
    ]
    fdir.mkdir(parents=True, exist_ok=True)
    copied = []
    for pat in patterns:
        for f in sorted(e8fig.glob(f"{pat}.png")) + sorted(e8fig.glob(f"{pat}.pdf")):
            shutil.copy(f, fdir / f.name)
            copied.append(f.name)
    return copied


# manuscript table id -> (builder|source path, kind)
def build_and_copy() -> dict[str, Path]:
    tdir, sdir, fdir, xdir = (
        EXPORT / "tables",
        EXPORT / "tables" / "source",
        EXPORT / "figures",
        EXPORT / "tex",
    )
    for d in (tdir, sdir, fdir, xdir):
        d.mkdir(parents=True, exist_ok=True)

    # Table A/B/E built here; C/D/F copied from existing CSVs.
    a, b, e = table_a(), table_b(), e5_ablation()
    a.to_csv(tdir / "table_a_detector_comparison.csv", index=False)
    b.to_csv(tdir / "table_b_architecture_breakdown.csv", index=False)
    e.to_csv(tdir / "table_e_orthogonality_ablation.csv", index=False)
    e3_cross_matrix().round(4).to_csv(tdir / "e3_cross_matrix_mAP.csv")
    shutil.copy("outputs/e3/backbone_comparison.csv", tdir / "table_c_clip_backbones.csv")
    shutil.copy("outputs/e2/beta_sensitivity.csv", tdir / "table_d_beta_sensitivity.csv")
    shutil.copy(
        "results/e4_cross_family/tables/domain_concept_correlation.csv",
        tdir / "table_f_concept_domain_correlation.csv",
    )

    # source tables (full detail) + summaries
    src = {
        "outputs/e1_cross/matrix_mAP.csv": "e1_forensic_matrix_mAP.csv",
        "outputs/e1_cross/matrix_auroc.csv": "e1_forensic_matrix_auroc.csv",
        "outputs/e2/beta_sensitivity.csv": "e2_beta_sensitivity.csv",
        "outputs/e3/backbone_comparison.csv": "e3_backbone_comparison.csv",
        "results/e4_cross_family/tables/per_generator_ap_clip.csv": "e4_per_generator_ap_clip.csv",
        "results/e4_cross_family/tables/concept_importance_by_domain.csv": "e4_concept_importance_by_domain.csv",
        "results/e4_cross_family/summary.md": "e4_summary.md",
        "results/e6_strong_baseline/commfor-model-384/tables/e6_main.csv": "e6_main.csv",
        "results/e6_strong_baseline/commfor-model-384/tables/e6_by_generator.csv": "e6_by_generator.csv",
        "results/e6_strong_baseline/commfor-model-384/summary.md": "e6_summary.md",
        "results/e7_community_eval/tables/community_eval_main.csv": "e7_community_eval_main.csv",
        "results/e7_community_eval/tables/community_eval_by_generator.csv": "e7_by_generator.csv",
        "results/e7_community_eval/summary.md": "e7_summary.md",
    }
    for s, dst in src.items():
        if Path(s).exists():
            shutil.copy(s, sdir / dst)

    # figures: PNG + PDF (rasterised via Pillow; re-render via run_cross_family_analysis for vector)
    figmap = {
        "per-generator-ap-clip": "per_generator_ap_clip",
        "concept-shift-heatmap": "concept_shift_heatmap",
        "domain-concept-correlation": "domain_concept_correlation",
    }
    for srcname, dstname in figmap.items():
        png = Path(f"results/e4_cross_family/figures/{srcname}.png")
        if png.exists():
            shutil.copy(png, fdir / f"{dstname}.png")
            Image.open(png).convert("RGB").save(fdir / f"{dstname}.pdf", "PDF", resolution=150)

    copy_e8_figures(fdir)

    # .tex stubs for A-F
    tex_tables = {
        "table_a": a,
        "table_b": b,
        "table_c": pd.read_csv(tdir / "table_c_clip_backbones.csv"),
        "table_d": pd.read_csv(tdir / "table_d_beta_sensitivity.csv"),
        "table_e": e,
        "table_f": pd.read_csv(tdir / "table_f_concept_domain_correlation.csv"),
    }
    for name, df in tex_tables.items():
        (xdir / f"{name}.tex").write_text(df.to_latex(index=False, float_format="%.3f"))

    return {"a": tdir / "table_a_detector_comparison.csv"}


MANIFEST = """# Revision export — manifest

Bundle of E1-E7 result artifacts for the write-up/rebuttal repo (ARRAY-D-26-00829). Generated by
`scripts/export/package_revision_export.py` from `outputs/`+`results/` — re-run `make package-revision` to refresh.

**Metric conventions:** SynthCLIC/SynthBuster+/CNNSpot columns are **overall mAP/AP** (not per-generator —
E1/E3 saved no per-image predictions). E6/E7 also provide **mAP-by-generator**. The **canonical CLIP
detector is the linear probe** (k=1/logistic) for ALL columns: cross-dataset from E3, CF-Eval from
`linear_probe_synthclic` (T0 — cached CF-Eval embeddings scored via the E7 pipeline, validated to
reproduce the ortho parquet per-image at corr 1.0). The k=8 orthogonal head is an appendix ablation
(E5 head-invariance; CF-Eval mAP-by-gen 0.732 vs 0.734). **CommunityForensics-Eval is in-distribution
for CommunityForensics-384** — caption as a reference, not an independent external test.

| manuscript table | file | reviewer pt | source exp | notes |
|---|---|---|---|---|
| A — detector comparison (main) | tables/table_a_detector_comparison.csv | R1.5, R3.3 | E1+E3+E6+E7 | overall mAP/AP; CLIP cross=linear probe, CF-Eval=linear probe; n/a = not run |
| **G — detector performance (per-generator mAP)** | tables/table_g_detector_mAP.csv, tex/g_detector_mAP.tex | R1.5 | E1+E3+E6+E7+T0 | **publication table**: 3 CLIP-linear + 3 forensic (per train set) + CommFor-384; cols SynthCLIC/SynthBuster+/CNNSpot/CF-Eval; all **per-generator mAP**; Forensic(SynthBuster+)→CF-Eval = n/a (checkpoint unavailable). `scripts/export/export_detector_table.py` |
| B — CF-Eval architecture breakdown | tables/table_b_architecture_breakdown.csv | R1.6 | E7 (CLIP/forensic) + E6 (CommFor) | per-arch mAP {GAN,LatDiff,PixDiff,Commercial,Other} |
| C — CLIP backbones | tables/table_c_clip_backbones.csv | R1 (backbones) | E3 | in-domain SynthCLIC per backbone |
| D — beta sensitivity | tables/table_d_beta_sensitivity.csv | R3 (E2) | E2 | mAP vs #active concepts |
| E — orthogonality ablation | tables/table_e_orthogonality_ablation.csv | R3 (E5) | E5 | test mAP + weight/activation ortho scores |
| F — concept-domain correlation | tables/table_f_concept_domain_correlation.csv | R1.6 (E4) | E4 | Spearman corr of per-concept importance across domains |
| (helper) E3 cross-dataset matrix | tables/e3_cross_matrix_mAP.csv | R1 | E3 | CLIP linear-probe train x eval mAP (feeds Table A). Latest run_id per cell wins — do NOT revert to glob order, the two E3 generations evaluate CNNSpot on different populations |
| **Appendix — per-generator detail (`tab:clip:test_results_detail`)** | tables/clip_per_generator_detail.csv, tex/clip_per_generator_detail.tex, clip_per_generator_detail_README.md | R1.5 | TableB (E3 predictions) | **regenerated 2026-08-09**, replaces the hand-written Table 13 which disagreed with Table 3 in 6 of 9 off-diagonal blocks. 38 generators x 4 train sets x {ACC, AP}, all from the matched no-aug `D_h`. ACC at p>0.5; CNNSpot `matched` pairing on the FULL benchmark test set (21 generators), SynthCLIC/SB+ `shared`. Column means reproduce e3_cross_matrix in 11/12 cells exactly; the 12th (Combined->CNNSpot) is a recorded frame mismatch **in the matrix** — see the README. `scripts/finalexp/run_appendix_per_generator.py` + `scripts/export/export_per_generator_table.py` |
| figures (E4) | figures/{per_generator_ap_clip,concept_shift_heatmap,domain_concept_correlation}.{png,pdf} | R1.6 | E4 | PDF is rasterised; re-render via scripts/run/run_cross_family_analysis.py for vector |
| **Fig 1 — concept local explanation** | figures/concept_explanation_<id>.{png,pdf} | R1.7 | concept model | content-controlled (real + synthetic counterparts); **switchable** via `scripts/plot/plot_concept_explanation.py --image-id`; code in src/clip_cues_research/figures/ |
| **Fig 5 — score distributions** | figures/score_distributions.{png,pdf} | R1.5 | E3/canonical | deterministic linear-probe (k=1) real-vs-synth score densities, AP/AUROC in titles; `scripts/plot/plot_score_distributions.py` |
| **Fig 6 — extreme-sample montages** | figures/{synthclic,cnnspot}-logreg-direction.{png,pdf} | R1.7 | E8 | deterministic (logreg/k=1) direction top/bottom images; `scripts/plot/plot_direction_samples.py --method logreg` |
| **Fig 7 — stable cue profile** | figures/cue_profile.{png,pdf} | R1.7 | E8 | bootstrap-stable cue importances + CV shading (replaces one-seed top-30 concepts); `scripts/plot/plot_cue_profile.py` |
| **App. — CLIP-IQA axes** | figures/clipiqa_axes.{png,pdf} | R1.7 | E8 | per-axis real-vs-synth AUROC; perceived quality decoupled from pixels; `scripts/plot/plot_clipiqa.py` |
| **App. — head-decomp heatmap** | figures/head_decomp_heatmap.{png,pdf} | R1.7 | E8 | layer×head direct discriminative AUROC; `scripts/plot/plot_head_decomp.py` |
| **App. — direct vs causal** | figures/head_decomp_direct_vs_causal.{png,pdf} | R1.7 | E8 | top-k direct vs forward-pass ablation, per-generator mAP (distributed/redundant caveat); `scripts/plot/plot_head_decomp.py` |
| **App. — head → concepts** | figures/head_concepts.{png,pdf} | R1.7 | E8 | top heads + nearest antonym concepts + alignment cosine (≈0.1, weak ⇒ themes); `scripts/analyze/name_heads.py` + `scripts/plot/plot_head_concepts.py` |
| **Paired cue shifts** | figures/paired_cue_shift.{png,pdf} | R1.7 | E8 | content-controlled cue shifts (SynthCLIC pairing): synthetic less hue/skin/hands/geometry, more grid/vibrance; `scripts/plot/plot_paired_cue_shift.py` |
| Paired cue shifts (TextSpan) | figures/paired_cue_shift_textspan.{png,pdf} | R1.7 | E8 | residual semantic signature: synthetic less like specific real-world locations / more generic |
| Paired per-head shift | figures/paired_head_shift.{png,pdf} | R1.7 | E8 | content-controlled per-head attribution; ★ = heads also flagged by the unpaired decomposition (cross-validated) |
| source tables (full detail) | tables/source/*.csv, *_summary.md | — | all | unaggregated tables + narrative blurbs for caption seeds |
| experimental details (per experiment) | experimental_details/e{1..8}.md | all | E1–E8 | curated methods/design/results write-ups; mirror `docs/revision_state/E*.md` (refresh from there) — e2–e5 are extended; e1/e6/e7/e8 added 2026-06-28 |
| LaTeX tables | tex/{e1_e3_e6_e7_detector_comparison,e6_e7_cfeval_architecture,e3_clip_backbones,e3_cross_dataset_mAP,e2_beta_sensitivity,e5_orthogonality_ablation,e4_concept_domain_correlation}.tex | all | all | paper-styled (booktabs, \\scriptsize, caption+label); **per-generator mAP only** (no AUROC/pooled AP); 2 decimals; `\\input{}`-able |
"""


def write_docs(a: pd.DataFrame, b: pd.DataFrame, e: pd.DataFrame) -> None:
    (EXPORT / "MANIFEST.md").write_text(MANIFEST)
    thesis = (
        "Low-level forensic CNNs are excellent in-domain but brittle under dataset/generator shift; "
        "CLIP-based detectors transfer more robustly between diffusion datasets but are uneven and weak on "
        "GAN-heavy data; the CommunityForensics out-of-the-box detector shows broad generator exposure "
        "substantially reduces dataset-locking yet retains architecture-specific gaps. SID evidence is "
        "complementary — combine semantic CLIP cues, forensic fingerprints, and broad/continual exposure."
    )
    lines = [
        "# Key numbers (headline; full values in tables/)",
        "",
        "## Central thesis",
        "",
        thesis,
        "",
        "## Table A — detector comparison (overall mAP/AP)",
        "",
        a.to_markdown(index=False) if _md() else a.to_string(index=False),
        "",
        "## Table B — CF-Eval architecture breakdown",
        "",
        b.to_markdown(index=False) if _md() else b.to_string(index=False),
        "",
        "## Table E — orthogonality ablation",
        "",
        e.to_markdown(index=False) if _md() else e.to_string(index=False),
    ]
    (EXPORT / "KEY_NUMBERS.md").write_text("\n".join(lines))


def _md() -> bool:
    try:
        import tabulate  # noqa: F401

        return True
    except Exception:
        return False


def main() -> None:
    if EXPORT.exists():
        shutil.rmtree(EXPORT)
    build_and_copy()
    write_docs(table_a(), table_b(), e5_ablation())
    with tarfile.open("revision_export.tar.gz", "w:gz") as tar:
        tar.add(EXPORT, arcname="revision_export")
    n = sum(1 for _ in EXPORT.rglob("*") if _.is_file())
    print(f"Packaged {n} files into {EXPORT}/ + revision_export.tar.gz")
    print("\nTable A (detector comparison):")
    print(table_a().to_string(index=False))


if __name__ == "__main__":
    main()
