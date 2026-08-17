#!/usr/bin/env python
"""Publication detector-performance table — per-generator mAP across all test sets.

Builds the main detector comparison the revision needs: rows = the trained detectors (CLIP linear probe
and forensic CNN, each trained on SynthCLIC / SynthBuster+ / CNNSpot) + CommunityForensics-384
(off-the-shelf); columns = test sets {SynthCLIC, SynthBuster+, CNNSpot, CommunityForensics-Eval}; every
cell is **per-generator mean AP (Convention A)** = mean AP over generator-paired real/synthetic groups.

Sources (no recompute here — all cells already exist except where noted):
  * CLIP linear probe  -> reproduction/revision_export/tables/e3_cross_matrix_mAP.csv (E3; rows=train, cols=eval)
  * Forensic CNN       -> outputs/e1_cross/matrix_mAP.csv (E1; rows=train, cols=eval)
  * CommunityForensics -> results/e6_strong_baseline/commfor-model-384/tables/e6_main.csv (E6)
  * CF-Eval column     -> results/e7_community_eval/tables/community_eval_overall.csv (E7 + T0:
                          linear_probe_{synthclic,synthbuster,cnnspot} via eval_heads_on_cf_embeddings.py)

Writes reproduction/revision_export/tables/table_g_detector_mAP.csv + tex/g_detector_mAP.tex.
    python scripts/export/export_detector_table.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

EXPORT = Path("revision_export")
NA = "n/a"


def _cell(df: pd.DataFrame, row: str, col: str) -> float | str:
    try:
        return round(float(df.loc[row, col]), 4)
    except Exception:
        return NA


def _cfeval_map() -> dict[str, float]:
    """detector -> CF-Eval per-generator mAP, from the E7 overall table (mAP_by_generator)."""
    ov = pd.read_csv("results/e7_community_eval/tables/community_eval_overall.csv").set_index(
        "detector"
    )
    return {d: round(float(ov.loc[d, "mAP_by_generator"]), 4) for d in ov.index}


def _fmt_tex_cell(v) -> str:
    if isinstance(v, str):
        try:
            return f"{float(v):.2f}"
        except ValueError:
            return v
    if pd.isna(v):
        return NA
    return f"{float(v):.2f}"


def _row_tex(train: str, vals: list) -> str:
    svals = " & ".join(_fmt_tex_cell(v) for v in vals)
    return f"& {train} & {svals} \\\\"


def main() -> None:
    e3 = pd.read_csv(EXPORT / "tables/e3_cross_matrix_mAP.csv", index_col=0)  # CLIP linear probe
    e1 = pd.read_csv("outputs/e1_cross/matrix_mAP.csv", index_col=0)  # forensic CNN
    e1.columns = [c.strip() for c in e1.columns]
    e6 = pd.read_csv("results/e6_strong_baseline/commfor-model-384/tables/e6_main.csv").set_index(
        "dataset"
    )
    cf = _cfeval_map()
    SC, SB, CN = "synthclic", "synthbuster-plus", "cnnspot"

    rows = [
        # CLIP linear probe (E3 cross matrix; CF-Eval from E7/T0 linear_probe_* parquet)
        {
            "detector": "CLIP linear probe",
            "train": "SynthCLIC",
            "cue": "CLIP semantic",
            "SynthCLIC": _cell(e3, SC, SC),
            "SynthBuster+": _cell(e3, SC, SB),
            "CNNSpot": _cell(e3, SC, CN),
            "CF-Eval": cf.get("linear_probe_synthclic", NA),
        },
        {
            "detector": "CLIP linear probe",
            "train": "SynthBuster+",
            "cue": "CLIP semantic",
            "SynthCLIC": _cell(e3, SB, SC),
            "SynthBuster+": _cell(e3, SB, SB),
            "CNNSpot": _cell(e3, SB, CN),
            "CF-Eval": cf.get("linear_probe_synthbuster", NA),
        },
        {
            "detector": "CLIP linear probe",
            "train": "CNNSpot",
            "cue": "CLIP semantic",
            "SynthCLIC": _cell(e3, CN, SC),
            "SynthBuster+": _cell(e3, CN, SB),
            "CNNSpot": _cell(e3, CN, CN),
            "CF-Eval": cf.get("linear_probe_cnnspot", NA),
        },
        {
            "detector": "CLIP linear probe",
            "train": "Combined",
            "cue": "CLIP semantic",
            "SynthCLIC": _cell(e3, "combined", SC),
            "SynthBuster+": _cell(e3, "combined", SB),
            "CNNSpot": _cell(e3, "combined", CN),
            "CF-Eval": cf.get("linear_probe_combined", NA),
        },
        # Forensic CNN (E1 cross matrix; CF-Eval from E7 cnnspot_* parquet)
        {
            "detector": "Forensic CNN",
            "train": "SynthCLIC",
            "cue": "low-level forensic",
            "SynthCLIC": _cell(e1, SC, SC),
            "SynthBuster+": _cell(e1, SC, SB),
            "CNNSpot": _cell(e1, SC, CN),
            "CF-Eval": cf.get("cnnspot_synthclic", NA),
        },
        {
            "detector": "Forensic CNN",
            "train": "SynthBuster+",
            "cue": "low-level forensic",
            "SynthCLIC": _cell(e1, SB, SC),
            "SynthBuster+": _cell(e1, SB, SB),
            "CNNSpot": _cell(e1, SB, CN),
            "CF-Eval": cf.get("cnnspot_synthbuster", NA),
        },  # NOT run (no checkpoint) -> n/a
        {
            "detector": "Forensic CNN",
            "train": "CNNSpot/ProGAN",
            "cue": "low-level forensic",
            "SynthCLIC": _cell(e1, "cnnspot-progan-zeroshot", SC),
            "SynthBuster+": _cell(e1, "cnnspot-progan-zeroshot", SB),
            "CNNSpot": _cell(e1, "cnnspot-progan-zeroshot", CN),
            "CF-Eval": cf.get("cnnspot_progan", NA),
        },
        {
            "detector": "Forensic CNN",
            "train": "Combined",
            "cue": "low-level forensic",
            "SynthCLIC": _cell(e1, "combined", SC),
            "SynthBuster+": _cell(e1, "combined", SB),
            "CNNSpot": _cell(e1, "combined", CN),
            "CF-Eval": cf.get("cnnspot_combined", NA),
        },
        # CommunityForensics-384 off-the-shelf (E6)
        {
            "detector": "CommunityForensics-384",
            "train": "broad (off-the-shelf)",
            "cue": "broad-generator",
            "SynthCLIC": _cell(e6, SC, "mAP_by_generator"),
            "SynthBuster+": _cell(e6, "synthbuster_plus", "mAP_by_generator"),
            "CNNSpot": _cell(e6, CN, "mAP_by_generator"),
            "CF-Eval": _cell(e6, "community_forensics_eval", "mAP_by_generator"),
        },
    ]
    df = pd.DataFrame(rows)

    (EXPORT / "tables").mkdir(parents=True, exist_ok=True)
    (EXPORT / "tex").mkdir(parents=True, exist_ok=True)
    df.to_csv(EXPORT / "tables/table_g_detector_mAP.csv", index=False)

    # Custom TeX layout for this table only (fixed grouping with multirow).
    clip_order = ["SynthCLIC", "SynthBuster+", "CNNSpot", "Combined"]
    forensic_order = ["SynthCLIC", "SynthBuster+", "CNNSpot/ProGAN", "Combined"]
    clip_rows = {r["train"]: r for _, r in df[df["detector"] == "CLIP linear probe"].iterrows()}
    forensic_rows = {r["train"]: r for _, r in df[df["detector"] == "Forensic CNN"].iterrows()}
    cf_row = df[df["detector"] == "CommunityForensics-384"].iloc[0]

    tex_lines = [
        "\\begin{table}[!ht]",
        "\\centering",
        "\\scriptsize",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Detector & Train & SynthCLIC & SynthBuster+ & CNNSpot & CF-Eval \\\\",
        "\\midrule",
        "\\multirow{4}{*}{CLIP (linear head)}",
    ]
    for t in clip_order:
        r = clip_rows[t]
        tex_lines.append(
            _row_tex(t, [r["SynthCLIC"], r["SynthBuster+"], r["CNNSpot"], r["CF-Eval"]])
        )
    tex_lines.append("\\midrule")
    tex_lines.append("\\multirow{4}{*}{Forensic CNN}")
    for t in forensic_order:
        r = forensic_rows[t]
        t_show = "CNNSpot" if t == "CNNSpot/ProGAN" else t
        tex_lines.append(
            _row_tex(t_show, [r["SynthCLIC"], r["SynthBuster+"], r["CNNSpot"], r["CF-Eval"]])
        )
    tex_lines.extend(
        [
            "\\midrule",
            (
                "CommunityForensics-384 & broad (off-the-shelf) & "
                f"{_fmt_tex_cell(cf_row['SynthCLIC'])} & {_fmt_tex_cell(cf_row['SynthBuster+'])} & "
                f"{_fmt_tex_cell(cf_row['CNNSpot'])} & {_fmt_tex_cell(cf_row['CF-Eval'])} \\\\"
            ),
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Detectors trained on different datasets are evaluated on \\synthclic{}, "
            "\\synthbp{}, \\cnnspot{}, and the CommunityForensics-Eval (CF-Eval) dataset. "
            "Shown are per-generator mAP values.}",
            "\\label{tab:detector_comparison}",
            "\\end{table}",
            "",
        ]
    )
    tex = "\n".join(tex_lines)
    (EXPORT / "tex/g_detector_mAP.tex").write_text(tex)

    print(df.to_string(index=False))
    print("\nwrote reproduction/revision_export/tables/table_g_detector_mAP.csv + tex/g_detector_mAP.tex")


if __name__ == "__main__":
    main()
