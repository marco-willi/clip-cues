"""The two F-experiment LaTeX tables (`tab:rev:cascade`, `tab:rev:stability`).

These files are pasted straight into the manuscript, so a text-mode `_` in a header or a caption is
a build break in the paper repo rather than a cosmetic slip — that is exactly how the shipped
`stability-summary.tex` came to contain a bare `cue_profile`. There is no LaTeX toolchain in the
devcontainer, so the compile is approximated: the checks below are the specific text-mode errors
`to_latex` output is prone to, plus the house envelope from `reproduction/revision_export/tex/`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from clip_cues_research.figures.compact_panels import F1, F2, F3, F4, latex_tables
from clip_cues_research.figures.latex import escape_text, paper_table

SHIPPED = Path("reproduction/experiments/figures/tables")

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in (F1, F2, F3, F4)),
    reason="F-experiment summaries not built in this checkout",
)

#: Characters that are commands, not text, outside maths (`&` is excluded — in a tabular it is the
#: column separator). `\`-escaped ones are stripped before the check.
TEXT_MODE_SPECIALS = "_%#"


def strip_math(tex: str) -> str:
    """Everything outside `$...$`, with escaped specials removed — i.e. the text-mode content."""
    outside = re.sub(r"\$[^$]*\$", "", tex)
    return re.sub(r"\\.", "", outside)  # a `\x` pair is an escape or a command, never bare text


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> dict[str, str]:
    """Freshly emitted tables, written to a scratch tree rather than over the shipped artifact."""
    paths = latex_tables(root=tmp_path_factory.mktemp("figures"))
    return {k: p.read_text() for k, p in paths.items()}


def test_shipped_artifact_is_current(tables):
    """`reproduction/experiments/figures/tables/` must be what the builder emits today, not an older run."""
    for name, stem in (
        ("cascade", "cascade-information-restriction"),
        ("stability", "stability-summary"),
    ):
        shipped = SHIPPED / f"{stem}.tex"
        assert shipped.read_text() == tables[name], (
            f"{shipped} is stale — rerun `make tables-compact`"
        )


@pytest.mark.parametrize("name", ["cascade", "stability"])
def test_no_bare_specials_in_text_mode(tables, name):
    """The compile blocker: `cue_profile` in a header, `image_id` in a caption."""
    leftover = [c for c in TEXT_MODE_SPECIALS if c in strip_math(tables[name])]
    assert not leftover, f"{name}: unescaped {leftover} outside maths"


@pytest.mark.parametrize("name", ["cascade", "stability"])
def test_math_delimiters_balanced(tables, name):
    assert tables[name].count("$") % 2 == 0


@pytest.mark.parametrize("name", ["cascade", "stability"])
def test_house_envelope(tables, name):
    """Matches `reproduction/revision_export/tex/*.tex`: float placement, centring, caption AFTER the tabular."""
    tex = tables[name]
    assert tex.startswith("\\begin{table}[!ht]\n\\centering\n\\scriptsize\n\\begin{tabular}")
    assert tex.endswith("\\end{table}\n")
    assert tex.index("\\end{tabular}") < tex.index("\\caption{") < tex.index("\\label{")
    for rule in ("\\toprule", "\\midrule", "\\bottomrule"):
        assert rule in tex


def test_cascade_deltas_are_math_signed(tables):
    """A stringified `-0.029` typesets as a hyphen; every signed cell must be maths."""
    body = tables["cascade"]
    assert "$-0.029$ [$-0.037$, $-0.021$]" in body
    assert "$-0.059$ [$-0.075$, $-0.043$]" in body
    assert re.search(r"(?<![$+-])\s-0\.\d", body) is None, "bare hyphen-minus in a numeric cell"
    assert "& -- \\\\" in body, "the first row has no previous row: it must read '--', not blank"


def test_factorized_head_is_typeset_as_maths(tables):
    """The manuscript writes the factorized head as `$k{=}8$`; a bare `k=8` is upright text."""
    stab = tables["stability"]
    assert stab.count("$k{=}8$") == 3  # two row labels and the caption
    assert "$k{=}1$" in stab
    assert re.search(r"(?<!\$)k=\d", stab) is None


def test_headers_name_their_statistic(tables):
    header = tables["stability"].split("\\toprule\n")[1].split("\\\\")[0]
    assert "cue_profile" not in header and "representation" not in header
    assert "Cue profile $\\rho$" in header


def test_captions_state_the_seed_and_ci_conventions(tables):
    assert "cluster bootstrap" in tables["cascade"]
    assert "relative to the row above" in tables["cascade"]
    assert "5 seed refits" in tables["stability"]


def test_source_frames_stay_raw(tables):
    """Display formatting is LaTeX-only — the CSVs written next to the figures keep raw floats."""
    from clip_cues_research.figures.compact_panels import cascade_table, stability_table

    casc, stab = cascade_table(), stability_table()
    assert pd.api.types.is_float_dtype(casc["auroc"])
    assert pd.api.types.is_float_dtype(casc["delta"]) and pd.isna(casc.loc[0, "delta"])
    # Raw labels stay mathtext-renderable for the retired plots: `$D_h$` is fine, `$k{=}8$` is not.
    assert not any("{=}" in r for r in stab["representation"])
    assert not any("{=}" in s for s in casc["stage"])


def test_escape_text_handles_summary_json_keys():
    assert escape_text("source photo (image_id)") == "source photo (image\\_id)"


def test_paper_table_keeps_the_bundle_convention():
    """The nine `reproduction/revision_export/tex/` tables are 2-decimal by default; only the F tables override."""
    df = pd.DataFrame({"Set": ["CNNSpot"], "mAP": [0.9312]})
    assert "0.93" in paper_table(df, "c", "tab:rev:x")
    assert "0.931" in paper_table(df, "c", "tab:rev:x", float_format=lambda x: f"{x:.3f}")
