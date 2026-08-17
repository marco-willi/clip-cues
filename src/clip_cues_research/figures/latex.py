"""The house LaTeX table emitter — one envelope, one numeric convention, every exported table.

The manuscript's tables come from two places: :mod:`scripts.export.rebuild_export_local` (the nine
E-experiment tables in ``reproduction/revision_export/tex/``) and
:mod:`clip_cues_research.figures.compact_panels` (the two F-experiment tables in
``reproduction/experiments/figures/tables/``). They used to disagree — the second emitted bare
``DataFrame.to_latex`` output, which puts the caption *before* the tabular, ships no float
placement, and passes snake_case column names straight through (a bare ``cue_profile`` is a
subscript in text mode, so the file did not compile). Both now render through :func:`paper_table`,
so a fix to the envelope reaches every table at once.

The envelope matches ``docs/initial_submission.tex`` (e.g. ``tab:clip:test_results``)::

    \\begin{table}[!ht]
    \\centering
    \\scriptsize
    \\begin{tabular}{...}
    ...
    \\end{tabular}
    \\caption{...}
    \\label{tab:rev:...}
    \\end{table}

Two rules the callers own, because ``to_latex`` escapes nothing here (that is what lets ``$D_h$``
through):

  * **Header and row labels are display strings**, not DataFrame column names. Rename before
    emitting — ``_`` and ``%`` in text mode are LaTeX errors, not typos.
  * **Signed numbers go through** :func:`signed` / :func:`signed_ci`. A stringified ``-0.029`` is
    typeset as a hyphen, not a minus. Only the LaTeX path formats; the CSVs written next to the
    figures stay raw floats.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

#: Em-dash cell for "no value here" — the convention in ``reproduction/revision_export/tex/e3_clip_backbones.tex``.
MISSING = "--"


def escape_text(text: str) -> str:
    """Make a free-text string safe for LaTeX text mode.

    Captions interpolate strings that came out of a ``summary.json`` — cluster names, vocabulary
    names, dataset ids. Those are written for machines and routinely contain ``_``: the cluster key
    ``"source photo (image_id)"`` is a subscript-in-text-mode compile error the moment it lands in a
    caption. Anything not authored as LaTeX goes through here first.
    """
    for ch in "\\&%$#_{}":
        text = text.replace(ch, f"\\{ch}")
    return text.replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}")


def signed(value: float, decimals: int = 3) -> str:
    """A signed number as a LaTeX math cell: ``-0.029`` -> ``$-0.029$`` (a minus, not a hyphen)."""
    return f"${value:+.{decimals}f}$"


def signed_ci(value: float, lo: float, hi: float, decimals: int = 3) -> str:
    """A signed estimate with its interval: ``$-0.029$ [$-0.037$, $-0.021$]``."""
    return f"{signed(value, decimals)} [{signed(lo, decimals)}, {signed(hi, decimals)}]"


def paper_table(
    df: pd.DataFrame,
    caption: str,
    label: str,
    *,
    position: str = "!ht",
    index: bool = False,
    float_format: Callable[[float], str] = lambda x: f"{x:.2f}",
    na_rep: str = MISSING,
    size: str = "scriptsize",
) -> str:
    """Render ``df`` as a paper-styled booktabs table (caption and label *after* the tabular).

    Args:
        df: display frame — column names and object cells are emitted verbatim, so they must
            already be LaTeX (see the module docstring).
        caption: caption text, LaTeX.
        label: ``\\label`` target, e.g. ``tab:rev:cascade``.
        position: float placement specifier.
        index: whether to emit the DataFrame index as a leading column.
        float_format: applied to numeric cells; defaults to the 2-decimal manuscript convention.
        na_rep: what a missing cell becomes.
        size: font size command, without the backslash.
    """
    tabular = df.to_latex(index=index, float_format=float_format, na_rep=na_rep)
    return (
        f"\\begin{{table}}[{position}]\n\\centering\n\\{size}\n"
        f"{tabular}\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n"
    )
