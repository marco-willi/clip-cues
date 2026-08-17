"""Canonical detection metrics for the revision — Convention A (the paper's mAP).

The initial submission defines mAP as a **per-generator mean Average Precision**
(initial_submission.tex:364,368): *"The performance of detecting samples from a specific
generative model is evaluated on its synthetic images and the corresponding real images
(balanced classes) … We then calculate mAP over all generators."* Concretely, for each
generator we compute AP over {that generator's fakes ∪ ALL reals in the frame}, then average the
per-generator APs. This reproduces the original implementation
``archive/detection_via_clip/analyse.py::calculate_metrics`` and matches the CommunityForensics
convention already used by E6/E7 (``export_e6_tables.per_generator_vs_all_reals``,
``export_community_eval_tables.per_generator``).

This is **not** the same as ``clip_cues.concept_modeling.metrics.SimpleMetrics``, whose "mAP" is a
single *pooled* binary AveragePrecision over the whole split — a prevalence-sensitive quantity
that runs systematically higher (see ``.claude/plans/PLAN_METRIC_METHODOLOGY_AUDIT.md``). Use the
helpers here for any number compared against the published paper.

All experiments should compute mAP through this module so the definition stays identical
everywhere. Inputs are a tidy per-image predictions frame with binary ``label`` (0=real,
1=synthetic), a continuous ``score`` (higher ⇒ more synthetic), and a ``source`` column naming the
generator.

**Two real-pairing rules**, both from the original paper (and both reproduced here):

- ``real_pairing="shared"`` — each generator's fakes vs **all reals in the frame**. Used for
  SynthCLIC / SynthBuster+, where every generator is paired against one shared real set
  (``analyse.py::calculate_metrics``). Default.
- ``real_pairing="matched"`` — each generator's fakes vs **reals of the same source**. Used for
  CNNSpot, whose ``source`` tags both halves of each generator's own real/fake pair; real-only
  sources (imagenet/laion/seeingdark) are consequently ignored
  (``analyse.py::calculate_metrics_for_cnnspot``).

Use ``pairing_for_dataset(name)`` to pick the right rule from a dataset label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# Dataset label -> real-pairing rule. CNNSpot is source-matched; everything else shares one real
# set. Matching is loose (substring) so backbone-suffixed labels like "cnnspot_clip_b16" still map.
_MATCHED_DATASETS = ("cnnspot",)


def pairing_for_dataset(dataset: str) -> str:
    """Return the paper's real-pairing rule (``"matched"``/``"shared"``) for a dataset label."""
    d = dataset.lower()
    return "matched" if any(m in d for m in _MATCHED_DATASETS) else "shared"


def binary_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average Precision for one binary group; ``nan`` if a class is absent.

    AP is undefined without both a positive and a negative example, so degenerate groups (e.g. a
    cross-dataset eval where a generator has no paired reals) return ``nan`` rather than raising.
    """
    y = np.asarray(y_true)
    s = np.asarray(y_score)
    if not (0 < int(y.sum()) < len(y)):  # need at least one real and one fake
        return float("nan")
    return float(average_precision_score(y, s))


def per_generator_ap(
    predictions: pd.DataFrame,
    *,
    label: str = "label",
    score: str = "score",
    source: str = "source",
    real_pairing: str = "shared",
    passthrough: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Per-generator AP table (Convention A).

    Args:
        predictions: per-image frame with ``label`` (0/1), ``score``, and ``source`` columns.
        label / score / source: column names (override for frames using other names, e.g. the E7
            parquet which already uses these defaults).
        real_pairing: ``"shared"`` (each generator's fakes ∪ all reals; SynthCLIC/SynthBuster) or
            ``"matched"`` (each generator's fakes ∪ same-source reals; CNNSpot). See module docstring.
        passthrough: extra per-generator-constant columns to carry into the output (e.g.
            ``("architecture",)``); the first value within each source group is taken.

    Returns:
        DataFrame ``[generator, n_fake, n_real, ap, *passthrough]`` sorted by ``ap`` ascending
        (worst-transferring generators first).
    """
    if real_pairing not in ("shared", "matched"):
        raise ValueError(f"real_pairing must be 'shared' or 'matched', got {real_pairing!r}")
    all_reals = predictions[predictions[label] == 0]
    rows: list[dict] = []
    for gen, g in predictions[predictions[label] == 1].groupby(source):
        reals = all_reals if real_pairing == "shared" else all_reals[all_reals[source] == gen]
        sub = pd.concat([g, reals])
        row = {
            "generator": gen,
            "n_fake": int(len(g)),
            "n_real": int(len(reals)),
            "ap": binary_ap(sub[label].to_numpy(), sub[score].to_numpy()),
        }
        for c in passthrough:
            row[c] = g[c].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("ap").reset_index(drop=True)


def per_generator_accuracy(
    predictions: pd.DataFrame,
    *,
    label: str = "label",
    score: str = "score",
    source: str = "source",
    real_pairing: str = "shared",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Per-generator accuracy at a fixed decision threshold, on the **same groups** as AP.

    AP is threshold-free, so a table reporting ACC beside it must declare a threshold. The default
    ``0.5`` on the sigmoid is ``z > 0`` on the logit — the head's own decision rule, and the only
    threshold available without tuning on the evaluation population.

    The grouping is deliberately identical to :func:`per_generator_ap`: with
    ``real_pairing="shared"`` every generator's group contains the one shared real pool, so the
    same real-side errors recur in every row of that block. That is a property of the paper's
    metric convention, not of this function, and reporting ACC on any other grouping would make
    the two columns describe different populations.

    Returns:
        DataFrame ``[generator, n_fake, n_real, acc]`` sorted by generator.
    """
    if real_pairing not in ("shared", "matched"):
        raise ValueError(f"real_pairing must be 'shared' or 'matched', got {real_pairing!r}")
    all_reals = predictions[predictions[label] == 0]
    rows: list[dict] = []
    for gen, g in predictions[predictions[label] == 1].groupby(source):
        reals = all_reals if real_pairing == "shared" else all_reals[all_reals[source] == gen]
        sub = pd.concat([g, reals])
        pred = (sub[score].to_numpy() > threshold).astype(int)
        rows.append(
            {
                "generator": gen,
                "n_fake": int(len(g)),
                "n_real": int(len(reals)),
                "acc": float((pred == sub[label].to_numpy()).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("generator").reset_index(drop=True)


def per_generator_map(
    predictions: pd.DataFrame,
    *,
    label: str = "label",
    score: str = "score",
    source: str = "source",
    real_pairing: str = "shared",
) -> float:
    """Mean per-generator AP — the paper's headline mAP.

    Equivalent to averaging the ``average_precision`` values produced by the original
    ``calculate_metrics`` / ``calculate_metrics_for_cnnspot``. Generators with an undefined AP
    (``nan``) are skipped, matching pandas' default ``Series.mean`` behaviour; in the paper's
    balanced setting every generator is well-defined so no skipping occurs.
    """
    tbl = per_generator_ap(
        predictions, label=label, score=score, source=source, real_pairing=real_pairing
    )
    return float(tbl["ap"].mean())


def map_by_architecture(
    predictions: pd.DataFrame,
    *,
    label: str = "label",
    score: str = "score",
    source: str = "source",
    architecture: str = "architecture",
    real_pairing: str = "shared",
) -> pd.DataFrame:
    """mAP grouped by architecture = mean of per-generator APs within each architecture.

    Returns DataFrame ``[architecture, mAP, n_generators]`` sorted by mAP ascending. Mirrors the
    CF-Eval architecture breakdown in ``export_community_eval_tables`` / ``export_e6_tables``.
    """
    tbl = per_generator_ap(
        predictions,
        label=label,
        score=score,
        source=source,
        real_pairing=real_pairing,
        passthrough=(architecture,),
    )
    out = (
        tbl.groupby(architecture)
        .agg(mAP=("ap", "mean"), n_generators=("ap", "size"))
        .reset_index()
        .sort_values("mAP")
        .reset_index(drop=True)
    )
    return out


def detection_metrics(
    predictions: pd.DataFrame,
    *,
    label: str = "label",
    score: str = "score",
    source: str = "source",
    real_pairing: str = "shared",
) -> dict:
    """Convenience bundle: per-generator mAP (Convention A) plus pooled AP for reference.

    Returns a dict with ``mAP`` (the paper's metric), ``pooled_ap`` (the old SimpleMetrics
    quantity, kept only so the two can be reported side by side), ``auroc`` (pooled),
    ``real_pairing``, ``n_generators``, ``n_real``, and the ``per_generator`` table. Reporting both
    mAP and pooled AP makes the convention explicit and surfaces the gap rather than hiding it.
    """
    tbl = per_generator_ap(
        predictions, label=label, score=score, source=source, real_pairing=real_pairing
    )
    y = predictions[label].to_numpy()
    s = predictions[score].to_numpy()
    return {
        "mAP": float(tbl["ap"].mean()),
        "pooled_ap": binary_ap(y, s),
        "real_pairing": real_pairing,
        "n_generators": int(tbl["ap"].notna().sum()),
        "n_real": int((predictions[label] == 0).sum()),
        "per_generator": tbl,
    }
