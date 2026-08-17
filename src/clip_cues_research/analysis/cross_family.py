"""E4: cross-family generalization analysis (the 0.37 mAP failure).

Reviewer 1 wants the striking cross-family drop analyzed, not just reported. No new training — this
reuses existing artifacts:

    Part 1 — per-generator AP breakdown (which generators collapse?): from the E7 CommunityForensics
        per-image predictions (`results/e7_community_eval/predictions/*.parquet`), which carry
        ``label``/``score``/``source`` (=generator)/``architecture``.
    Part 2 — concept shift across training domains (do learned concepts differ?): the concept-model
        checkpoints `cm_antonyms_{synthclic,synthbuster,cnnspot,combined}.ckpt` each expose
        ``model.W_classifier.weight`` (1 x 168) — the per-concept importance the detector places on
        each antonym concept. Comparing these vectors across domains quantifies the shift.

Used by `scripts/run/run_cross_family_analysis.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from clip_cues_research.analysis.metrics import per_generator_ap as _canonical_per_generator_ap


# ── Part 1: per-generator AP ─────────────────────────────────────────────────────────────────
def per_generator_ap(
    predictions: pd.DataFrame,
    *,
    y_true: str = "y_true",
    y_score: str = "y_score",
    generator: str = "generator",
    real_pairing: str = "matched",
    passthrough: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Average Precision per generator family — thin wrapper over ``analysis.metrics``.

    Delegates to the canonical Convention-A helper so the metric definition lives in one place.
    Defaults to ``real_pairing="matched"`` because the E7 CommunityForensics parquet tags each
    generator's paired reals with the same ``source`` (1000 real + 1000 fake per source); use
    ``real_pairing="shared"`` for SynthCLIC/SynthBuster-style frames with one shared real set.

    Args:
        predictions: per-image frame with truth/score/generator columns.
        y_true / y_score / generator: column names (pass ``y_true="label", y_score="score",
            generator="source"`` for the E7 parquet).
        real_pairing: ``"matched"`` (default) or ``"shared"`` — see ``analysis.metrics``.
        passthrough: extra per-generator-constant columns to carry through (e.g. ``architecture``).

    Returns:
        DataFrame [generator, n_fake, n_real, ap, *passthrough] sorted by ap ascending (worst first).
    """
    return _canonical_per_generator_ap(
        predictions,
        label=y_true,
        score=y_score,
        source=generator,
        real_pairing=real_pairing,
        passthrough=passthrough,
    )


# ── Part 2: concept shift across training domains ────────────────────────────────────────────
def per_concept_importance(checkpoint_path: str) -> np.ndarray:
    """Signed per-concept detector importance ``W_classifier`` (shape (n_concepts,)) from a
    concept-model checkpoint. |value| = how much the concept drives the synthetic/real decision."""
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return ck["state_dict"]["model.W_classifier.weight"].squeeze(0).numpy()


def concept_shift(
    importance_by_domain: dict[str, np.ndarray], concept_names: list[str]
) -> pd.DataFrame:
    """Per-concept importance across training domains (normalised within domain) + spread.

    Args:
        importance_by_domain: domain -> signed per-concept importance vector (n_concepts,).
        concept_names: concept labels (length n_concepts).

    Returns:
        DataFrame [concept, <domain>..., importance_std] where each domain column is |W| normalised
        to its own max (so domains with different weight scales are comparable). ``importance_std``
        is the spread across domains (high ⇒ the concept's role shifts between training domains).
    """
    df = pd.DataFrame(index=pd.Index(concept_names, name="concept"))
    for dom, w in importance_by_domain.items():
        a = np.abs(np.asarray(w, dtype=float))
        df[dom] = a / a.max() if a.max() > 0 else a
    df["importance_std"] = df.std(axis=1)
    return df.reset_index()


def domain_concept_correlation(importance_by_domain: dict[str, np.ndarray]) -> pd.DataFrame:
    """Spearman rank correlation between domains' per-concept importance.

    Low correlation ⇒ the domains' detectors rely on different concepts (concept shift), which helps
    explain weak cross-family transfer.
    """
    m = pd.DataFrame(
        {d: np.abs(np.asarray(w, dtype=float)) for d, w in importance_by_domain.items()}
    )
    return m.corr(method="spearman")
