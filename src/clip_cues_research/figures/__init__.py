"""Revision figure-reproduction code (revision-only; never back-ported to clip_cues).

Re-implements the paper figures for the revision, replacing the non-identifiable k=8-orthogonal-axis
figures with the canonical **k=1/logistic linear-probe** equivalents (see docs/REVISION_CODE_PLAN.md +
docs/GPT_PLAN2.md). Each module exposes reusable functions; thin ``scripts/plot/plot_*`` drivers call
them.

Modules:
    score_distributions  — Fig 5: deterministic real-vs-synthetic decision-score densities.
    cue_profile          — Fig 7: bootstrap-stable cue profile.
    concept_explanation  — Fig 1: content-controlled concept local explanation (switchable image).
    clipiqa              — Appendix: CLIP-IQA perceptual axes (perceived quality, not pixels).
    head_decomp          — Appendix: per-head decomposition heatmap + direct-vs-causal ablation.
    head_concepts        — Appendix: associating heads with nearest antonym concepts (Gandelsman naming).
    paired_cue_shift     — Content-controlled cue shifts (SynthCLIC real↔synthetic pairing).
    linear_probe_samples — Top/bottom linear-probe decision-score montages per (train, eval) dataset.
(Fig 6 montages: scripts/plot/plot_direction_samples.py --method logreg.)
"""


__all__ = [
    "probe_scores_from_cache",
    "score_distribution_figure",
    "linear_probe_sample_figure",
    "HFImageIndex",
]
