# F1 — Canonical 1024-d detector stability (spec E1)

**Status:** ✅ done · **Driver:** `scripts/finalexp/run_f1_canonical_stability.py` ·
**Artifacts:** [`artifacts/`](artifacts/) (SynthCLIC), [`artifacts_cnnspot/`](artifacts_cnnspot/) ·
**Runs:** `runs/seed{123..127}`, `runs/cnnspot_seed{123..127}`

```bash
uv run python scripts/finalexp/run_f1_canonical_stability.py                    # SynthCLIC
uv run python scripts/finalexp/run_f1_canonical_stability.py --dataset cnnspot  # for F5
```

**Inputs** (snapshot ids, sha256 pinned in each run's `run_meta.json`): `pooler/synthclic`,
`cue_scores/synthclic__antonyms`, `vocab/antonyms`, `reference/e3_seed123`.

## Why

To interpret the *actual* canonical detector, its decision direction has to be reasonably stable
across refits. The manuscript previously side-stepped this by noting that an auxiliary logistic fit
is convex and reproducible — but that argues about a *different model*. F1 tests the canonical
detector directly, over 5 seeds with everything else held fixed.

## Regression anchor — passed

The seed-123 run reproduces the persisted `run_linear_probe.py` result that stands behind the
manuscript's Table A row, so the shared trainer is the same estimator:

| | mAP | AUROC |
|---|---|---|
| anchor (`reference/e3_seed123`) | 0.9239 | 0.9227 |
| F1 seed 123 | 0.9230 | 0.9209 |
| Δ | 0.0009 | 0.0018 |

Tolerance 0.005 → **PASS**. Residual difference is dataloader/RNG detail, not a different model.

## Results (SynthCLIC test, 5 seeds, 10 seed pairs)

Detection: **AUROC 0.9212 [0.9182, 0.9227]**, mAP 0.9228 [0.9185, 0.9246].

| quantity | mean | range |
|---|---|---|
| **direction, Σ-metric cosine** (primary) | **0.9891** | [0.9828, 0.9930] |
| direction, raw cosine (secondary) | 0.9788 | [0.9572, 0.9894] |
| test-logit Spearman | 0.9864 | [0.9794, 0.9911] |
| decision agreement | 0.9823 | [0.9790, 0.9869] |
| **168-cue profile Spearman** | **0.9913** | [0.9841, 0.9967] |
| top-50 extreme overlap (Jaccard) | 0.6951 | [0.5873, 0.8182] |
| bottom-50 extreme overlap | 0.7713 | [0.6949, 0.8182] |

CNNSpot (trained for F5): mAP 0.9755 [0.9751, 0.9759], AUROC 0.9524 [0.9465, 0.9604].

## Reading

**The canonical decision direction and its aggregate cue profile are stable across refits.** That
is the licence the spec asked for, and it is a stronger statement than the convexity argument it
replaces because it is about the deployed model class rather than an auxiliary one.

Two qualifications belong next to it:

1. **The extreme images are markedly less stable than the direction** (top-50 Jaccard 0.70 vs Σ-cos
   0.99). Seeds that agree on the boundary to three decimal places still disagree about roughly a
   third of the most extreme images. This is a direct caveat for F5's montage — the figure is an
   illustration of a stable direction, not a stable image list — and it is why F5 reports its own
   seed overlap.
2. **Here the metric caution did not bite.** Σ-cosine (0.989) and raw cosine (0.979) agree closely,
   unlike the N21 case where they diverged sharply (0.07 vs 0.938). The Σ metric is still reported
   first because it is the one that means "same decision function", but on this comparison the
   conclusion is metric-robust.

## Files

| file | contents |
|---|---|
| `artifacts/summary.json` | headline numbers, recipe, anchor check, per-metric seed-pair summaries |
| `artifacts/stability.csv` | all 10 seed pairs × every metric |
| `artifacts/per_seed_metrics.csv` | per-seed mAP / AUROC / best val-CE |
| `runs/seed*/weights.npz` | decision direction + bias |
| `runs/seed*/logits_test.csv` | per-image test logits (consumed by F2, F3, F5, F7) |
| `runs/seed*/cue_profile.csv` | per-cue pooled and within-class correlations |
| `runs/*/run_meta.json` | script, argv, git commit, input shas, host, duration |
