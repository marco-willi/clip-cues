# F6 — Cross-dataset projected heads + boundary decomposition (spec E6)

**Status:** ✅ done · **Driver:** `scripts/finalexp/run_f6_cross_dataset.py` ·
**Artifacts:** [`artifacts/`](artifacts/) · **Runs:** `runs/{synthclic,cnnspot,synthbuster-plus}_seed{123,124,125}`

```bash
uv run python scripts/finalexp/run_f6_cross_dataset.py
```

**Inputs:** `projected/{synthclic,cnnspot,synthbuster-plus}`, `vocab/antonyms`.
**Runtime:** heads 2–5 s each; the lasso decomposition **829 s** — the slowest step in F1–F7, and
the only one approaching the 20-minute Lambda escalation threshold.

## Why

E11b already computed the cross-dataset boundary cosines and the signed Δ decomposition in the
derived shared space — but with **CV-tuned scikit-learn probes** (C: SC 0.01, CNNSpot 0.001,
SB+ 0.01). F6 swaps **only the probe-fitting step**: the normals now come from the matched
`LinearHead(768)`. Everything downstream is E11's machinery, so there is no separate family of
"cross-dataset boundary probes" and no standardization/back-transformation step. N21 is the
pre-registered comparison target.

**Split discipline:** SynthBuster+ **train/val only**. The frozen protocol permits no further SB+
*test* reads; `finalexp.spaces` raises on any attempt.

## Boundary cosines

| pair | raw cosine | Σ-metric cosine | N21 (raw, CV-tuned) |
|---|---|---|---|
| synthclic ~ cnnspot | **−0.061** | **−0.207** | −0.102 |
| synthclic ~ synthbuster-plus | **+0.139** | **+0.311** | +0.161 |
| cnnspot ~ synthbuster-plus | −0.044 | **+0.353** | +0.082 |

Seed stability of each dataset's normal (Σ-cos, mean over 3 seed pairs): synthclic 0.997,
cnnspot 0.9995, synthbuster-plus 0.989.

**The first two replicate N21 in sign and rough magnitude.** The third flips sign in the raw metric
(−0.044 vs +0.082) — but both values sit near zero, so this is two estimators agreeing that the
boundaries are *approximately orthogonal*, not a genuine disagreement about direction.

**The Σ-metric sharpens the picture, and this is the more informative reading.** Where the data
actually lie, the GAN-era boundary is *anti*-aligned with both diffusion-era boundaries (−0.21 vs
SynthCLIC) while the two diffusion-era boundaries agree with each other (+0.31, +0.35). The raw
cosines compress all three toward zero and lose that structure — the same metric caution that F1/F2
report, showing up again at the level of whole datasets.

## Signed Δ decomposition — what CNNSpot weights toward "synthetic" relative to SynthCLIC

Data-weighted lasso of `Δ = ŵ(cnnspot) − ŵ(synthclic)` onto the canonical 168 signed cue axes, on
the union of both train sets. Knee: **148 axes**, val score-R² **0.762**, cos coverage 0.488. All
axes below at bootstrap selection frequency 1.00.

| toward CNNSpot-synthetic (Δ > 0) | toward SynthCLIC-synthetic (Δ < 0) |
|---|---|
| color_bleeding +0.360, upscaler_artifacts +0.290, compression_artifacts +0.245, retouching +0.232, vignette_edit +0.200, lens_flare +0.196 | posterization −0.393, pinhole_camera_cues −0.317, chromatic_aberration −0.252, film_grain −0.229, proportion_realism −0.222, depth_layering −0.219, color_harmony −0.209 |

**This replicates E11b's central claim under a different probe recipe.** The GAN-trained boundary is
a **compression/processing-artifact** detector; the diffusion-trained boundary keys on **optical and
analog-capture characteristics** plus composition. Four of N21's named CNNSpot-side axes
(color_bleeding, upscaler_artifacts, compression_artifacts, retouching) and three of its
SynthCLIC-side axes (posterization, film_grain, depth_layering) reappear with the same sign.

The emphasis on the SynthCLIC side shifts slightly: N21 described it as *aesthetics/provenance*
(iqa_naturalness, provenance_press/upload), whereas the matched normal leans more toward
*optical/analog capture* (pinhole camera, chromatic aberration, film grain, posterization). Both are
"properties of how a real photograph was made" rather than synthesis artifacts, so the
interpretation stands; the specific ranking is estimator-sensitive and should be quoted as such.

Fidelity is lower than N21's (score-R² 0.762 vs 0.96–0.97) and the knee sits at 148 of 168 axes —
so this is a *ranked additive attribution*, not a few-concept explanation, exactly as E11a
concluded.

## Identifiability caveat — read the cosines with this in mind

| dataset | val AUROC | val mAP |
|---|---|---|
| synthclic | 0.985 | 0.986 |
| **cnnspot** | **1.000** | **1.000** |
| synthbuster-plus | 0.996 | 0.997 |

CNNSpot is **perfectly separable** in this space (E11b used C=0.001 for this reason; E12/N25 saw the
same). With the matched recipe's fixed `weight_decay = 0.01` and no per-dataset tuning, the CNNSpot
probe is *less* regularized than E11b's, so its normal is if anything **more weakly identified** —
a large set of near-optimal directions all achieve AUROC 1.000. Its seed stability is high
(Σ-cos 0.9995), but that measures agreement among refits under one recipe, not identifiability.

Treat CNNSpot-involving cosines and the Δ axis list as **indicative**. The robust part is the
near-orthogonality itself, which cannot be produced by under-identification of sign alone, and which
now replicates across two independent probe recipes.

## Files

| file | contents |
|---|---|
| `artifacts/summary.json` | cosines, seed stability, separability caveat, full Δ decomposition |
| `artifacts/boundary_cosines.csv` | pairwise raw + Σ-metric cosines |
| `artifacts/delta_axes.csv` | top 25 Δ axes with coefficients, bootstrap frequency, side |
| `artifacts/delta_path.csv` | the full lasso path (alpha, nnz, score-R², AUROC, coverage) |
| `artifacts/seed_stability.csv` | per-dataset seed pairs |
| `runs/*/` | per-dataset, per-seed weights, metrics, `run_meta.json` |
