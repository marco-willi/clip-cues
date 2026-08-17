# F4 — 168-cue restricted-information probe (spec E4)

**Status:** ✅ done · **Driver:** `scripts/finalexp/run_f4_cue_capacity.py` ·
**Artifacts:** [`artifacts/`](artifacts/) · **Runs:** `runs/{antonyms,optimized_v2_canon}_seed{123..127}`

```bash
uv run python scripts/finalexp/run_f4_cue_capacity.py
```

**Inputs:** `cue_scores/synthclic__{antonyms,optimized_v2_canon}`, `vocab/*`, `projected/synthclic`.

## Why

The question is *how much of the linearly accessible synthetic-image signal the named vocabulary can
carry* — not "is this another CLIP detector". Per the spec this model is reframed as a
**restricted-information probe**: same head, same recipe, same images as `D_e`; only the input is
restricted to named cue scores `c_j = ⟨e/‖e‖, v_j⟩`. The comparison against F3's unrestricted 768-d
`D_e` therefore isolates what the vocabulary cannot express.

## Results (SynthCLIC test, 5 seeds, vs `D_e` = 0.8925)

| vocabulary | cues | AUROC | ΔAUROC vs `D_e` | excess recovery |
|---|---|---|---|---|
| **antonyms (published)** | 168 | 0.8336 | **−0.0589 [−0.0749, −0.0429]** | **0.850** |
| optimized_v2_canon | 128 | 0.8374 | −0.0551 [−0.0687, −0.0415] | 0.860 |

Paired cluster bootstrap by source photo, 2,000 draws, seed 0.
Prior record (N5c, CV-tuned probes): ant168t −0.036 [−0.050, −0.022], recovery 90.5%.

## Reading

**The named vocabulary recovers ~85% of the unrestricted probe's excess AUROC, and the remaining
~15% is a reliable deficit.** The direction and magnitude reproduce N5c; the deficit here is
somewhat larger (−0.059 vs −0.036) and the recovery correspondingly lower (0.85 vs 0.905).

The most likely cause is the same estimator difference noted in [F3](../F3-projected-head/README.md):
N5c compared CV-tuned models, each able to pick its own regularization, whereas F4 fixes
`weight_decay` across a 168-d and a 768-d input. A fixed penalty is relatively harsher on the
lower-dimensional restricted probe. So the F4 figure should be read as a *conservative* estimate of
vocabulary coverage under a deliberately uniform recipe, not as a downward revision of N5c.

The two vocabularies are statistically indistinguishable here, with the **128-cue optimized set
marginally ahead of the 168-cue published set** despite having fewer cues — consistent with E9/N18's
finding that compactness transfers at least as well as coverage.

## A methodological warning worth keeping

Before the scale fix (see the [task README](../README.md#one-deviation-from-the-spec-and-why)), this
experiment returned **cue AUROC 0.800 versus `D_e` 0.722** — the restricted probe apparently beating
the unrestricted probe it is a strict linear subspace of. That is impossible, and it is what exposed
the under-training caused by holding `weight_decay` fixed across spaces whose feature norms differ
by ~60× (cue scores 0.55, pooler 32.95). The inversion is a useful sanity check to keep: **if a
restricted probe ever beats its own superset, suspect optimization before interpretation.**

## Files

| file | contents |
|---|---|
| `artifacts/summary.json` | per-vocabulary AUROC, ΔAUROC + CI, recovery, framing and feature-construction notes |
| `artifacts/capacity.csv` | per-seed, per-vocabulary rows with bootstrap CIs |
| `runs/*/` | weights, test logits, `run_meta.json` |
