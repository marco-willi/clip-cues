# F3 — Projected 768-d analysis head, matched recipe (spec E3)

**Status:** ✅ done · **Driver:** `scripts/finalexp/run_f3_projected_head.py` ·
**Artifacts:** [`artifacts/`](artifacts/) · **Runs:** `runs/seed{123..127}`

```bash
uv run python scripts/finalexp/run_f3_projected_head.py
```

**Inputs:** `projected/synthclic`, `pooler/synthclic`, `cue_scores/synthclic__antonyms`,
`vocab/antonyms`, `projection/wp_l14_336`.

## Why

Text directions live in the 768-d shared space, so quantities like `cos(w, v_q)` need a classifier
direction *there*. That role was previously played by a standardized, CV-tuned scikit-learn probe
(`P768t`) — a third training recipe, requiring coefficient back-transformation machinery in the
appendix. F3 replaces it with the same head, optimizer, loss and schedule as F1, so `D_h` and `D_e`
differ **only** by the projection.

Features are `e = Wp h`, derived from the very same cached pooler frame F1 uses (the
"both-sides-derived rule" of the frozen external-validation protocol, and what E11b did), then
rescaled by one global scalar — see the deviation note in the
[task README](../README.md#one-deviation-from-the-spec-and-why). Scale factor here: **1.750**.

## Results (SynthCLIC test, 5 seeds)

| quantity | value | prior record |
|---|---|---|
| `D_h` AUROC | 0.9212 | — |
| **`D_e` AUROC** | **0.8925** [0.8871, 0.8966] | P768t 0.885 |
| **projection cost `D_h − D_e`** | **+0.0286 [+0.0205, +0.0366]** | N2c +0.021 [+0.013, +0.030] |
| `D_h`~`D_e` test-logit Spearman | 0.9283 | — |
| `D_h`~`D_e` 168-cue profile Spearman | **0.9870** | E12 proxy agreement ρ ≈ 0.95 pooled |

Paired cluster bootstrap by source photo (`image_id`), 2,000 draws, seed 0, percentile CIs.

`D_e` seed stability: Σ-cos and raw cosine, logit Spearman and cue-profile Spearman are in
`artifacts/stability.csv`.

## Reading

**Projection into CLIP's shared image–text space costs a small but real amount of detection
performance and preserves the detector's cue profile.** That is exactly the claim the spec wanted
supported, now measured between two models that differ in nothing but the projection.

The cost, **+0.029 [+0.021, +0.037]**, is slightly larger than the matched-tuning proxy estimate
(N2c, +0.021 [+0.013, +0.030]) and the CIs overlap only partially. The likely reason is the recipe
change itself: N2c compared two CV-tuned probes, each free to pick its own regularization strength,
while F3 holds `weight_decay` fixed by construction. A fixed penalty suits the higher-variance
1024-d space slightly better than the 768-d one. This is a *difference in estimator*, not a
contradiction — both say the projection cost is small, positive, and reliably non-zero.

Cue-profile agreement (ρ 0.987) is **higher** than the proxy-based figure, which supports using
`D_e` as the analysis surrogate wherever shared image–text coordinates are required.

## Files

| file | contents |
|---|---|
| `artifacts/summary.json` | AUROC both spaces, projection cost + CI, agreement, seed stability, the space record (scale factor) |
| `artifacts/projection_cost.csv` | per-seed `D_h` vs `D_e` with bootstrap CIs |
| `artifacts/stability.csv` | `D_e` seed pairs × every metric |
| `runs/seed*/` | weights, test logits, cue profile, `run_meta.json` |
