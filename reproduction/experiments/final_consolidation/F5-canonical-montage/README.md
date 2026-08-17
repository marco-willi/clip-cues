# F5 — Extreme-image ranking from the matched canonical probe (spec E5)

**Status:** ✅ done · **Driver:** `scripts/finalexp/export_f5_rankings.py` ·
**Artifacts:** [`artifacts/`](artifacts/) · **Snapshot ids:** `ranking/f5_synthclic`, `ranking/f5_cnnspot`

```bash
uv run python scripts/finalexp/run_f1_canonical_stability.py --dataset cnnspot   # once
uv run python scripts/finalexp/export_f5_rankings.py
```

## Why

The montage previously ranked images by an **auxiliary fixed-`C=1` single-direction logistic fit** —
a model that exists nowhere else in the paper and has to be explained in the caption. It is deleted.
Images are now ranked by the detector's own logit `z = wᵀh + b`; since `b` is constant, ranking by
`z` is ranking by projection onto `w`.

Per the 2026-08-08 decision the ranking detector is the **matched** probe (F1's primary seed), not
the published augmented checkpoint: nothing is published yet, so the figure should show the detector
the rest of the paper reports. F7 records how closely the two agree.

**Caption:** *Images with the highest and lowest logits of the evaluated canonical CLIP detector.*

## Results

| dataset | n (test) | logit range | top-50 seed overlap | bottom-50 seed overlap |
|---|---|---|---|---|
| synthclic | 2,140 | [−3.47, +4.93] | **0.726** | 0.763 |
| cnnspot | 4,000 | [−3.26, +3.28] | **0.662** | 0.613 |

Poles are clean in both: the top 8 are all label = 1 (synthetic), the bottom 8 all label = 0 (real).
SynthCLIC's synthetic pole is dominated by **FLUX.1-dev / FLUX.1-schnell / SD3-medium**; its real
pole is entirely **clic2020**.

> **Reading CNNSpot's `source` column.** CNNSpot names *real* images by the generator group they are
> paired into, so `progan` / `stylegan` / `stylegan2` appear in the **real** pole. That is the
> dataset's pairing convention, not a mislabelled image — check the `label` column, which is 0
> throughout the bottom pole.

## The caveat that belongs next to the figure

**The extreme images are much less stable across seeds than the direction that ranks them.** F1
found the decision direction stable at Σ-cos 0.989 and its cue profile at ρ 0.991 — but the top-50
image sets agree at only **0.73** (SynthCLIC) and **0.66** (CNNSpot) between seeds.

So the montage should be read as *an illustration of a stable direction*, not as a claim about
these particular images. Two refits that agree on the boundary to three decimal places still
disagree about roughly a third of the images at its extremes. Stating this is what makes the figure
honest; it costs nothing, because the figure's purpose is qualitative.

## Snapshot note

Image pixels cannot be checksummed into `experiments/data/` (far too large — see
[`../../data/EXCLUDED.md`](../../data/EXCLUDED.md)). Instead the **ranking** is registered as a
snapshot artifact with its own sha256, so the montage is reproducible from `ranked_scores_*.csv`
plus the HF dataset id and revision.

## Files

| file | contents |
|---|---|
| `artifacts/summary.json` | per-dataset poles, logit ranges, seed-overlap detail |
| `artifacts/ranked_scores_{synthclic,cnnspot}.csv` | full ranking: rank, image_id, source, label, logit |
| `experiments/data/rankings/f5_*.csv` | the same rankings, manifest-registered |

**Rendering the montage** (needs the HF image cache, `HF_HOME=data/hf_cache`):
`scripts/plot/plot_linear_probe_samples.py` produces the figure; point it at the matched head rather
than `data/checkpoints/` to match this ranking.
