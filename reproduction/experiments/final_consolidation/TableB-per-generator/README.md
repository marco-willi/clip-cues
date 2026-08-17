# TableB — regenerating the appendix per-generator table (`tab:clip:test_results_detail`)

**Status:** ✅ computed · **Driver:** `scripts/finalexp/run_appendix_per_generator.py` ·
**Exporter:** `scripts/export/export_per_generator_table.py` ·
**Artifacts:** [`artifacts/`](artifacts/)

```bash
make finalexp-data WITH_CFEVAL=1 WITH_APPENDIX=1   # +610 MB: combined + full-CNNSpot frames, E3 predictions
make finalexp-tableb                                # regenerate, check, export
```

**Inputs** (snapshot ids, sha256 pinned in `artifacts/run_meta.json`): `e3pred/*` (11 cells),
`ckpt/linear_probe_combined_matched` + `pooler/cnnspot_full` (the 12th), and
`pooler/{synthclic,cnnspot,synthbuster-plus,combined}` for the recipe-equivalence refit.
**Runtime:** ~70 s, dominated by unpickling the 467 MB full-CNNSpot frame.

## The problem

The manuscript's Table 13 — 38 generator rows × 4 training sets × {ACC, AP} — entered the paper
repo at `b37fce3` hand-written, derived from no tracked artifact. Its per-training-set AP column
means **do not reproduce** `revision_export/tables/e3_cross_matrix_mAP.csv`, the matrix behind main
Table 3: 6 of the 9 off-diagonal blocks disagree, by up to **+0.166**. So the appendix and the main
text were describing different models.

The regeneration spec's diagnosis — that the old table was produced from the **published,
augmented** checkpoints while Table 3 reports the **matched, no-augmentation** `D_h` — is partly
confirmed and partly not. Scoring the published checkpoints on these test frames reproduces three
of the disputed cells to ~0.005 (SynthCLIC→CNNSpot 0.3804 vs the table's 0.3757; SynthCLIC in-domain
0.9249 vs 0.9225; SB+→SynthCLIC 0.6377 vs 0.6350) but misses the rest by 0.03–0.06. The old table is
a **mixture** and cannot be reconstructed from any single source — which is the argument for
regenerating it rather than patching it.

## What was produced

12 evaluations = 4 training sets × 3 test corpora, all from **one** detector definition: `D_h`, the
canonical 1024-d pooler head under the F1 matched recipe (Adam lr 1e-3 / wd 0.01 coupled L2, label
smoothing 0.1, batch 64, ≤200 epochs, early stop on val CE patience 5, restore best, frozen cached
pooler, **no augmentation**, no standardization, seed 123).

**Eleven of the twelve cells already existed** as per-image predictions from the E3 cross-dataset
runs, and E3 *is* the matched recipe — `scripts/run/run_linear_probe.py`'s defaults are the recipe
knob-for-knob, F1's own regression anchor is one of these runs (F1 0.9230 vs E3 0.9239), and
[`TableA-uniform`](../TableA-uniform/README.md) already characterises the same runs as "the
re-trained probe on cached embeddings, no augmentation". Re-aggregating those predictions rather
than refitting has two consequences that refitting would have lost:

1. the acceptance check against Table 3 becomes **exact** rather than approximate — a refit lands
   ~0.005 away and 2-decimal agreement would then be luck;
2. the SynthBuster+ block needs **no new read of the closed SB+ test split** (§3).

The twelfth cell (Combined → CNNSpot) has no stored predictions, so it is computed by scoring the
matched combined head on the full CNNSpot test frame. That head is checked against the combined
predictions that *do* exist before it is used (max |Δscore| **1.2e-07** vs
`e3pred/combined__to__synthclic`).

**Metric conventions, both declared in the caption:**

| corpus | `real_pairing` | test images | generators |
|---|---|---|---|
| CNNSpot | `matched` — each generator scored against its own reals | 108,310 | 21 |
| SynthBuster+ | `shared` — all generators vs the one RAISE-1k pool (200 test reals) | 2,800 | 13 |
| SynthCLIC | `shared` — all generators vs the one `clic2020` pool (428 test reals) | 2,140 | 4 |

**Display names.** Row and column labels come from [`config/mappings.yaml`](../../../config/mappings.yaml)
— `model_name_map` for the four trained models across the top, and each corpus's **own**
`<dataset>_source_map` for the generator rows. Per-corpus matters: the config spells `FLUX.1-dev` as
`FluxDev` under SynthBuster+ and `FLUXDev` under SynthCLIC (likewise `FLUX.1-schnell`), so both
spellings appear a few rows apart in the rendered table. Applied as configured and flagged in the
export README rather than silently unified — the config is the authority. An unmapped generator is a
hard error, not a fall-through to the raw id. The CSV keeps the raw ids in `generator` /
`train_set` / `test_corpus` (the join keys into the snapshot and the E3 predictions) and carries the
label alongside in `generator_display`.

**ACC threshold.** No F artifact records one, so it is declared: **`p̂ > 0.5`, equivalently logit
`z > 0`** — the head's own decision rule, the only threshold available without tuning on the
evaluation population — applied uniformly to every cell and computed on the same generator group as
AP. Under `shared` pairing that means every generator in a block carries the same real-side errors;
that is a property of the paper's metric convention, and reporting ACC on any other grouping would
make the two columns describe different populations.

## Acceptance check — 11/12 exact, 1 recorded exception

Generated per-training-set column means vs `e3_cross_matrix_mAP.csv`:

| test corpus | train set | generated | authoritative | Δ | old table | status |
|---|---|---|---|---|---|---|
| CNNSpot | CNNSpot | 0.9640 | 0.9640 | −0.0000 | 0.9643 | ok |
| CNNSpot | SynthBuster+ | 0.6692 | 0.6692 | −0.0000 | ~0.6752 | ok |
| CNNSpot | SynthCLIC | 0.4222 | 0.4222 | −0.0000 | 0.3757 | ok |
| CNNSpot | Combined | **0.8799** | 0.8982 | −0.0183 | 0.8410 | **frame mismatch** |
| SynthBuster+ | CNNSpot | 0.4962 | 0.4962 | 0.0000 | 0.6623 | ok |
| SynthBuster+ | SynthBuster+ | 0.9952 | 0.9952 | 0.0000 | ~0.9942 | ok |
| SynthBuster+ | SynthCLIC | 0.7940 | 0.7940 | 0.0000 | ~0.7850 | ok |
| SynthBuster+ | Combined | 0.9761 | 0.9761 | 0.0000 | 0.9608 | ok |
| SynthCLIC | CNNSpot | 0.5214 | 0.5214 | −0.0000 | 0.5575 | ok |
| SynthCLIC | SynthBuster+ | 0.6112 | 0.6112 | 0.0000 | 0.6350 | ok |
| SynthCLIC | SynthCLIC | 0.9239 | 0.9239 | −0.0000 | 0.9225 | ok |
| SynthCLIC | Combined | 0.8670 | 0.8670 | −0.0000 | ~0.8700 | ok |

(`~` = reconstructed from the deltas the spec quotes rather than read off the manuscript.)

### The one disagreement is upstream, and it is in the authoritative matrix

`e3_cross_matrix_mAP.csv`'s **CNNSpot column is itself inhomogeneous**. Two generations of E3
CNNSpot evaluations exist under one file-name pattern:

| generation | CNNSpot eval frame | images | generators |
|---|---|---|---|
| 2026-06-24 04:4x | cnnspot-small test | 4,000 | 20 (no `seeingdark` synthetics) |
| **2026-06-24 20:0x** | **full CNNSpot benchmark test** | **108,310** | **21** |
| 2026-06-30 / 07-01 (combined only) | cnnspot-small test | 4,000 | 20 |

The two generations are the same trained heads (scores agree to 1.5e-07); only the CNNSpot
*population* differs. The matrix's CNNSpot-, SynthBuster+- and SynthCLIC-trained cells come from the
full frame; its **Combined cell (0.8982) is the only one measured on cnnspot-small**, because the
combined runs were done later and never scored the full frame.

This table evaluates all four training sets on the **full** frame, which is what the manuscript's
21-row CNNSpot block requires and what three of the four matrix cells already report. So:

> **Recommended paper-side edit (not applied here):** `e3_cross_matrix_mAP.csv` /
> `tex/e3_cross_dataset_mAP.tex`'s `combined → CNNSpot` cell should move from **0.8982 to 0.8799**
> so the column shares one evaluation population. It is a main-text number, so the change is flagged
> rather than made.

### Recipe equivalence — the reused predictions are the F1 estimator

Refitting each training set with `finalexp.trainer` (F1's trainer, seed 123) and scoring the two
corpora whose test splits are open:

| train set | test corpus | refit mAP | shipped cell | Δ |
|---|---|---|---|---|
| CNNSpot | CNNSpot | 0.9638 | 0.9640 | −0.0002 |
| CNNSpot | SynthCLIC | 0.5258 | 0.5214 | +0.0045 |
| SynthBuster+ | CNNSpot | 0.6640 | 0.6692 | −0.0052 |
| SynthBuster+ | SynthCLIC | 0.6164 | 0.6112 | +0.0052 |
| SynthCLIC | CNNSpot | 0.4217 | 0.4222 | −0.0004 |
| SynthCLIC | SynthCLIC | 0.9230 | 0.9239 | −0.0009 |
| Combined | CNNSpot | 0.8728 | 0.8799 | −0.0071 |
| Combined | SynthCLIC | 0.8783 | 0.8670 | +0.0113 |

Every deviation is at or below the ±0.005 dataloader/RNG scatter F1 already reports for its own
anchor, except the two Combined rows (−0.007 / +0.011) — the combined training set is 2.3× larger,
so its shuffle order diverges further from the persisted run. All are far below the 0.03–0.17
discrepancies that motivated the regeneration. **SynthBuster+ test is excluded from this check on
purpose** (§3).

## The three decisions the spec asked for

**§4.1 — the Combined training corpus.** *Re-admitted.*
`data/embeddings/combined_clip_large_patch14.pkl` is registered as `pooler/combined` under
`build_data_snapshot.py --with-appendix`, and `EXCLUDED.md` records the re-admission rather than
dropping the exclusion note. The frame is the exact union of the three per-corpus frames (32,814 =
8,000 + 13,999 + 10,815) with identical split assignments, so the Combined column is on the same
images as the other three. The full Combined column ships; the appendix is **not** narrower than
main Table 3.

One thing the re-admission surfaced: `data/checkpoints/linear_probe_combined.ckpt` is **not** the
published augmented probe its README claims. The 2026-07-01 combined E3 run overwrote it in place
with the matched no-augmentation head. It is registered as
`ckpt/linear_probe_combined_matched` and `EXCLUDED.md` warns against quoting it as an augmented
bridge target beside `ckpt/linear_probe_{synthclic,cnnspot}`, which genuinely are published.

**§4.2 — the closed SynthBuster+ test split.** *No read is required, so no protocol call is needed.*
The SB+ block re-aggregates per-image predictions the E3 cross-dataset runs wrote on **2026-06-24**,
before `EXTERNAL_VALIDATION_PROTOCOL.md` was frozen (2026-07-18). The protocol scopes its lockbox to
the interpretation program and states in its own opening that SB+ test *is* "used by the manuscript
and E1/E3 detection re-runs; never by any vocabulary/interpretation analysis". This table is a
detection table built from exactly those runs. No model here is fitted on or scored against SB+
test — the recipe-equivalence refit skips it deliberately, and `finalexp.spaces.CLOSED_SPLITS` stays
in force for every F-experiment. **The block is reported on test, and the caption needs no
qualification.**

**§4.3 — `SeeingDark`.** *Kept; 21 rows.* The row is absent from F1's CNNSpot split because F1
trains on **cnnspot-small**, where `seeingdark` contributes 9 real images and no synthetics. On the
full CNNSpot benchmark test set it has **180 synthetics and 180 reals** and is a perfectly ordinary
row (in-domain AP 0.832). Choosing the full frame therefore resolves this decision and the column-
mean requirement together: 21 generators, mean 0.9640, matching the matrix exactly.

## Reading

**The appendix and the main text now agree by construction.** Eleven cells are the same numbers
Table 3 reports, re-disaggregated; the twelfth is the same head on the same frame as its column
neighbours. The corrections to the old table are not cosmetic — `SynthBuster+ → CNNSpot` moves
**0.6623 → 0.4962** and `SynthCLIC → CNNSpot` moves **0.3757 → 0.4222**, so any prose reading a
transfer *ordering* off the old appendix needs re-checking. Section 4.1's "Full cross-dataset and
per-generator results are reported in Appendix B.7" is the sentence to re-audit first.

The regeneration also leaves a caveat worth keeping: the CNNSpot block's 21 generators are **not**
the population F5/F6/E11 interpret. Those use cnnspot-small (4,000 test images, 20 generators, and a
different in-domain mAP: 0.9752 vs 0.9640). The two frames are both legitimate and both now
registered under distinct snapshot ids; nothing should quote a CNNSpot number without saying which.

## Files

| file | contents |
|---|---|
| `artifacts/per_generator_detail.csv` | 152 rows = 38 generators × 4 training sets, with ACC, AP, n_fake, n_real, pairing (raw ids; display names are added at export) |
| `artifacts/acceptance.csv` | the 12-cell check against `e3_cross_matrix_mAP.csv` |
| `artifacts/recipe_equivalence.csv` | refit-under-F1 vs reused-E3 comparison |
| `artifacts/summary.json` | detector, recipe, threshold rule, pairing, cell provenance, acceptance verdict |
| `artifacts/run_meta.json` | script, argv, git commit, sha256 of every input read, host, duration |

Exported to `revision_export/` as `tables/clip_per_generator_detail.csv`,
`tex/clip_per_generator_detail.tex`, `tables/clip_per_generator_detail_acceptance.csv`,
`tables/clip_per_generator_detail_run_meta.json` and `clip_per_generator_detail_README.md`.
