# Table A — making the "CLIP linear probe (SynthCLIC)" row uniform

**Status:** ✅ computed · **Driver:** `scripts/finalexp/fix_table_a_row.py` ·
**Artifacts:** [`artifacts/`](artifacts/)

```bash
make finalexp-data WITH_CFEVAL=1          # adds the 206 MB CF-Eval frame to the snapshot
uv run python scripts/finalexp/fix_table_a_row.py
```

## The problem

`revision_export/tex/e1_e3_e6_e7_detector_comparison.tex`'s `CLIP linear probe (SynthCLIC)` row is
**provenance-mixed**:

| columns | source | training |
|---|---|---|
| SynthCLIC / SynthBuster+ / CNNSpot (0.92 / 0.79 / 0.42) | `results/e3_xdataset/` | re-trained on cached embeddings, **no augmentation** |
| CF-Eval (0.73) | `data/checkpoints/linear_probe_synthclic.ckpt` | **published, augmented** |

T0 (config-audit §F) fixed this row's *head* mismatch (k=8 → k=1) but not the *checkpoint
provenance* mismatch, even though T0's stated goal was "no table mixes heads/metrics".

## The fix (decision 2, 2026-08-08)

Make the row uniform on the **matched** probe — columns 1–3 already are — by scoring F1's
primary-seed head on the cached CF-Eval embeddings.

| detector | overall AP | **mAP-by-generator** (the Table A cell) |
|---|---|---|
| **matched probe (F1 seed 123)** | 0.7067 | **0.7233** |
| deployed k=1 (published, augmented) | 0.7326 | 0.7316 |
| deployed k=8 (appendix ablation) | 0.7296 | 0.7340 |

21 generators. Δ(matched − deployed) on the table's metric: **−0.008**.

> **Metric caution.** `cf_metrics`' `mAP` field is the *pooled* AP (0.7067 here), **not**
> mAP-by-generator. Quoting it against the deployed row's 0.7316 would compare two different
> quantities and overstate the gap ~3×. The cell above is computed with the export's own
> `per_generator` helper (mean AP within each `source` group).

## Reading

**Adopting the matched probe moves the CF-Eval cell from 0.73 to 0.72** — visible at two decimals,
immaterial to any conclusion. And it corroborates F7 independently: the deployed probe's advantage
here is **+0.008 mAP-by-gen**, essentially the same **+0.007 AUROC** augmentation effect F7 measured
on SynthCLIC. Two different datasets and two different metrics agree that augmentation is worth
under a hundredth — which is the measurement that retracts E12's attribution of the ~0.02–0.04
proxy gap to augmentation.

## Remaining step (not applied)

`package_revision_export.py`'s `table_a()` still pulls `clip_cf` from `e7_main()`. Repointing it at
this row is a one-line change, deliberately **left unapplied** so the export is not silently
altered: the numbers above are the input to that decision, not the decision itself. The deployed
k=1/k=8 CF-Eval numbers stay as the appendix ablation either way.
