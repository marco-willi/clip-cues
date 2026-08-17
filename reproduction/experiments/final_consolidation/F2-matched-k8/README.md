# F2 — Matched k=1 vs k=8: the boundary is stable, its decomposition is not (spec E2)

**Status:** ✅ done · **Driver:** `scripts/finalexp/run_f2_matched_k8.py` ·
**Artifacts:** [`artifacts/`](artifacts/) · **Runs:** `runs/k8_seed{123..127}`
(k=1 runs reused by reference from [F1](../F1-canonical-stability/README.md))

```bash
uv run python scripts/finalexp/run_f2_matched_k8.py
```

**Inputs:** `pooler/synthclic`, `cue_scores/synthclic__antonyms`, `vocab/antonyms`.

## Why

The k=1 probe and the k=8 orthogonal head were trained under **different protocols** — the appendix
says so — while the Results compared their detection performance anyway. F2 removes the confound:
identical cached features, identical recipe, identical seeds, no augmentation. The only changes are
the factorized parameterization and the activation-orthogonality penalty (λ=0.33, `non_linear=False`).

Two objects must be kept apart, and doing so is the point of the experiment:

- the **effective direction** `w_eff = w2 @ W1` — exact, because the head is linear when
  `non_linear=False` (verified to \<1e-5 against the module's own forward pass in
  `tests/test_finalexp.py`); this is E12's `W₀ᵀw_logit`;
- the **individual factorized axes** `W1[j]`, which are what earlier interpretation work read
  concepts off.

Individual axes are compared under **Hungarian matching on |cosine|**. Without it, arbitrary axis
ordering between seeds manufactures instability and the headline result would be an artifact of
permutation rather than a property of the factorization.

## Result

| quantity | k=1 refits | k=8 effective direction | k=8 individual axes |
|---|---|---|---|
| detection AUROC (mean) | 0.9212 | 0.9152 | — |
| detection mAP (mean) | 0.9228 | 0.9171 | — |
| direction cosine (Σ-metric) | **0.9891** | **0.9634** | — |
| direction cosine (raw) | 0.9788 | 0.9221 | **0.2893** (matched |cos|) |
| 168-cue profile Spearman | **0.9913** | **0.9790** | **0.1188** |
| top-50 extreme overlap | 0.6951 | 0.5027 | — |

Same-seed agreement between the two heads: Σ-cos(k=8 `w_eff`, k=1 `w`) = **0.9742**,
raw 0.9452, logit Spearman 0.9673, cue-profile ρ **0.9855**.

## Reading

**The classifier boundary is stable; its decomposition into eight axes is not.** Under a genuinely
matched protocol the k=8 head detects at the same level as k=1 (AUROC 0.9152 vs 0.9212, ranges
overlapping), its *effective* direction is stable across seeds (Σ-cos 0.963) and describes the same
cues (ρ 0.979) — while its *individual axes* reshuffle almost completely (matched |cos| 0.289, cue
profile ρ 0.119, one pair even negative at −0.108).

That is exactly the reason to retain k=1 and to stop interpreting individual k=8 axes: the axes are
an arbitrary basis of a stable subspace, not eight findings. It also explains why the k=8 head can
be scored through its final logit — as E12 did — without choosing among directions.

Consistency with the prior record: E12/N23 found the *deployed* k=1 and k=8 heads share a cue
profile at ρ 0.992; here the matched pair agrees at ρ 0.985 same-seed. The k=8 head buys no
different semantic description.

## Files

| file | contents |
|---|---|
| `artifacts/summary.json` | the 3×4 table, detection, same-seed cross-family agreement |
| `artifacts/stability.csv` | every seed pair × family (`k1_refits`, `k8_effective`, `k8_individual_axes`) |
| `artifacts/k1_vs_k8_same_seed.csv` | per-seed agreement between the two heads |
| `runs/k8_seed*/weights.npz` | `w_eff`, bias, and the 8 individual `axes` |
| `runs/k8_seed*/cue_profile.csv` | per-cue correlations of the effective direction |
