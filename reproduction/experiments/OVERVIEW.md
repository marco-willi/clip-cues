# Experiment ledger — clip-cues revision (ARRAY-D-26-00829)

**One row per experiment across the whole project.** This is an index, not a second source of
truth.

**F** and **E** are two independent series and their numbers do not correspond: F1–F7 are the
matched-recipe methodological spine, the E-experiments are the reviewer-driven questions around it.
See [`docs/EXPERIMENTS.md`](../../docs/EXPERIMENTS.md) for what separates them.

> ### Ownership rule (read this before editing)
>
> - **The E-experiments** are owned by [`docs/EXPERIMENTS.md`](../../docs/EXPERIMENTS.md). This
>   ledger **links to them and never restates their numbers.**
> - **F1–F7** are owned by [`final_consolidation/`](final_consolidation/). `docs/EXPERIMENTS.md`
>   links *back* here rather than copying.
>
> Without this rule the repo acquires a third competing dashboard that silently drifts from the
> other two.

Every F-run records its own provenance in `run_meta.json` (script, argv, git commit, the sha256 of
every input it read, package versions, host, wall time), and every input is checksummed in
[`data/MANIFEST.md`](data/MANIFEST.md).

______________________________________________________________________

## F1–F7 — final methodological consolidation

Collapses the model inventory to **two** CLIP linear directions — canonical 1024-d `D_h` and
projected 768-d `D_e` — by retraining everything under one matched recipe on one frozen snapshot.
Task file: [`final_consolidation/README.md`](final_consolidation/README.md).

| id | topic | status | driver script | headline | detail |
|---|---|---|---|---|---|
| **F1** | Canonical 1024-d stability, 5 seeds | ✅ | `scripts/finalexp/run_f1_canonical_stability.py` | direction stable (Σ-cos **0.989**), cue profile ρ **0.991**, but extreme images only 0.70 | [F1](final_consolidation/F1-canonical-stability/README.md) |
| **F2** | Matched k=1 vs k=8 | ✅ | `scripts/finalexp/run_f2_matched_k8.py` | boundary stable (Σ-cos 0.963), **individual axes are not** (0.289, cue ρ 0.119) | [F2](final_consolidation/F2-matched-k8/README.md) |
| **F3** | Projected 768-d analysis head | ✅ | `scripts/finalexp/run_f3_projected_head.py` | projection cost **+0.029** [+0.021, +0.037]; cue-profile ρ **0.987** | [F3](final_consolidation/F3-projected-head/README.md) |
| **F4** | 168-cue restricted-information probe | ✅ | `scripts/finalexp/run_f4_cue_capacity.py` | ant168 **−0.059** [−0.075, −0.043] vs `D_e`; excess recovery **0.85** | [F4](final_consolidation/F4-cue-capacity/README.md) |
| **F5** | Extreme-image montage from the matched probe | ✅ | `scripts/finalexp/export_f5_rankings.py` | ranks by the detector's own logit; retires the fixed-`C=1` fit. Figure's **seed overlap only 0.73 / 0.66** | [F5](final_consolidation/F5-canonical-montage/README.md) |
| **F6** | Cross-dataset heads + boundary decomposition | ✅ | `scripts/finalexp/run_f6_cross_dataset.py` | replicates N21's **artifacts-vs-optics** Δ split; Σ-metric sharpens near-orthogonality (sc~cnnspot **−0.21**) | [F6](final_consolidation/F6-cross-dataset/README.md) |
| **F7** | Bridge to the deployed checkpoints and the proxies | ✅ | `scripts/finalexp/run_f7_bridge.py` | **both primary pairs pass** ⇒ N7–N25 transfer by citation; augmentation effect only **+0.007** | [F7](final_consolidation/F7-bridge/README.md) |
| — | Table A uniform-row fix (Step 11) | ✅ | `scripts/finalexp/fix_table_a_row.py` | matched probe on CF-Eval mAP-by-gen **0.723** vs deployed 0.732 (−0.008) | [TableA](final_consolidation/TableA-uniform/README.md) |
| — | Table B — appendix per-generator table regeneration | ✅ | `scripts/finalexp/run_appendix_per_generator.py` | 38 gens × 4 train sets from one detector; column means reproduce Table 3 in **11/12 cells exactly**, 12th is a frame mismatch in the matrix | [TableB](final_consolidation/TableB-per-generator/README.md) |

Reproduce everything on CPU:

```bash
make finalexp-data      # build + verify the frozen input snapshot (~330 MB)
make finalexp-all       # verify, then run F1-F7

# the two table fixes need heavier optional inputs in the snapshot:
make finalexp-data WITH_CFEVAL=1 WITH_APPENDIX=1   # +816 MB (CF-Eval, combined, full CNNSpot, E3 predictions)
make finalexp-tableb                                # regenerate + export the appendix per-generator table
```

______________________________________________________________________

## The E-experiments (authoritative docs: [`docs/EXPERIMENTS.md`](../../docs/EXPERIMENTS.md))

Numbers deliberately omitted here — follow the link.

| id | reviewer | topic | status | driver script |
|---|---|---|---|---|
| E1 | R3 | Low-level forensic baseline | ✅ | `scripts/run/run_forensic_baseline.py` |
| E2 | R3 | β sensitivity (sparsity ↔ mAP) | ✅ | `scripts/run/run_beta_sweep.py` |
| E3 | R1 | Additional CLIP backbones | ✅ | `scripts/run/run_linear_probe.py` |
| E4 | R1 | Cross-family 0.37 mAP analysis | ✅ | `scripts/run/run_cross_family_analysis.py` |
| E5 | R3 | Activation- vs weight-orthogonality | ✅ | `scripts/run/run_orthogonality_ablation.py` |
| E6 | R1 | Stronger out-of-the-box baseline | ✅ | `scripts/run/run_e6_strong_baseline.py` |
| E7 | R1 + editor | Larger external benchmark (CommunityForensics) | ✅ | `scripts/run/run_community_eval.py` |
| E8 | R1 #5/#7 | Interpretation stability & mechanism | ✅ | `scripts/run/run_interpretation_stability.py` |
| E11 | R1 #5/#7 | Boundary-normal decomposition + external read | ✅ | `scripts/interpret/run_boundary_decomp.py` |
| E12 | R1 #7 | Score-space cue alignment vs deployed checkpoints | ✅ | `scripts/interpret/run_score_alignment.py` |

______________________________________________________________________

## Manuscript figures

Owned by [`figures/README.md`](figures/README.md) — the same ownership rule as above: that file
holds every figure path, claim, source and required caption caveat, and this ledger only links to
it. It also carries the **removal list** for the figures dropped from the previous manuscript.

Seven main-text figures (Fig 1 and Fig 2 out of scope) plus two appendix figures, rebuilt with
`make figures-all`. Fig 3-7 read either the checksummed `experiments/data/` snapshot or F-experiment
artifacts, so a figure cannot drift from the numbers reported above.

## Where things run

Local CPU throughout — the F-experiments train linear heads on cached features and take seconds per
seed. The shared runner times every step, so the cost below is measured rather than estimated.

Observed: training is 2–6 s per seed. Wall time is dominated by the analyses — **F6's lasso
decomposition 829 s**, F3's paired cluster bootstrap ~36 s, F5 30 s, F7 \<1 s. `make finalexp-all`
is a few minutes end to end, and no step needs a GPU or a remote machine.
