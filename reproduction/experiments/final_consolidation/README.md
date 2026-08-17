# Final methodological consolidation (F1–F7)

**The task file.** Recipe, configuration, code map, results, artifact inventory, and what changed
versus the old model inventory.

- **Ledger:** [`../OVERVIEW.md`](../OVERVIEW.md) · **Inputs:** [`../data/MANIFEST.md`](../data/MANIFEST.md)

## Goal

Reduce the paper's conceptual model inventory to **two** CLIP linear directions:

| object | space | role |
|---|---|---|
| **`D_h`** — canonical detector | 1024-d `pooler_output` | *the* detector and the primary interpretation target |
| **`D_e`** — projected analysis head | 768-d shared image–text | used only where direct geometric comparison with text embeddings is required |

Everything else is explicitly subordinate: the 168-cue classifier is a **restricted-information
probe**, the k=8 factorized head is a **diagnostic ablation** showing why individual axes should not
be interpreted, and the concept model is a separate, intrinsically text-grounded classifier.

## The matched recipe (one definition, F1–F4 and F6)

From the canonical probe (`scripts/run/run_linear_probe.py`), cross-checked against the configuration audit that accompanies the reproduction code:

| knob | value |
|---|---|
| optimizer | Adam, lr 1e-3, weight decay 0.01 (**coupled** L2, not AdamW) |
| loss | BCE-with-logits, label smoothing 0.1 |
| batch / epochs | 64 shuffled / ≤ 200 |
| model selection | early stop on val cross-entropy, patience 5, restore best |
| features | frozen, cached, **no augmentation**, **no standardization** |
| seeds | 123–127 (initialization + shuffle order) |
| head | `LinearHead(d)` · F2 uses `ActivationOrthogonalityHead(d, [8], non_linear=False, λ=0.33)` |

F1/F3/F4 differ **only** in input dimension (1024 / 768 / 168).

### One deviation from the spec, and why

The spec says to feed `e = Ph/‖Ph‖`. Doing that literally **breaks the comparison**, and the effect
is large:

| features | mean row norm | AUROC (same recipe) |
|---|---|---|
| pooler `h` | 32.95 | 0.921 |
| projected `Wp h` | 18.83 | 0.888 |
| projected, unit-normalized | 1.00 | **0.725** |

A fixed `weight_decay = 0.01` is only "the same regularization" if the spaces have comparable
scale. On unit-normalized features the penalty dominates the likelihood term, and **more epochs do
not fix it** (0.744 at 212 epochs). The giveaway was F4: the 168-cue *restricted* probe initially
scored **0.800**, above the 768-d *unrestricted* probe it is a strict subspace of — impossible
unless the latter is under-trained.

**So every space is rescaled by one global scalar** to the pooler train split's mean row norm
(`finalexp.features.match_scale` via `finalexp.spaces.load`). This preserves geometry exactly, uses
no per-dimension statistics — so it is *not* standardization and brings back no coefficient
back-transformation machinery — and makes the fixed weight decay an honest constant. Cue scores
keep their unit normalization because `c_j = ⟨e/‖e‖, v_j⟩` is a cosine by definition, then get the
same global rescale.

## Code map

| id | driver script | modules | artifacts |
|---|---|---|---|
| — | `scripts/finalexp/build_data_snapshot.py` | `finalexp.snapshot`, `finalexp.data` | `experiments/data/{manifest.json, MANIFEST.md, EXCLUDED.md}` |
| — | `scripts/finalexp/prepare_features.py` | `finalexp.features`, `finalexp.snapshot` | `experiments/data/embeddings/{projected_derived,cue_scores}/` |
| — | `scripts/finalexp/verify_data_snapshot.py` | `finalexp.snapshot` | exit code + report |
| **F1** | `run_f1_canonical_stability.py` | `trainer`, `stability`, `profiles`, `evaluation`, `runner` | `F1-canonical-stability/` |
| **F2** | `run_f2_matched_k8.py` | `trainer`, `stability`, `profiles` | `F2-matched-k8/` |
| **F3** | `run_f3_projected_head.py` | `spaces`, `trainer`, `evaluation`, `stability` | `F3-projected-head/` |
| **F4** | `run_f4_cue_capacity.py` | `spaces`, `trainer`, `evaluation` | `F4-cue-capacity/` |
| **F5** | `export_f5_rankings.py` | `stability`, `snapshot` | `F5-canonical-montage/`, `experiments/data/rankings/` |
| **F6** | `run_f6_cross_dataset.py` | `spaces`, `trainer`, `vocab_opt.boundary` | `F6-cross-dataset/` |
| **F7** | `run_f7_bridge.py` | `spaces`, `stability`, `profiles` | `F7-bridge/` |

Package modules live in `src/clip_cues_research/finalexp/`:
`data` (manifest-mediated, checksum-verified input access) · `snapshot` (manifest write side) ·
`features` (projection, cue scores, scale matching) · `spaces` (the three input spaces on a common
scale) · `trainer` (the matched recipe) · `stability` (direction/score/profile metrics) ·
`profiles` (the E12 cue-association estimators, now shared) · `evaluation` (Convention-A metrics,
cluster bootstrap) · `runner` (run folders + `run_meta.json`).

## Provenance

Two layers, both mandatory:

- **Inputs** — every artifact F1–F7 reads is copied into `experiments/data/`, checksummed, tagged
  with its embedding `space`, and reachable only through `finalexp.data.get_*`, which **verifies the
  sha256 on load**. A guard test (`tests/test_finalexp_data.py`) fails if any F-code uses a literal
  `data/...` path. See [`../data/EXCLUDED.md`](../data/EXCLUDED.md) for what was deliberately left
  out — including the retracted W² vocabulary that is *shape-identical* to the canonical one.
- **Code** — every run folder carries `run_meta.json`: script, full argv, git commit + dirty flag,
  snapshot manifest version, `{input_id: sha256}` for everything read, package versions, host, and
  wall time.

## Results

See each experiment's README; headline numbers are in [`../OVERVIEW.md`](../OVERVIEW.md).

| id | topic | headline |
|---|---|---|
| [F1](F1-canonical-stability/README.md) | canonical stability | direction Σ-cos **0.989**, cue profile ρ **0.991**; extreme images only **0.70** |
| [F2](F2-matched-k8/README.md) | k=1 vs k=8 | boundary stable (0.963), **individual axes are not** (0.289; cue ρ 0.119) |
| [F3](F3-projected-head/README.md) | projection cost | `D_h` 0.921 vs `D_e` 0.893; **+0.029 [+0.021, +0.037]**; cue-profile ρ 0.987 |
| [F4](F4-cue-capacity/README.md) | cue capacity | ant168 −0.059 [−0.075, −0.043]; **excess recovery 0.85** |
| [F5](F5-canonical-montage/README.md) | extreme images | ranked by the detector's own logit; **seed overlap 0.73 / 0.66** |
| [F6](F6-cross-dataset/README.md) | cross-dataset boundaries | Δ split replicates N21 (artifacts vs optics); CNNSpot weakly identified (val AUROC 1.000) |
| [F7](F7-bridge/README.md) | bridge | **both primary pairs pass** ⇒ downstream N-numbers transfer by citation |

## What changed versus the old inventory

**Retired:**

- the auxiliary **fixed-`C=1` single-direction logistic fit** used only to rank montage images —
  F5 now ranks by the detector's own logit `z = wᵀh + b`;
- the framing of the 168-cue model as "another CLIP detector" — it is a **restricted-information
  probe** (F4);
- the **unmatched k=8 protocol**: the k=8 head is now trained on the same features under the same
  recipe, and is an appendix ablation whose *individual axes* are explicitly not interpretable (F2);
- `P768t` / `P1024t` from the **conceptual inventory**. Their definitions stay in
  INTERPRETATION.md §0 as historical analysis targets — deleting them would make N7–N14 and N18
  unreadable — and F7 measures how the matched heads relate to them.

**Also retired:** the standardization + coefficient back-transformation machinery in the appendix.
The matched heads are fit in raw feature space, so `raw_normal()`-style back-transformation is no
longer part of the boundary-decomposition pipeline (F6).

**Kept:** the deployed checkpoints, solely as F7 bridge targets and appendix ablation rows.

**Table A row made uniform** ([TableA-uniform](TableA-uniform/README.md)): its CF-Eval cell came
from the deployed augmented checkpoint while columns 1–3 came from the matched no-aug probe. Scoring
the matched probe on CF-Eval gives **mAP-by-generator 0.723** vs the deployed 0.732 — a −0.008
change, which independently corroborates F7's +0.007 augmentation effect. The one-line repoint of
`package_revision_export.py` is deliberately left unapplied so the export is not silently altered.
