# Experiments

The paper has two kinds of experiment, and the difference is **what they are evidence for** — not
when they were run.

|  | **F1–F7** — the spine | **E-experiments** — the surrounding questions |
| --- | --- | --- |
| answers | *Are the interpretation claims sound?* | *Does the finding survive contact with X?* |
| method | one matched recipe, one frozen checksummed snapshot, five seeds each | whatever each question needs — a baseline model, a sweep, a second backbone |
| models | only `D_h` / `D_e` / `D_cue`, retrained from scratch | the deployed checkpoints, external detectors, variants |
| reproduce | `make finalexp-all` — CPU, a few minutes, fully checksummed | per-experiment drivers under `scripts/run/` and `scripts/interpret/` |

**If you want to check the paper's central claims, run F1–F7.** They are self-contained and cheap.
The E-experiments are the robustness and baseline evidence around those claims; several need inputs
that are not published (raw images, external checkpoints).

The two are numbered independently and the numbers do **not** correspond — F3 is the projected
analysis head, E3 is the additional-backbone table.

> **Ownership rule.** One home per number. This file owns the E-experiments; F1–F7 are owned by
> [`reproduction/experiments/final_consolidation/`](../reproduction/experiments/final_consolidation/README.md); figures are
> owned by [`reproduction/experiments/figures/README.md`](../reproduction/experiments/figures/README.md). Each links to the
> others rather than restating them, so nothing can drift.

## F1–F7 — the methodological spine

These are what the manuscript's interpretation claims now rest on. All CPU, a few minutes end to end
(`make finalexp-all`); every input is checksummed and every run records its own provenance.

| id | question | headline |
| --- | --- | --- |
| **F1** | Is the canonical direction stable across seeds? | direction Σ-cos **0.989**, cue profile ρ **0.991** — but the top-50 extreme *images* agree only 0.73 |
| **F2** | k=1 vs k=8, matched | the effective boundary is stable (0.963); the **individual axes are not** (0.289, cue ρ 0.119) |
| **F3** | What does analysing in the projected space cost? | **+0.029** AUROC [+0.021, +0.037]; cue profile ρ 0.987 |
| **F4** | What does restricting to 168 named cues cost? | **−0.059** [−0.075, −0.043] vs `D_e`; excess recovery 0.85 |
| **F5** | Extreme-image montages | ranked by the detector's own logit; retires an auxiliary fixed-`C` fit |
| **F6** | Cross-dataset heads and boundary decomposition | GAN-era and diffusion-era boundaries near-orthogonal; the difference is **artifacts vs optics** |
| **F7** | Do the deployed checkpoints behave like the matched heads? | both primary pairs pass; augmentation effect only **+0.007** |

Detail per experiment: [`reproduction/experiments/final_consolidation/`](../reproduction/experiments/final_consolidation/README.md).

## The E-experiments

Each has a section or table in the manuscript. Where a finding is appendix-only, the row says so.

| id | topic | finding | driver |
| --- | --- | --- | --- |
| **E1** | Low-level forensic baseline | a forensic CNN is excellent in-domain (0.98 mAP on SynthCLIC) and collapses under dataset and generator shift | `scripts/run/run_forensic_baseline.py` |
| **E2** | β sensitivity of the concept model *(appendix)* | sparsity and accuracy trade off gently; the useful range is β ≈ 1e-4 – 3e-4, and active-concept count is only an approximate measure of explanation complexity | `scripts/run/run_beta_sweep.py` |
| **E3** | Additional CLIP backbones *(appendix)* | the finding is not an artifact of ViT-L/14-336: B/16 and B/32 show the same in-domain pattern | `scripts/run/run_linear_probe.py` |
| **E4** | Why cross-family transfer fails | the drop concentrates in GAN and pixel-space-diffusion generators; detectors trained on different corpora learn largely unrelated concepts | `scripts/run/run_cross_family_analysis.py` |
| **E5** | Activation- vs weight-orthogonality *(appendix)* | equal detection performance; the activation form is the principled choice, and costs nothing | `scripts/run/run_orthogonality_ablation.py` |
| **E6** | A stronger out-of-the-box baseline | CommunityForensics-384 transfers robustly across all three corpora | `scripts/run/run_e6_strong_baseline.py` |
| **E7** | A larger external benchmark | on CF-Eval (21 generators) the CLIP head transfers best overall but is weakest on GANs, exactly where the forensic CNN is strongest ⇒ the two are complementary | `scripts/run/run_community_eval.py` |
| **E8** | Interpretation stability and mechanism | individual concept identities are seed-driven; the causal picture is distributed and redundant. Content-controlled pairing gives the cleanest cue estimates | `scripts/run/run_interpretation_stability.py` |
| **E11** | Boundary-normal decomposition *(appendix)* | GAN-era and diffusion-era decision normals are near-orthogonal; the signed difference splits into **artifacts** (CNNSpot) versus **aesthetics and provenance** (SynthCLIC). Erasing the top axes moves the inverted transfer only partway ⇒ partly nameable, mostly distributed | `scripts/interpret/run_boundary_decomp.py` |
| **E12** | Score-space cue alignment *(appendix)* | interprets the **published** checkpoints directly, with no re-fit: deployed k=1 and k=8 cue profiles agree at ρ 0.992, and proxy-based interpretation transfers. A coherent *idealization* axis appears (synthetic ⇒ cleaner, smoother, more composed) but stays diffuse. The same detector keeps one named axis across corpora **even where its label association inverts** ⇒ the cue profile belongs to the training corpus, not to the task | `scripts/interpret/run_score_alignment.py` |

**E9 and E10 are absent by design.** They were exploratory work — a vocabulary-optimization search
and an interventional edit battery — that informed the analysis but produced no manuscript claim.
Their code has been removed rather than shipped unused. The numbering keeps its gap because E11 and
E12 are cited by name in the checksummed `run_meta.json` provenance records.

## Two conclusions that shaped the repository

**Interpretation is corpus-specific, not task-specific.** A detector trained on SynthCLIC and one
trained on CNNSpot key on unrelated cues (ρ 0.008), and the same detector applied across corpora
keeps its axis while its label association flips. Any claim of the form "CLIP detects synthetic images
by X" must name the corpus.

**One cue is a collection shortcut.** `watermark` ranks 8th on SynthCLIC and falls below the null
elsewhere — a property of how that corpus was assembled, not of synthetic images. It is a useful
reminder of what a cue ranking can pick up.

Before building any table from these results, check the caveats that apply to it: the artifact-level
ones in [`reproduction/experiments/data/`](../reproduction/experiments/data/) (including
`EXCLUDED.md`, which names every retracted artifact), and the per-figure ones in the
[figure ledger](../reproduction/experiments/figures/README.md).
