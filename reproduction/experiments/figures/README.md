# Figure ledger — clip-cues revision (ARRAY-D-26-00829)

**One row per figure: the claim it supports, where its numbers come from, how to rebuild it, and the
caveat its caption must carry.** This file owns figure paths; `experiments/OVERVIEW.md` links here
rather than restating them (same ownership rule the F-experiments follow).

```bash
make figures-all          # everything below
make fig5                 # one figure
```

> **What this repository ships.** The builders produce the four files below plus, for several
> figures, alternative layouts and candidate variants. **Only the variant printed in the manuscript
> is published here**, as PDF + CSV + caption; the PNG twins and the unprinted variants stay in the
> build. Where this file describes a variant that is not shipped, it is documenting what the code
> produces, not what is in this directory.

Every builder writes four files: `<name>.png`, `<name>.pdf`, `<name>.csv` and
`<name>-caption.txt`. The CSV is the exact data the figure was drawn from, so any bar can be checked
against a number without rerunning anything. The caption is a **suggested caption generated from
that data** rather than hand-written, so the counts and values it quotes cannot drift from the
figure on a rebuild — explanatory text that used to sit on the figures now lives there instead.
Figures 3, 4 and 8 decode raw pixels and need `HF_HOME=data/hf_cache` (the `make` targets set it).

______________________________________________________________________

## Main text

| # | figure | claim | source | rebuild |
|---|---|---|---|---|
| 1 | *(method schematic)* | the model hierarchy `D_h` → `D_e` → `D_cue`, concept model separate | — handled outside this repo | — |
| 2 | [`fig2-corpus-examples/`](fig2-corpus-examples/) | paired content, image quality, generator diversity | the published selection, pinned from the archived notebook that drew it | `make fig2` |
| 3 | [`fig3-extreme-scores/`](fig3-extreme-scores/) | what the *deployed-style* detector treats as most real / most synthetic | F5 ranking `ranking/f5_{synthclic,cnnspot}` + HF images | `make fig3` |
| 4 | [`fig4-paired-example/`](fig4-paired-example/) | with content held fixed, synthesis still moves named photographic cues | snapshot `cue_scores/synthclic__antonyms` + HF images | `make fig4` — several variants, one per source photo |
| 5 | [`fig5-cue-population/`](fig5-cue-population/) | those movements persist across the dataset — **and are a different estimand from detector reliance** | same snapshot array + F1 `cue_profile.csv` | `make fig5` |
| 6 | [`fig6-boundary-delta/`](fig6-boundary-delta/) | the semantic boundary changes with the training distribution: artifacts vs optics | F6 `delta_axes.csv`, `boundary_cosines.csv` | `make fig6` |
| 7 | [`fig7-concept-model/`](fig7-concept-model/) | an intrinsically text-grounded classifier can make sparse image-specific predictions | published `cm_antonyms_*` checkpoints + snapshot `projected/*` | `make fig7` |

**Fig 2 is content-frozen.** It reproduces the submitted Figures 2 and 3 (SynthBuster+ and SynthCLIC
example collages) and changes exactly one thing: the output is **PDF**, not the submitted 100-dpi
PNG, so the column headings are vector text like the rest of the set. The photographs are
embedded at 150 dpi. Both collages are drawn far wider than they print (24 in and 12 in of figure
into a 6.4 in `cas-sc` text block), so the on-page resolution is ~3.7x the dpi: 150 lands ~550 ppi,
comfortably above the 300 ppi print standard. Size scales steeply with it — 100 dpi gives a 2 MB
PDF, 150 about 4 MB, 300 about 17 MB for resolution the page cannot show. The four source photos per
collage and the column order are pinned in `figures/corpus_examples.py` from the executed cells of
`archive/detection-via-clip/notebooks/51-mw-visualize-datasets.ipynb`, which is where the submitted
PNGs came from; `tests/test_corpus_examples.py` checks the pin against that notebook's own recorded
draw. This is the one builder that writes no `.png` — that is the point of it — and its CSV is a
cell-by-cell image manifest rather than plotted values.

## Appendix

| # | figure | claim | source | rebuild |
|---|---|---|---|---|
| 8 | [`appendix/fig8-cnnspot-examples`](appendix/) | CNNSpot looks nothing like SynthCLIC — why the two regimes diverge | F5 ranking (median-logit image per generator group) + HF images | `make figures-appendix` |
| 9a | [`appendix/fig9-clipiqa-distributions`](appendix/) | how the two corpora differ in *perceived* quality, independently of any detector | snapshot `projected/*` + `vocab_canon/clipiqa_prompts.pt` | `make figures-appendix` |
| 9b | [`appendix/fig9-clipiqa-axes`](appendix/) | each perceptual axis alone is a weak detector — the residual is diffuse | `outputs/e8/clipiqa/` | `make figures-appendix` |

**Fig 9a reproduces the published figure** (the boxplot of attribute distributions by source, from
the archived notebook), 9b is the per-axis separability view. `outputs/e8/clipiqa/` stores only
aggregates, so 9a **recomputes** per-image scores from the canonical snapshot using the same
definition as `scripts/analyze/analyze_clipiqa.py` — `unit(e) · unit(pos − neg)` — over the **16**
attributes of `clipiqa_full` (`vocab/clipiqa_full` + `vocab/clipiqa_full_poles` in the snapshot),
which is what the published figure shows. The 8-pair `clipiqa_prompts` set is a subset and is not
what this figure is built from. No GPU, no image decoding. One deviation from the original: it used per-source
boxes, which suits SynthCLIC's five sources but not CNNSpot's 23 generator groups, so CNNSpot
contrasts Real against Synthetic instead.

## Tables that replaced figures

[`tables/`](tables/) holds LaTeX emitted by the same builders whose renders were retired, so the
tables are the identical numbers rather than a re-derivation. Rebuild with `make tables-compact`.

| table | contents |
|---|---|
| [`cascade-information-restriction.tex`](tables/cascade-information-restriction.tex) | `D_h` 0.921 → `D_e` 0.893 (−0.029) → `D_cue`, 168 named cues, 0.834 (−0.059), cluster-bootstrap CIs. Each Δ is signed **relative to the row above**, which is the opposite sign from the manuscript's Δ_proj = AUROC(`D_h`) − AUROC(`D_e`) = +0.029 — the header and caption say so explicitly |
| [`stability-summary.tex`](tables/stability-summary.tex) | direction / cue-profile agreement for `D_h` (0.989 / 0.991), the k=8 effective direction (0.963 / 0.979), and k=8 **individual axes** (0.289 / 0.119). The direction column is Σ-cos for the two effective directions but Hungarian-matched |cos| for the individual axes — an axis alone has no calibrated score scale |

Both go through the house emitter in [`figures/latex.py`](../../../src/clip_cues_research/figures/latex.py),
the same envelope as `revision_export/tex/`; [`tests/test_compact_tables.py`](../../../tests/test_compact_tables.py)
holds the LaTeX-safety checks (no text-mode `_`, math-mode minus signs, shipped file not stale).

______________________________________________________________________

## Caption caveats (each is required, not optional)

**Fig 3 — the images are less stable than the direction that ranks them.** F1 puts the decision
direction at Σ-cos **0.989** across seeds and its cue profile at ρ **0.991**, but the top-50 image
sets agree at only **0.73** (SynthCLIC) and **0.66** (CNNSpot). The montage illustrates a stable
direction; it is not a claim about these particular images.

**Fig 3 / Fig 8 — CNNSpot's `source` column is an evaluation subset, not a provenance.** CNNSpot
files each real photograph under the generator group it is paired into, so `source == "progan"` on a
real image means *the real half of the ProGAN evaluation subset* — **not** that ProGAN produced it.
Real panels are therefore labelled from `cnnspot_real_source_map` in
[`config/mappings.yaml`](../../config/mappings.yaml) (Wang et al. 2020, appendix B.1) plus the
subsplit in the image's own path: `LSUN / cat`, `ImageNet`, `COCO`, `FaceForensics++`, … Groups B.1
does not cover (`cyclegan` and the diffusion-era groups) fall back to the subsplit, explicitly marked
as a *subset*, rather than being guessed. Two tests enforce this — the map's contents are pinned, and
no `label == 0` panel may carry a bare generator name.

Worth a sentence in the Fig 3 caption: B.1 also records that the ProGAN and BigGAN **reals** were
centre-cropped on the long edge and resized to 256×256 by the benchmark authors, so CNNSpot's "real"
images are preprocessed crops rather than raw photographs.

**Fig 3 / Fig 8 — split.** Every image is from the **test** split, verified against the published
dataset: all 4,000 ranked ids resolve in `marco-willi/cnnspot-small`'s test split, **0** in train or
validation. Note the F-experiment frame is the **4k** CNNSpot eval frame, a strict subset of that
108,310-row test split.

**Fig 4 — the example is selected, and the rule is stated.** Of 428 source photos paired with all
three generators, the candidates are the most *typical*: highest mean cosine between a pair's own Δ
profile and the population mean Δ (the top one scores 0.76 against a median of 0.40). They are not
the most extreme, which would be a more dramatic and less honest figure.

Variants are named `fig4-paired-example-<id8>` so the figure can be chosen on how clearly it reads.
Because the selection rule is fixed and applied before anyone looks, every candidate is a defensible
choice rather than a curated one; **the manuscript uses one of these files, and its caption should
give the content id.**

Layout is a single row of four images over three bar panels (`--layout grid` gives a 2x2 alternative
with roughly twice the label width, at the cost of a tall portrait figure). Bars are labelled with
the antonym **phrase** the synthetic image moved toward — from `data/vocabularies/antonyms.csv`, e.g.
`lighting_quality` renders as "soft directional light" / "harsh overhead light" — because a bare cue
name does not say which direction the change went. Phrases wrap at 22 characters, which splits every
phrase in the vocabulary onto at most two lines. `--label-style name` restores cue ids, which is what
Fig 5 uses.

Ten variants are rendered; **`b6612206` is the one the manuscript prints and the only one whose data is shipped here.** Each panel also carries the **canonical detector's own score** for that
image — `z` and `P(synthetic)` — read from F5's ranking, i.e. the same matched probe whose extremes Fig 3
shows. Including the *real reference* is the point: the reader sees what the detector made of every
image whose cue changes are being described.

| variant | typicality | P(synthetic) on the reference | note |
|---|---|---|---|
| **`f3273211`** | 0.748 | **0.03** | **suggested** — reference confidently real and all three synthetics correctly called; content control obvious (one subject across all four); `harsh overhead light` negative in all three generators. Caveat: `correct facial anatomy` scores on a flower photograph — cues are not content-gated |
| `b6612206` | 0.764 | 0.19 | all four correct; dark source photo, and `watermark` appears among its cues |
| `5a5dc6fb` | 0.707 | 0.42 | all four correct but the reference is marginal |
| `18063501` | 0.746 | 0.56 | same tree-lined path in all four panels — the best content control of the set — but the detector calls the reference synthetic |
| 6 others | 0.698–0.759 | 0.50–0.78 | reference classified synthetic |

Typicality spans only 0.698–0.764 against a population median of **0.399**, so every variant is
equivalent under the stated selection rule; the choice is about legibility and about whether the
detector is right on the panels shown (a reference is correctly called real when `P(synthetic) < 0.5`).

> **A result, not just a selection criterion.** The canonical detector calls the **real** CLIC
> photograph synthetic in **7 of these 10** pairs. They are typical pairs by construction, not
> selected for difficulty, so this is evidence about the SynthCLIC regime rather than an artifact of
> the figure — worth a sentence in the text, since it is otherwise visible only in whichever variant
> is printed.

**Fig 5 — the two panels answer different questions.** Panel A is within-class association with the
detector's score; Panel B is the paired effect of synthesis. A cue can be large in one and absent
from the other, and reading B as "what the detector uses" is the specific error the figure exists to
prevent. Over the top-12 of each, only **3 cues** appear in both — a number the text should quote,
since the figure no longer marks the overlap. It stays in the exported CSV as `in_both_panels`.

The plot is deliberately plain: uniform markers, no cue-family colouring. Families are still computed
and exported, but drawing them needed a seven-category legend that competed with the two panels while
adding nothing to the comparison being made. Both layouts (`independent`, `shared`) are built; the manuscript prints — and this directory ships — `shared`.

**Rows are labelled with each cue's positive phrase** (`instant_camera_cues` -> "instant film
texture"), from the same `antonyms.csv` Fig 4 uses, so the two figures name cues identically. Because
a phrase alone still does not say which direction is which, each panel's axis carries pole
annotations — A: `← scored more real` / `scored more synthetic →`; B: `← less in synthetic` /
`more in synthetic →`. A row then reads end to end without decoding an identifier: *instant film
texture* at −0.30 in A means more of it is scored more **real**, and at −0.71 in B that synthetic
images have **less** of it. Cue ids remain in the CSV's `cue` column.

Labels are **not** flipped by sign (showing the negative phrase for negative values): that reads well
per cell but breaks the shared y-axis and misleads whenever a cue's two signs disagree. All ten cues
currently shown happen to agree in sign, which is what makes one phrase per row safe here.

Worth a sentence in the text: the signs line up into one story. Synthetic images carry **more** clean
low light, clean framing, high clarity and mirrorless capture cues, and **less** documentary look and
instant film texture — synthesis moves toward a polished, camera-ideal aesthetic rather than toward
any single artifact.

**Fig 6 — the annotated similarity is data-weighted.** cos_Σ = **−0.21**; the raw weight cosine on
the same normals is **−0.06**, and both are on the figure. E11a's conclusion was that boundaries must
be compared where the images actually lie.

**Fig 7 — an illustration of one trained concept model.** E8 found the population concept signal
diffuse and seed-sensitive, so individual concept names are qualitative; the claim is about the
*form* of the explanation.

Two variants, `--variant` selects:

- **`fig7-concept-importance-3panel`** — a faithful rebuild of the **published** figure
  (`plot_concept_importance_summary` in the archived research repo): class separation, activation
  probability, predictive power; concepts ordered by single-concept AUC; the same blue/orange
  real–synthetic dumbbells. Changed only as the plan allows — K 30 → 14, both datasets as rows of
  one figure. `--panels 2` drops the AUC column to a table.
- **`fig7-concept-model`** — the leaner alternative: mean-contribution bars with class-mean ticks
  and an activation strip.

> **Statistic note.** "Activation probability" in the original means the **usage rate** — the
> fraction of images with `a_q > 0.5` — *not* the mean of `a_q`. They differ sharply here: mean `a_q`
> exceeds 0.5 for essentially no concept, yet usage rates reach 0.79, because the gate fires
> decisively on *some* images. `concept_profile` now returns both.

Two things the figure shows that are worth a sentence in the text: CNNSpot's concepts are far more
separable than SynthCLIC's (AUC 0.83–0.95 vs 0.77–0.84), and on SynthCLIC several top concepts fire
**more on real images** (`glitch_artifacts` 0.29 real vs 0.03 synthetic, likewise `scan_artifacts`,
`motion_freeze`, `print_texture`), whereas CNNSpot's are almost all synthetic-indicating.

______________________________________________________________________

## Which variant is published

Where a figure has variants, all of them are built; the one the manuscript prints is the one shipped
here. The reason is stated so the choice can be checked.

| figure | recommended | why |
|---|---|---|
| Fig 4 | `b6612206` (printed) | one of only two candidates where the detector is correct on **all four** panels (P(synthetic) 0.19 on the real reference; the other is `f3273211` at 0.03). Typicality 0.764, the highest of the ten. Its caption should give the content id, and note that `watermark` appears among its cues — a known SynthCLIC collection shortcut |
| Fig 5 | `shared` | the figure exists to stop "changed by synthesis" being read as "used by the detector", and only the shared y-axis lets a row be read across both panels. Cost: `watermark absent` and `legible text` drop out of Panel A's top-6; the watermark cue connects to E12's corpus-shortcut finding, so quote it from `fig5-all-cues.csv` in the text |
| Fig 7 | `3panel` | the faithful rebuild of the published figure. `--panels 2` drops the AUC column if it is better as a table |
| Fig 9 | **9a** | the manuscript's Figure 10 is the attribute-distribution figure, so 9a is what ships. 9b (per-axis AUROC) remains a build target |

## Removal list — figures deliberately not carried over

Recorded so each absence is a decision rather than an oversight.

| removed | why |
|---|---|
| every **k=8 per-axis montage** (CNNSpot Dim1/Dim3, SynthCLIC Dim1/Dim3) | F2: individual axes agree across seeds at Hungarian-matched |cos| **0.289**, cue profile ρ **0.119**. Per-axis exemplars are not a reproducible object. Fig 3 replaces them by ranking on the actual detector logit, which *is* stable. |
| per-axis **activation / AUC** figure | same result |
| **dimension-contribution heatmap** | same result |
| the k ∈ {2,4,8,16} visual comparison | F2 reports it as a table; the visual added nothing the numbers did not |
| **t-SNE** of the embedding space | weaker than the quantitative results and adds nothing after the revision |
| **detector-performance** figure (3 mAP heatmaps) | tables give exact cross-dataset numbers more cleanly → `e1_e3_e6_e7_detector_comparison.tex` |
| **`D_h` / `D_e` / `D_cue`** performance figure | → [`tables/cascade-information-restriction.tex`](tables/cascade-information-restriction.tex) |
| **orthogonality / stability** figure | → [`tables/stability-summary.tex`](tables/stability-summary.tex) |
| **β-sensitivity** plot | the reviewer asked for a sensitivity *analysis*; E2's table answers it |
| **backbone-comparison** plot | E3's table answers it |
| Fig 6's 3×3 **similarity matrix** panel | simplified to one panel; the matrix (including SynthBuster+, which shows the near-orthogonality is not CNNSpot-specific) belongs in the text as a small table from `F6-cross-dataset/artifacts/boundary_cosines.csv` |

______________________________________________________________________

## Provenance and guards

Figure code lives in [`src/clip_cues_research/figures/`](../../../src/clip_cues_research/figures/) with
one thin driver per figure in [`scripts/plot/`](../../../scripts/plot/). The six-family cue taxonomy and
the save contract are in `figures/style.py` — one definition, so figures cannot drift apart.

**Presentation choices live in [`config/figures.yaml`](../../config/figures.yaml)**, which
`figures/style.py` is the only reader of: palette, colormaps, typography, output DPI. Changing the
palette is a config edit, not a code edit.

The palette is the **original paper's** — `color_palette_real_synthetic` from the archived research
repo's `config/plotting.yaml`, i.e. matplotlib `tab10` blue `#1f77b4` / orange `#ff7f0e`. It is more
saturated than seaborn's colorblind defaults, and the published Fig 7 already used it. Roles are
named rather than raw: `real`/`synthetic` for class contrasts, `positive`/`negative` for signed
quantities that are not class contrasts (Fig 4's cue shifts, Fig 6's boundary difference) mapped to
the same two hues, so **orange = toward synthetic** reads identically in every figure.

Panel and figure titles are Title Case via `style.title_case`, which is hand-written because
`str.title()` destroys exactly what these titles contain — it renders `CNNSpot` as "Cnnspot",
`AUROC` as "Auroc" and `real-like` as "Real-Like". Axis labels stay sentence case.

Seven things are pinned by [`tests/test_figures_canonical.py`](../../../tests/test_figures_canonical.py):

1. **No figure reads a retracted W² artifact.** The 2026-07-17 double-projection bug left
   `outputs/e8/paired/paired_cue_shifts_*` and `outputs/e8/stable_interp/*_cue_profile.csv` in a
   space that must not reach the paper. The guard inspects string *constants* (not docstrings), so
   modules can name the hazard in prose while a real path fails the build.
2. **Fig 4 ≡ Fig 5 Panel B.** Both are asserted equal to 1e-9 on the same pair, because a single
   example is only worth showing if it is provably an instance of the aggregate.
3. **Fig 6's annotation sides.** These inverted once during a layout change, which reverses the
   figure's central claim; the test asserts the data still supports the hard-coded placement.
4. **Fig 7's concept names.** The checkpoint carries its own `model.text_embeddings`; those are
   verified against `vocab_canon/antonyms.pt` (diagonal cosine 1.0000) *and* the shape-identical
   retracted vocabulary is asserted to be rejected.
5. **No figure hardcodes a colour.** A bare hex literal in figure code means a figure has gone its
   own way — the drift the config exists to stop. This caught four survivors on its first run,
   including `paired_cue_delta.py`, which was still on the pre-config palette.
6. **The palette matches the paper**, including that the signed roles reuse the class hues.
7. **Title Case preserves acronyms** — `SynthCLIC`, `CNNSpot`, `real-like`.
