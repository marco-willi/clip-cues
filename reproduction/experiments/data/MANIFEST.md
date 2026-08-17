# `reproduction/experiments/data/` — input snapshot manifest

> **Version 1.0.0** · built 2026-08-16T19:24:01+00:00 · git `ad9eea9a0f28` (dirty)
> Builder: `scripts/finalexp/build_data_snapshot.py` · Plan: `.claude/plans/PLAN_FINAL_CONSOLIDATION.md`

Generated file — **do not edit by hand**; rebuild with `make finalexp-data`.

Every F1–F7 input lives here and is reached only through
`clip_cues_research.finalexp.data.get_*`, which verifies the sha256 on load and asserts the
declared embedding `space`. See [EXCLUDED.md](EXCLUDED.md) for what was deliberately left
out and why.

**Spaces:** `pooler_1024` = frozen CLIP ViT-L/14-336 `pooler_output` · `crossmodal_768_canon` = shared image–text space, canonical (post-2026-07-17 fix) · `n/a` = not an embedding.

## Artifacts

| id | path | kind | space | shape | sha256 (short) | MB | used by |
|---|---|---|---|---|---|---|---|
| `ckpt/clip_orthogonal_synthclic` | `checkpoints/clip_orthogonal_synthclic.ckpt` | checkpoint | `n/a` | — | `0ddba778343c2018` | 0.04 | F7 |
| `ckpt/linear_probe_cnnspot` | `checkpoints/linear_probe_cnnspot.ckpt` | checkpoint | `n/a` | — | `b8201afaaceda70b` | 0.01 | F7 |
| `ckpt/linear_probe_synthclic` | `checkpoints/linear_probe_synthclic.ckpt` | checkpoint | `n/a` | — | `c7a310eb0b14290d` | 0.01 | F7 |
| `cue_scores/cnnspot__antonyms` | `embeddings/cue_scores/cnnspot__antonyms.npz` | cue_scores | `n/a` | 8000x168 | `bec2249c8c30948d` | 4.98 | F4, F1, F2, F3, F6, F7 |
| `cue_scores/synthbuster-plus__antonyms` | `embeddings/cue_scores/synthbuster-plus__antonyms.npz` | cue_scores | `n/a` | 13999x168 | `d0bca50475192b44` | 8.71 | F4, F1, F2, F3, F6, F7 |
| `cue_scores/synthclic__antonyms` | `embeddings/cue_scores/synthclic__antonyms.npz` | cue_scores | `n/a` | 10815x168 | `dd9f94957ae2c5a9` | 6.73 | F4, F1, F2, F3, F6, F7 |
| `pooler/cnnspot` | `embeddings/pooler_l14/cnnspot.pkl` | pooler_embeddings | `pooler_1024` | 8000x1024 | `c542d7976a3f7c1d` | 33.26 | F5, F6 |
| `pooler/synthbuster-plus` | `embeddings/pooler_l14/synthbuster-plus.pkl` | pooler_embeddings | `pooler_1024` | 13999x1024 | `ba6472329703271c` | 58.18 | F6 |
| `pooler/synthclic` | `embeddings/pooler_l14/synthclic.pkl` | pooler_embeddings | `pooler_1024` | 10815x1024 | `4aebc7adfb9f8460` | 44.96 | F1, F2, F3, F4, F5, F6, F7 |
| `projected/cnnspot` | `embeddings/projected_derived/cnnspot.pkl` | projected_embeddings | `crossmodal_768_canon` | 8000x768 | `2cf119317d81eb52` | 25.07 | F3, F4, F6 |
| `projected/synthbuster-plus` | `embeddings/projected_derived/synthbuster-plus.pkl` | projected_embeddings | `crossmodal_768_canon` | 13999x768 | `2708e81a7c0e48c9` | 43.84 | F3, F4, F6 |
| `projected/synthclic` | `embeddings/projected_derived/synthclic.pkl` | projected_embeddings | `crossmodal_768_canon` | 10815x768 | `b40f00c48fc901b6` | 33.88 | F3, F4, F6 |
| `projection/wp_l14_336` | `projection/clip_l14_336_visual_projection.npy` | projection_matrix | `n/a` | 768x1024 | `56b0a8ba23d0e783` | 6.29 | F3, F4, F6 |
| `ranking/f5_cnnspot` | `rankings/f5_cnnspot.csv` | ranking | `n/a` | — | `869e1293d9127db0` | 0.26 | F5 |
| `ranking/f5_synthclic` | `rankings/f5_synthclic.csv` | ranking | `n/a` | — | `76b724d2a7e2c35c` | 0.15 | F5 |
| `reference/e3_seed123` | `reference/e3_xdataset_synthclic_seed123_metrics.json` | reference_metrics | `n/a` | — | `29507d6f2a83f813` | 0.00 | F1 |
| `reference/projected_cached_cnnspot` | `reference/projected_cached_cnnspot.pkl` | projected_embeddings_reference | `crossmodal_768_canon` | 4000x768 | `500eb7e8e631e779` | 12.62 | F3-crosscheck |
| `reference/projected_cached_synthclic` | `reference/projected_cached_synthclic.pkl` | projected_embeddings_reference | `crossmodal_768_canon` | 10815x768 | `e2f27a68fe38174f` | 33.88 | F3-crosscheck |
| `vocab/antonyms` | `vocabularies/antonyms.pt` | cue_vocabulary | `crossmodal_768_canon` | 168x768 | `394916babf0a76b6` | 0.52 | F1, F2, F3, F4, F6, F7 |
| `vocab/clipiqa_full` | `vocabularies/clipiqa_full.pt` | cue_vocabulary | `crossmodal_768_canon` | 16x768 | `f08fa821b39ca38b` | 0.05 | Fig9a |
| `vocab/clipiqa_full_poles` | `vocabularies/clipiqa_full_poles.pt` | cue_vocabulary | `crossmodal_768_canon` | 32x768 | `d232033175536405` | 0.10 | Fig9a |
| `vocab_terms/antonyms` | `vocabularies/antonyms.csv` | vocabulary_terms | `n/a` | — | `62c10a56b9d4f466` | 0.03 | F1, F2, F3, F4, F6, F7 |

## Provenance

### `ckpt/clip_orthogonal_synthclic`

- **File:** `checkpoints/clip_orthogonal_synthclic.ckpt` (0.04 MB)
- **sha256:** `0ddba778343c20186c6d3ab06198a956da3572058806cfbbf287a275f97ca5e4`
- **Space:** `n/a`
- **Copied from:** `data/checkpoints/clip_orthogonal_synthclic.ckpt`
- **Origin:** PUBLISHED k=8 ActivationOrthogonalityHead (augmented); F7 bridge target only

### `ckpt/linear_probe_cnnspot`

- **File:** `checkpoints/linear_probe_cnnspot.ckpt` (0.01 MB)
- **sha256:** `b8201afaaceda70bf0dbfc3ffc2107dddbda49047c73557ae316cd5796c194ee`
- **Space:** `n/a`
- **Copied from:** `data/checkpoints/linear_probe_cnnspot.ckpt`
- **Origin:** PUBLISHED CNNSpot k=1 probe (augmented); F7 bridge target only

### `ckpt/linear_probe_synthclic`

- **File:** `checkpoints/linear_probe_synthclic.ckpt` (0.01 MB)
- **sha256:** `c7a310eb0b14290d6ede87c85505ecc0b8522f9618ec7b21e1a428334c4d749f`
- **Space:** `n/a`
- **Copied from:** `data/checkpoints/linear_probe_synthclic.ckpt`
- **Origin:** PUBLISHED k=1 probe, trained end-to-end WITH augmentation (RandomResizedCrop 0.5-1.0 -> 512 + HFlip + JPEG 65-100); F7 bridge target only

### `cue_scores/cnnspot__antonyms`

- **File:** `embeddings/cue_scores/cnnspot__antonyms.npz` (4.98 MB)
- **sha256:** `bec2249c8c30948d92fbc77ac955d2ce1ccee31351bb8148ca2d505cada585b2`
- **Space:** `n/a`
- **Derived from:** `projected/cnnspot` (`2cf119317d81`), `vocab/antonyms` (`394916babf0a`)
- **Origin:** scripts/finalexp/prepare_features.py — c_j = <e/||e||, v_j> on the derived projected frame against the CANONICAL cue basis (never the retracted W-squared vocabularies).

### `cue_scores/synthbuster-plus__antonyms`

- **File:** `embeddings/cue_scores/synthbuster-plus__antonyms.npz` (8.71 MB)
- **sha256:** `d0bca50475192b441976a098316c8b1685ff6d6d14cb539d895141509e1f1c18`
- **Space:** `n/a`
- **Derived from:** `projected/synthbuster-plus` (`2708e81a7c0e`), `vocab/antonyms` (`394916babf0a`)
- **Origin:** scripts/finalexp/prepare_features.py — c_j = <e/||e||, v_j> on the derived projected frame against the CANONICAL cue basis (never the retracted W-squared vocabularies).

### `cue_scores/synthclic__antonyms`

- **File:** `embeddings/cue_scores/synthclic__antonyms.npz` (6.73 MB)
- **sha256:** `dd9f94957ae2c5a9ddbd6ba2feef60e65c4b63aec6e9bfac2e964f15bc28105d`
- **Space:** `n/a`
- **Derived from:** `projected/synthclic` (`b40f00c48fc9`), `vocab/antonyms` (`394916babf0a`)
- **Origin:** scripts/finalexp/prepare_features.py — c_j = <e/||e||, v_j> on the derived projected frame against the CANONICAL cue basis (never the retracted W-squared vocabularies).

### `pooler/cnnspot`

- **File:** `embeddings/pooler_l14/cnnspot.pkl` (33.26 MB)
- **sha256:** `c542d7976a3f7c1de4f6c234aa987366d3fee6a2c76b90a583ba2540c3f12ec2`
- **Space:** `pooler_1024`
- **Copied from:** `data/embeddings/cnnspot_clip_large_patch14.pkl`
- **Origin:** scripts/extract/extract_embeddings.py; train split = ds_train_very_small (2,000), confirmed 2026-08-08 (config-audit.md §A)
- **Splits:** test 4000, train 2000, validation 2000

### `pooler/synthbuster-plus`

- **File:** `embeddings/pooler_l14/synthbuster-plus.pkl` (58.18 MB)
- **sha256:** `ba6472329703271c00b1bd8bdb213af4fc864da2f8dbbc410e78c2ed5d4f1e95`
- **Space:** `pooler_1024`
- **Copied from:** `data/embeddings/synthbuster-plus_clip_large_patch14.pkl`
- **Origin:** scripts/extract/extract_embeddings.py; F6 uses train/val ONLY (SB+ test is closed under EXTERNAL_VALIDATION_PROTOCOL.md)
- **Splits:** test 2800, train 8960, validation 2239

### `pooler/synthclic`

- **File:** `embeddings/pooler_l14/synthclic.pkl` (44.96 MB)
- **sha256:** `4aebc7adfb9f84606e3861a802e4a60701a93b36fc4c7f89874c813e5f17b0ee`
- **Space:** `pooler_1024`
- **Copied from:** `data/embeddings/synthclic_clip_large_patch14.pkl`
- **Origin:** scripts/extract/extract_embeddings.py, frozen CLIP ViT-L/14-336 pooler_output; the canonical SynthCLIC frame behind results/e3_xdataset and E9-E12
- **Splits:** test 2140, train 8165, validation 510

### `projected/cnnspot`

- **File:** `embeddings/projected_derived/cnnspot.pkl` (25.07 MB)
- **sha256:** `2cf119317d81eb528220f320fe113058c7a9736a561696fcc1a82ce7e5013842`
- **Space:** `crossmodal_768_canon`
- **Derived from:** `pooler/cnnspot` (`c542d7976a3f`), `projection/wp_l14_336` (`56b0a8ba23d0`)
- **Origin:** scripts/finalexp/prepare_features.py — derived as e = Wp h from the cached pooler frame (both-sides-derived rule, EXTERNAL_VALIDATION_PROTOCOL.md). NOT a separate extraction: D_h and D_e therefore see the same image representation and differ only by the projection.
- **Splits:** test 4000, train 2000, validation 2000

### `projected/synthbuster-plus`

- **File:** `embeddings/projected_derived/synthbuster-plus.pkl` (43.84 MB)
- **sha256:** `2708e81a7c0e48c983d53abee1843af479e0639e1a4f159b5f04b64b0e3277f3`
- **Space:** `crossmodal_768_canon`
- **Derived from:** `pooler/synthbuster-plus` (`ba6472329703`), `projection/wp_l14_336` (`56b0a8ba23d0`)
- **Origin:** scripts/finalexp/prepare_features.py — derived as e = Wp h from the cached pooler frame (both-sides-derived rule, EXTERNAL_VALIDATION_PROTOCOL.md). NOT a separate extraction: D_h and D_e therefore see the same image representation and differ only by the projection.
- **Splits:** test 2800, train 8960, validation 2239

### `projected/synthclic`

- **File:** `embeddings/projected_derived/synthclic.pkl` (33.88 MB)
- **sha256:** `b40f00c48fc901b641a67252e9cfeca6ae0a7601f6de31169ca163fcc6f51524`
- **Space:** `crossmodal_768_canon`
- **Derived from:** `pooler/synthclic` (`4aebc7adfb9f`), `projection/wp_l14_336` (`56b0a8ba23d0`)
- **Origin:** scripts/finalexp/prepare_features.py — derived as e = Wp h from the cached pooler frame (both-sides-derived rule, EXTERNAL_VALIDATION_PROTOCOL.md). NOT a separate extraction: D_h and D_e therefore see the same image representation and differ only by the projection.
- **Splits:** test 2140, train 8165, validation 510

### `projection/wp_l14_336`

- **File:** `projection/clip_l14_336_visual_projection.npy` (6.29 MB)
- **sha256:** `56b0a8ba23d0e783723dc738a60dc99a789b8e3457f031fddb8f83c75ef191ed`
- **Space:** `n/a`
- **Copied from:** `data/embeddings/clip_l14_336_visual_projection.npy`
- **Origin:** openai/clip-vit-large-patch14-336 visual_projection weight (768x1024); image_embeds = visual_projection(pooler_output) exactly

### `ranking/f5_cnnspot`

- **File:** `rankings/f5_cnnspot.csv` (0.26 MB)
- **sha256:** `869e1293d9127db08b8cd8606f4b9438e24119dbf3ed9b59d6be255568aec48f`
- **Space:** `n/a`
- **Derived from:** `pooler/cnnspot` (`c542d7976a3f`)
- **Origin:** scripts/finalexp/export_f5_rankings.py — test images ranked by the MATCHED canonical probe (F1 cnnspot seed 123) logit z = w.h + b. Stands in for the montage's image pixels, which are not snapshottable; the figure is reproducible from this ranking plus the HF dataset id + revision.

### `ranking/f5_synthclic`

- **File:** `rankings/f5_synthclic.csv` (0.15 MB)
- **sha256:** `76b724d2a7e2c35c98e8630f7be0f7d930776883a95e8bda0a4b3aadb37ed33a`
- **Space:** `n/a`
- **Derived from:** `pooler/synthclic` (`4aebc7adfb9f`)
- **Origin:** scripts/finalexp/export_f5_rankings.py — test images ranked by the MATCHED canonical probe (F1 synthclic seed 123) logit z = w.h + b. Stands in for the montage's image pixels, which are not snapshottable; the figure is reproducible from this ranking plus the HF dataset id + revision.

### `reference/e3_seed123`

- **File:** `reference/e3_xdataset_synthclic_seed123_metrics.json` (0.00 MB)
- **sha256:** `29507d6f2a83f8135197d7ae3af2ba18da5e68394a2ef63ac5ee13db51486dc5`
- **Space:** `n/a`
- **Copied from:** `results/e3_xdataset/clip_large_patch14__synthclic__to__synthclic/202606242002/metrics.json`
- **Origin:** scripts/run/run_linear_probe.py seed 123 under the matched recipe — F1's regression anchor (mAP 0.9239, AUROC 0.9227); the number in the manuscript's Table A
- **Content:** `{"backbone": "clip_large_patch14", "train_dataset": "synthclic", "eval_dataset": "synthclic", "input_dim": 1024, "val/auroc": 0.9889705882352942, "mAP": 0.9238769662881319, "pooled_ap": 0.9780796453961101, "auroc": 0.9226761070835879, "real_pairing": "shared", "n_generators": 4}`

### `reference/projected_cached_cnnspot`

- **File:** `reference/projected_cached_cnnspot.pkl` (12.62 MB)
- **sha256:** `500eb7e8e631e779b3d4066d52cff935a3434e36afdd8bb0dd20319e0ae24238`
- **Space:** `crossmodal_768_canon`
- **Copied from:** `data/embeddings/cnnspot_projected_embeddings.pkl`
- **Origin:** SEPARATELY EXTRACTED projected embeddings; CROSS-CHECK ONLY
- **Splits:** train 2000, validation 2000

### `reference/projected_cached_synthclic`

- **File:** `reference/projected_cached_synthclic.pkl` (33.88 MB)
- **sha256:** `e2f27a68fe38174fe430e4c2de0a08edeb7d1bcff3029ba2bb2712957c2f3766`
- **Space:** `crossmodal_768_canon`
- **Copied from:** `data/embeddings/synthclic_projected_embeddings.pkl`
- **Origin:** SEPARATELY EXTRACTED projected embeddings (CLIPVisionModelWithProjection). CROSS-CHECK ONLY — never fitted on; F3/F4/F6 use the derived features (both-sides-derived rule)
- **Splits:** test 2140, train 8165, validation 510

### `vocab/antonyms`

- **File:** `vocabularies/antonyms.pt` (0.52 MB)
- **sha256:** `394916babf0a76b62528f0c351f73250415cd6349128a9d613a2c47101736f8a`
- **Space:** `crossmodal_768_canon`
- **Copied from:** `data/embeddings/vocab_canon/antonyms.pt`
- **Origin:** scripts/interpret/embed_vocab.py, CANONICAL re-embed after the 2026-07-17 double-projection fix; the paper's published 168-cue antonym set
- **Vocabulary size:** 168

### `vocab/clipiqa_full`

- **File:** `vocabularies/clipiqa_full.pt` (0.05 MB)
- **sha256:** `f08fa821b39ca38bd147499790dad104d7819879df4386f2eba6ef06bbc6ccbd`
- **Space:** `crossmodal_768_canon`
- **Copied from:** `data/embeddings/vocab_canon/clipiqa_full.pt`
- **Origin:** the 16 CLIP-IQA attribute directions (Wang et al. 2022) behind appendix Figure 10; canonical text space, re-embedded after the 2026-07-17 double-projection fix
- **Vocabulary size:** 16

### `vocab/clipiqa_full_poles`

- **File:** `vocabularies/clipiqa_full_poles.pt` (0.10 MB)
- **sha256:** `d23203317553640558c2eb150cb4751494a57490ddc29487966b39ebcbd0592a`
- **Space:** `crossmodal_768_canon`
- **Copied from:** `data/embeddings/vocab_canon/clipiqa_full_poles.pt`
- **Origin:** the 32 positive/negative pole embeddings of the same 16 CLIP-IQA attributes; figures/clipiqa.py reads both this and vocab/clipiqa_full
- **Vocabulary size:** 32

### `vocab_terms/antonyms`

- **File:** `vocabularies/antonyms.csv` (0.03 MB)
- **sha256:** `62c10a56b9d4f46623cd84564bb380138d9e1a59e6ba3448b928ddedb20be9de`
- **Space:** `n/a`
- **Copied from:** `data/vocabularies/antonyms.csv`
- **Origin:** term list + poles for vocab/antonyms (row order matches the .pt)

