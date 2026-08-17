---
license: mit
task_categories:
  - image-classification
tags:
  - synthetic-image-detection
  - clip
  - interpretability
  - embeddings
  - reproducibility
pretty_name: CLIP-Cues reproducibility snapshot
size_categories:
  - 100K<n<1M
---

# CLIP-Cues — frozen input snapshot

Cached CLIP features and the checksummed inputs behind
**"Synthetic Image Detection with CLIP: Understanding and Assessing Predictive Cues"**
(Willi, Mathys & Graber). This repository holds no images: it is the set of frozen arrays that lets
every number and figure in the paper be reproduced **on a CPU in minutes**, without re-extracting
features from 33k photographs.

Code: <https://github.com/marco-willi/clip-cues> · Image datasets:
[`synthclic`](https://huggingface.co/datasets/marco-willi/synthclic) ·
[`synthbuster-plus`](https://huggingface.co/datasets/marco-willi/synthbuster-plus) ·
[`cnnspot-small`](https://huggingface.co/datasets/marco-willi/cnnspot-small)

## Use it

```bash
git clone https://github.com/marco-willi/clip-cues.git && cd clip-cues
uv sync --extra all
make finalexp-fetch     # downloads this repo and verifies every file
make finalexp-all       # F1-F7 end to end, a few minutes on CPU
```

`finalexp-fetch` checks each file against `release_manifest.json` before anything loads it, so a
truncated download or a swapped file fails loudly instead of silently changing a result.

## Contents (289 MB, 20 artifacts)

| path | what |
| --- | --- |
| `embeddings/pooler_l14/` | frozen CLIP ViT-L/14-336 `pooler_output`, 1024-d — SynthCLIC (10815), SynthBuster+ (13999), CNNSpot (8000). The space the canonical detector `D_h` lives in |
| `embeddings/projected_derived/` | the same three corpora in the 768-d shared image–text space, after the single visual projection `W_p` |
| `embeddings/cue_scores/` | per-image × per-cue cosines against the 168 antonym cue directions |
| `projection/` | `W_p`, the 768×1024 CLIP visual projection matrix |
| `vocabularies/` | the 168 antonym cue directions (`.npz`) and their term list (`.csv`) |
| `checkpoints/` | the three published heads used as bridge targets |
| `rankings/` | detector-score rankings, so the montage figures rebuild without shipping pixels |
| `reference/` | regression anchors and the derived-vs-cached crosscheck |

Governance files travel with the data: `MANIFEST.md` (every artifact with its sha256, shape,
declared vector space, origin and consumers), `manifest.json` (the machine-readable form), and
**`EXCLUDED.md`** — every plausible-but-wrong neighbour of a snapshot artifact, with the reason it is
*not* used.

## Format

Every array is an **object-free `.npz`**: it loads with `allow_pickle=False`, so nothing here
executes code on download, and it is portable across Python versions. Embedding frames pack the
metadata columns alongside the matrix (`embeddings`, `columns`, `col__*`):

```python
import numpy as np
z = np.load("embeddings/pooler_l14/synthclic.npz", allow_pickle=False)
emb = z["embeddings"]                       # (10815, 1024) float32
split = z["col__split"]                     # train / validation / test
label = z["col__label"]                     # 1 = synthetic
```

`release_manifest.json` records two hashes per artifact: `sha256` is the file as *built* — the
provenance anchor every `run_meta.json` in the paper's experiment records cites — and
`release_sha256` is the distributed file, which is what a download is verified against. They differ
exactly where `converted` is `true`.

## Three things to carry into any table you build from this

1. **"mAP" is overloaded.** The code's default is *pooled* AP over all test images; the paper's
   tables report *per-generator mean* AP. They are different numbers — state which one you mean.
2. **CNNSpot has two evaluation frames.** The arrays here are the 4,000-image / 20-generator
   evaluation frame, a strict subset of the 108,310-image benchmark test split used by the appendix
   per-generator table.
3. **CNNSpot's `source` column is an evaluation subset, not a provenance.** A real photograph filed
   under `progan` means *the real half of the ProGAN evaluation subset* — not that ProGAN produced it.

## What is deliberately absent

Text embeddings written before 2026-07-17 are in a **double-projected ("W²") space** — a fixed bug —
and are *shape-identical* to the correct ones, so they cannot be told apart by inspection. They are
excluded here, and `EXCLUDED.md` names each one. Also absent: raw images (see the dataset
repositories above), third-party detector weights, and the B/16–B/32 backbone caches.

## Citation

```bibtex
@article{willi2026synthetic,
  title={Synthetic Image Detection with CLIP: Understanding and Assessing Predictive Cues},
  author={Willi, Marco and Mathys, Melanie and Graber, Michael},
  year={2026}
}
```

The MIT license covers these derived arrays and the accompanying code. The photographs they were
computed from remain subject to their own licenses.
