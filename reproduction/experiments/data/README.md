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

`finalexp-fetch` verifies every file against `release_manifest.json` before anything loads it.
`main` is snapshot **1.1.0**. The paper's original release is tagged `v1.0.0`.

## Contents (289 MB, 22 artifacts)

| path | what |
| --- | --- |
| `embeddings/pooler_l14/` | frozen CLIP ViT-L/14-336 `pooler_output`, 1024-d — SynthCLIC (10815), SynthBuster+ (13999), CNNSpot (8000) |
| `embeddings/projected_derived/` | the same three corpora in the 768-d shared image–text space |
| `embeddings/cue_scores/` | per-image × per-cue cosines against the 168 antonym cue directions |
| `projection/` | `W_p`, the 768×1024 CLIP visual projection matrix |
| `vocabularies/` | the 168 antonym cue directions (`.npz`) and their term list (`.csv`) |
| `checkpoints/` | the three published heads used as bridge targets |
| `rankings/` | detector-score rankings for the montage figures |
| `reference/` | regression anchors and the derived-vs-cached crosscheck |

`MANIFEST.md` lists every artifact with its sha256 and origin. `EXCLUDED.md` lists files that were
deliberately left out, and why.

### Optional: forensic CNN baselines (`forensic_cnn/`, 286 MB)

Fetched only with `make finalexp-fetch WITH_FORENSIC=1`.

| path | what |
| --- | --- |
| `forensic_cnn/cnnspot_{synthclic,synthbuster,combined}.safetensors` | retrained ResNet-50 baselines (E1/E7); load with `clip_cues_research.forensics.patch_cnn.build_model(path)` |
| `forensic_cnn/cfeval_predictions/cnnspot_synthclic.parquet` | per-image CommunityForensics-Eval scores of the SynthCLIC CNN |
| `forensic_cnn/cfeval_predictions/cnnspot_synthclic_smoketest.parquet` | **historical:** the scores behind the paper's CF-Eval row for the SynthCLIC CNN, which came from a 4-step smoke-test checkpoint. See [issue #1](https://github.com/marco-willi/clip-cues/issues/1) |

## Format

Arrays are object-free `.npz` (load with `allow_pickle=False`); embedding frames store metadata
columns as `col__*`:

```python
import numpy as np
z = np.load("embeddings/pooler_l14/synthclic.npz", allow_pickle=False)
emb, label = z["embeddings"], z["col__label"]   # (10815, 1024) float32; 1 = synthetic
```

Note: the paper's tables report *per-generator mean* AP; the code's default `mAP` is *pooled* AP.

## Citation

```bibtex
@article{willi2026synthetic,
  title={Synthetic Image Detection with CLIP: Understanding and Assessing Predictive Cues},
  author={Willi, Marco and Mathys, Melanie and Graber, Michael},
  journal={Array},
  pages={101222},
  year={2026},
  month={sep},
  doi={10.1016/j.array.2026.101222}
}
```

MIT license for these derived arrays and the code; the source photographs keep their own licenses.
