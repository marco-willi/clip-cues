# Reproduction

Every number in the paper comes from linear heads over **frozen** CLIP features. That makes the
expensive part a one-time GPU pass and everything after it cheap: the full methodological
consolidation runs on a laptop CPU in a few minutes.

```bash
uv sync --extra all
make finalexp-fetch     # ~290 MB of cached features, verified against their sha256
make finalexp-verify    # re-check every artifact (finalexp-all does this for you)
make finalexp-all       # F1-F7 end to end
```

## What you get, and what it costs

| step | hardware | wall time |
| --- | --- | --- |
| `make finalexp-fetch` | — | a few minutes of download |
| `make finalexp-all` (F1–F7) | CPU | ~15 min total; F6's lasso decomposition dominates at ~830 s |
| a single F-experiment (`make finalexp-f3`) | CPU | seconds to ~1 min |
| training one linear head (5 seeds) | CPU | 2–6 s per seed |
| `make figures-all` | CPU, plus image cache for Figs 3/4/8 | minutes |
| extracting embeddings from images | **GPU** | one pass over ~33k images per backbone |

Figures 3, 4 and 8 decode raw pixels, so they download the image datasets into `data/hf_cache`
(the `make` targets set `HF_HOME` accordingly). Every other figure is built from the snapshot.

## The matched recipe

F1–F4 and F6 all train the same way and differ **only** in input dimension (1024 / 768 / 168). The
recipe — every hyperparameter, plus the one deviation that matters — is defined once in
[`reproduction/experiments/final_consolidation/README.md`](../reproduction/experiments/final_consolidation/README.md),
the task file that owns F1–F7.

The deviation is worth knowing before you read any result: feeding unit-normalized projected features
`e = Ph/‖Ph‖` **breaks the comparison**, because a fixed weight decay is only "the same
regularization" when the spaces have comparable scale. Uncorrected it drops the projected head from
0.888 to 0.725 AUROC, which made a *restricted* probe outscore the unrestricted probe it is a strict
subspace of. Every space is therefore rescaled by one global scalar to the pooler training split's
mean row norm — geometry preserved exactly, no per-dimension statistics, so it is not
standardization.

## Provenance

Two layers, both mandatory:

- **Inputs.** Every artifact F1–F7 reads is in `reproduction/experiments/data/`, checksummed, tagged with its
  embedding space, and reachable only through `clip_cues_research.finalexp.data.get_*`, which
  verifies the sha256 **on load** and asserts the declared space. A guard test fails the build if any
  F-code reaches for a literal `data/...` path.
- **Code.** Every run directory carries `run_meta.json`: the script, full argv, git commit and dirty
  flag, the snapshot manifest version, `{input_id: sha256}` for everything read, package versions,
  host and wall time.

## GPU steps

Only two things need a GPU, and neither is required to reproduce the paper's main results.

1. **Embedding extraction** (`scripts/extract/`) — the one-time pass that produces the cached
   features. The published snapshot is exactly this output, so you only need it for new images or a
   new backbone.
2. **The additional-backbone table** (ViT-B/16 and ViT-B/32, appendix). Those caches are **not**
   published — the paper's claims rest on ViT-L/14-336, and the B/16–B/32 arrays are ~215 MB of
   material that supports one appendix table. Reproducing that table means re-extracting with
   `scripts/extract/extract_clip_embeddings.py` for each backbone, then running the probe.

## Rebuilding the snapshot instead of fetching it

`make finalexp-data` rebuilds `reproduction/experiments/data/` from the original extraction outputs. It is the
maintainer path, not the reproduction path, and it needs inputs that are not published (raw
extraction caches, one E3 run output). Two things to know:

- The builder **refuses to run** if `make finalexp-fetch` has written the legacy
  `data/embeddings/` paths. Those copies hold identical numbers but different bytes, so rebuilding
  from them would silently re-hash every artifact and break the `derived_from` chain. Delete
  `data/embeddings/.fetched-mirror` and restore the original extraction outputs first.
- `make finalexp-release` converts a built snapshot into the distributed, object-free `.npz` form,
  writing `release_manifest.json` with both hashes per artifact.

## Checking your run

`make finalexp-all` prints each experiment's headline. Against the published values:

| | expected |
| --- | --- |
| F1 direction Σ-cos / cue-profile ρ | 0.989 / 0.991 |
| F2 k=8 effective direction / individual axes | 0.963 / 0.289 |
| F3 `D_h` 0.921 vs `D_e` 0.893; projection cost | +0.029 [+0.021, +0.037] |
| F4 cue-restricted delta; excess recovery | −0.059 [−0.075, −0.043]; 0.85 |
| F5 top-50 seed overlap (SynthCLIC / CNNSpot) | 0.73 / 0.66 |
| F6 boundary cos_Σ (SynthCLIC ~ CNNSpot) | −0.21 |
| F7 both primary bridge pairs; augmentation effect | pass; +0.007 |

Small differences in the last digit are expected across BLAS builds; a difference in the second
decimal is not, and means an input changed.
