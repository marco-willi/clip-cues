# scripts/

Thin drivers. The logic lives in `src/clip_cues_research/`; everything here parses arguments, calls
into the package and writes an artifact. **Prefer `make` targets** (`make help`) — they set the
environment (notably `HF_HOME`) that several scripts assume.

| directory | what it drives | entry point |
| --- | --- | --- |
| `finalexp/` | **F1–F7**, the matched-recipe consolidation on the frozen snapshot | `make finalexp-all` |
| `plot/` | the manuscript figure set | `make figures-all` |
| `run/` | the E-experiments: baselines, sweeps, backbones, stability | per script |
| `interpret/` | E11 boundary decomposition, E12 score alignment, cue-vocabulary embedding | per script |
| `analyze/` | E8 mechanism analyses feeding the interpretation section | per script |
| `extract/` | the one-time **GPU** pass that produces the cached CLIP features | per script |
| `export/` | LaTeX tables and the write-up bundle | `make package-revision` |
| `utils/` | dataset download, checkpoint validation, publication gates | `make check-publication` |

What separates the F- and E-series — and why their numbers do not correspond — is in
[`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md).

## F1–F7 — the reproduction path

The only path that runs end to end from published inputs:

```bash
make finalexp-fetch     # the checksummed snapshot (~290 MB), verified on arrival
make finalexp-all       # verify, then F1-F7 on CPU
```

Each driver writes a `run_meta.json` next to its results recording the script, argv, git commit, the
sha256 of every input read, package versions, host and wall time. `build_data_snapshot.py`,
`prepare_features.py` and `export_snapshot_release.py` are the **maintainer** path that produces that
snapshot; they need extraction outputs that are not published. See
[`docs/REPRODUCTION.md`](../docs/REPRODUCTION.md).

## Figures

One driver per figure; the shared style and the cue-family taxonomy live in
`src/clip_cues_research/figures/style.py`. Every driver writes `png + pdf + the source CSV`, so a
number in a figure can be checked without re-rendering it.

Figures 3, 4 and 8 decode raw pixels and need the HuggingFace image cache; the `make` targets set
`HF_HOME=data/hf_cache` for you. Every other figure reads only the checksummed snapshot, so it cannot
drift from the F-experiment numbers. Each figure's claim, source and **required caption caveat** is
in [`reproduction/experiments/figures/README.md`](../reproduction/experiments/figures/README.md).

## Export

Run in this order — step 1 wipes the bundle, the rest are additive:

1. `package_revision_export.py` — core tables and figures (**destructive**)
2. `rebuild_export_local.py` — regenerates tables from local results
3. `export_per_generator_table.py` — adds the appendix per-generator table; run `make finalexp-tableb` first if its inputs are stale

These read run outputs under `outputs/`, which is a build directory, not repository content — so the
export path only works on a machine that has actually run the experiments it bundles.

## GPU

Only `extract/` needs one. The published snapshot *is* its output, so you need this path only for new
images or a new backbone — see [`docs/REPRODUCTION.md`](../docs/REPRODUCTION.md#gpu-steps).
`utils/train_concept_model_synthclic.sh` chains extraction and concept-model training for one corpus.
