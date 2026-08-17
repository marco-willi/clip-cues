# Reproduction

Everything needed to reproduce the paper, kept apart from the model package and its usage. Nothing
here is required to *use* a released detector — for that, see the top-level
[README](../README.md).

```bash
uv sync --extra all
make finalexp-fetch     # the checksummed input snapshot (~290 MB), verified on arrival
make finalexp-all       # F1-F7 end to end, CPU, a few minutes
```

| directory | what it holds |
| --- | --- |
| [`experiments/`](experiments/OVERVIEW.md) | the experiment ledger, the F1–F7 records with their `run_meta.json` provenance, the figure ledger with the exact plotted data, and the input snapshot's manifest |
| `config/` | the hyperparameters: per-experiment YAML, the training config, the β sweep, figure styling, and `mappings.yaml` (required by the Fig 3/8 caption caveats) |
| `revision_export/` | the write-up bundle — `\input`-able LaTeX tables, result CSVs and per-experiment details |

The code lives outside this directory: `src/clip_cues_research/` (the reproduction layer) and
`scripts/` (the drivers). That split is deliberate — the code is installed as a package, the material
here is data and records.

See [`docs/REPRODUCTION.md`](../docs/REPRODUCTION.md) for the matched training recipe, what needs a
GPU, and the expected values to check a run against. Before building any table, read the caveats
alongside the data — [`experiments/data/EXCLUDED.md`](experiments/data/EXCLUDED.md) for retracted and
superseded artifacts, and [`experiments/figures/README.md`](experiments/figures/README.md) for the
required per-figure caption caveats.
