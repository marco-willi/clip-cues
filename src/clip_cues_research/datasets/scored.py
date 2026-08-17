"""Uniform per-image eval splits for scoring detectors (E6/E7), revision-only.

Returns a map-style dataset exposing the canonical columns ``score_cf_split`` consumes —
``image`` (PIL), ``image_id``, ``label`` (real=0/fake=1), ``source`` (generator), and the metadata
``architecture``/``real_source``/``subset`` — for ANY of: CommunityForensics-Eval (via the CF
adapter), or the project datasets SynthCLIC / SynthBuster+ / CNNSpot (via ``get_dataset``, with the
metadata columns filled in where a dataset doesn't have them). Local ``data/datasets/<name>`` is used
when present (fast, offline); otherwise the dataset is pulled from HuggingFace (e.g. on the box).
"""

from __future__ import annotations

import os

from datasets import load_from_disk

from clip_cues.dataset import get_dataset
from clip_cues_research.datasets.community_forensics import (
    is_community_forensics,
    load_community_forensics,
)

# short name -> local saved-dataset dir (sync excludes data/, so these exist locally only)
_LOCAL_DIR = {
    "synthclic": "data/datasets/synthclic",
    "synthbuster-plus": "data/datasets/synthbuster-plus",
    "synthbuster_plus": "data/datasets/synthbuster-plus",
}
_DEFAULTS = {"source": "unknown", "architecture": "na", "real_source": "na"}


def as_scored_split(name: str, split: str = "test", cache_dir: str = "data/hf_cache"):
    """Map-style split with canonical {image, image_id, label, source, architecture, real_source, subset}."""
    if is_community_forensics(name):
        return load_community_forensics(name, split, cache_dir=cache_dir)

    local = _LOCAL_DIR.get(name)
    dd = (
        load_from_disk(local)
        if local and os.path.isdir(local)
        else get_dataset(name, cache_dir=cache_dir)
    )
    ds = dd[split]
    n = len(ds)
    if "image_id" not in ds.column_names:
        ds = ds.add_column("image_id", [f"{name}_{i}" for i in range(n)])
    for col, default in _DEFAULTS.items():
        if col not in ds.column_names:
            ds = ds.add_column(col, [default] * n)
    if "subset" not in ds.column_names:
        ds = ds.add_column("subset", [name] * n)
    return ds
