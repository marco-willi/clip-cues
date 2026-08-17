"""E7: OwensLab/CommunityForensics dataset adapter (revision-only).

CommunityForensics ships a different schema than the repo's eval code expects. This adapter maps it
onto the canonical columns so ``scripts/utils/validate_model.py`` (CLIP heads, end-to-end),
``forensics/patch_cnn.evaluate`` (forensic CNN), and ``scripts/extract/extract_embeddings.py`` consume it
unchanged.

CommunityForensics fields → canonical:
    image_data (raw JPEG/PNG bytes) → ``image``   (HF ``Image`` feature → decoded PIL on access)
    image_name                      → ``image_id``
    model_name                      → ``source``   (paired real+fake group ⇒ per-source AP == per-generator)
    label                           → ``label``    (already real=0 / fake=1)
Extra metadata kept for the E7 breakdown tables: ``architecture``, ``real_source``, ``subset``, ``model_name``.

Notes:
    * Default is a **map-style** ``Dataset`` (indexable, has ``len``) — required by ``validate_model``'s
      slice indexing and by ``patch_cnn.evaluate``. Storage is not a constraint (Lambda A10 = 1.4 TB),
      so the full split is materialised.
    * ``streaming=True`` returns an ``IterableDataset`` (no ``len``/indexing) — only for cheap smoke
      tests, not the Option-A eval path.
"""

from __future__ import annotations

import io

from datasets import load_dataset
from PIL import Image as PILImage

CF_EVAL = "OwensLab/CommunityForensics-Eval"
CF_SMALL = "OwensLab/CommunityForensics-Small"


def is_community_forensics(name: str) -> bool:
    """True if ``name`` is a CommunityForensics HF id (so eval scripts can route to this adapter)."""
    return name.startswith("OwensLab/CommunityForensics")


def _normalize_batch(batch: dict) -> dict:
    """Decode image_data -> PIL and project onto the canonical + metadata columns."""
    return {
        "image": [PILImage.open(io.BytesIO(b)).convert("RGB") for b in batch["image_data"]],
        "image_id": list(batch["image_name"]),
        "label": list(batch["label"]),
        "source": list(batch["model_name"]),
        "architecture": list(batch["architecture"]),
        "real_source": list(batch["real_source"]),
        "subset": list(batch["subset"]),
    }


def load_community_forensics(
    name: str, split: str, cache_dir: str | None = None, streaming: bool = False
):
    """Load a CommunityForensics split with canonical columns.

    Args:
        name: HF id, e.g. ``OwensLab/CommunityForensics-Eval`` / ``-Small``.
        split: ``CompEval`` (Eval) or ``train`` (Small).
        cache_dir: HF cache dir (e.g. ``data/hf_cache``).
        streaming: if True return an IterableDataset (smoke tests only); default materialises the split.

    Returns:
        A dataset whose items expose ``image`` (PIL), ``image_id``, ``label`` (0/1), ``source``,
        plus ``architecture``/``real_source``/``subset``.

    Note:
        The map-style path uses ``with_transform`` to decode **lazily on access** rather than
        rewriting the image-bytes column through Arrow. On the full CompEval split a ``.map``/
        ``cast_column`` rewrite overflows pyarrow's 2 GB single-array offset limit
        (``ArrowInvalid: offset overflow while concatenating arrays``); the lazy transform never
        materialises the bytes into one array, so it is both correct and faster.
    """
    ds = load_dataset(name, split=split, cache_dir=cache_dir, streaming=streaming)
    if streaming:
        return ds.map(_normalize_batch, batched=True, remove_columns=["image_data"])
    return ds.with_transform(_normalize_batch)
