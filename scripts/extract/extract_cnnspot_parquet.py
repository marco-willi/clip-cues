#!/usr/bin/env python
"""Extract CNNSpot embeddings directly from the already-downloaded HF parquet shards.

The disk can't fit HF's `download_and_prepare` Arrow copy of the 108k-image cnnspot-small test
(~30 GB peak vs ~4 GB free). The parquet is already cached, so we read it **directly** shard-by-shard
(bounded RAM, no Arrow copy), decode images from bytes, run the frozen CLIP encoder, and save the
same pkl payload as `scripts/extract/extract_embeddings.py` — using the **full** test split (no subsample).

    uv run python scripts/extract/extract_cnnspot_parquet.py --extractor clip_large_patch14 \
        --out data/embeddings/cnnspot_clip_large_patch14.pkl
"""

from __future__ import annotations

import argparse
import glob
import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from clip_cues.feature_extractor import EXTRACTOR_CLASSES

SNAP = "data/hf_cache/hub/datasets--marco-willi--cnnspot-small/snapshots/*/data"
SPLIT_GLOB = {
    "train": "train-*.parquet",
    "validation": "validation-*.parquet",
    "test": "test-*.parquet",
}


class _ShardDataset(Dataset):
    """Decode images from one in-memory parquet shard (image stored as struct{bytes,path})."""

    def __init__(self, df: pd.DataFrame, split: str, transform):
        self.df = df.reset_index(drop=True)
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        img = Image.open(io.BytesIO(r["image"]["bytes"]))
        if img.mode != "RGB":
            img = img.convert("RGB")
        meta = {
            "image_id": r["image_id"],
            "label": int(r["label"]),
            "ds_name": r.get("ds_name", "cnnspot"),
            "split": self.split,
            "source": r.get("source", "unknown"),
        }
        return self.transform(img), meta


def _collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extractor", default="clip_large_patch14", choices=list(EXTRACTOR_CLASSES))
    p.add_argument("--out", required=True)
    p.add_argument("--layer", default="pooler_output")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--cache-dir", default="data/hf_cache")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    ex = EXTRACTOR_CLASSES[args.extractor](args.cache_dir, layer_id_to_extract=args.layer)
    ex.freeze()
    ex.model.to(device).eval()
    transform = ex.transforms

    snap_dirs = glob.glob(SNAP)
    if not snap_dirs:
        raise SystemExit(f"No cnnspot-small snapshot under {SNAP} — is it downloaded?")
    data_dir = snap_dirs[0]

    all_emb: list[np.ndarray] = []
    records: list[dict] = []
    for split, pat in SPLIT_GLOB.items():
        shards = sorted(glob.glob(f"{data_dir}/{pat}"))
        if not shards:
            print(f"  (no {split} shards, skipping)")
            continue
        n = sum(pd.read_parquet(s, columns=["label"]).shape[0] for s in shards)
        bar = tqdm(total=n, desc=f"{split}")
        for shard in shards:  # one shard in memory at a time -> bounded RAM
            df = pd.read_parquet(shard)
            loader = DataLoader(
                _ShardDataset(df, split, transform),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=_collate,
                pin_memory=(device == "cuda"),
            )
            with torch.no_grad():
                for tensors, metas in loader:
                    feats = ex.model(tensors.to(device))["extracted_features"].float().cpu().numpy()
                    all_emb.extend(feats)
                    records.extend(metas)
                    bar.update(len(metas))
            del df
        bar.close()

    embeddings = np.asarray(all_emb, dtype=np.float32)
    df = pd.DataFrame(records)
    print(f"Extracted {embeddings.shape}; splits: {df['split'].value_counts().to_dict()}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(
            {
                "embeddings": embeddings,
                "df": df,
                "identifier": out.stem,
                "model": args.extractor,
                "layer": args.layer,
            },
            f,
        )
    print(f"✓ Saved {out}")


if __name__ == "__main__":
    main()
