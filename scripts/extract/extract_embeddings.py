#!/usr/bin/env python
"""Extract CLIP (variant) image embeddings and cache them as a pickle (+ optional W&B artifact).

This is the one-time GPU step that makes the rest of the revision portable: extract once on a
GPU box (Lambda), cache the embeddings, then every downstream run (E2 sweep, E3 heads, E5
ablation) reuses the cache and needs neither images nor a big GPU.

Backbones come straight from the (backbone-parameterized) clip_cues feature extractor, so the
embeddings stay drop-in compatible with the published heads / concept-modeling code. The output
format matches scripts/extract/extract_clip_embeddings.py: a dict with ``embeddings`` (N, D) and a ``df``
metadata frame (image_id/label/ds_name/split/source).

Usage:
    python scripts/extract/extract_embeddings.py \
        --dataset marco-willi/synthclic \
        --extractor clip_base_patch16 \
        --out data/embeddings/synthclic_clip_base_patch16.pkl \
        --device cuda --wandb
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from clip_cues.feature_extractor import EXTRACTOR_CLASSES

# Short dataset name -> HuggingFace id (used when no local copy exists, e.g. on Lambda).
HF_DATASET_IDS = {
    "synthclic": "marco-willi/synthclic",
    "synthbuster-plus": "marco-willi/synthbuster-plus",
    "cnnspot": "marco-willi/cnnspot-small",
}


def load_split_dataset(dataset: str, local_dir: Path):
    """Load a dataset from local disk if present, else from HuggingFace.

    ``dataset`` may be a short name (``synthclic``) or a full HF id (``marco-willi/synthclic``);
    the local copy is looked up at ``local_dir/<short-name>``.
    """
    short = dataset.split("/")[-1]
    local_path = local_dir / short
    if local_path.exists():
        print(f"Loading dataset from local disk: {local_path}")
        return load_from_disk(str(local_path))
    hf_id = dataset if "/" in dataset else HF_DATASET_IDS.get(dataset)
    if hf_id is None:
        raise FileNotFoundError(
            f"Dataset not found locally at {local_path} and no HF id mapped for '{dataset}'."
        )
    print(f"Local dataset not found; loading from HuggingFace: {hf_id}")
    return load_dataset(hf_id)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dataset", required=True, help="HuggingFace dataset id or short name (synthclic)"
    )
    p.add_argument("--extractor", default="clip_large_patch14", choices=list(EXTRACTOR_CLASSES))
    p.add_argument("--out", required=True, help="Output .pkl path")
    p.add_argument("--layer", default="pooler_output", help="Layer id or int index")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--num-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel DataLoader workers for image decode (GPU-saturating).",
    )
    p.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Cap the 'test' split to N (shuffle seed=42), e.g. for cnnspot's 108k test. "
        "Matches the E1 cross-dataset eval subsample.",
    )
    p.add_argument("--cache-dir", default="data/hf_cache")
    p.add_argument("--dataset-dir", default="data/datasets", help="Where local datasets live")
    p.add_argument("--wandb", action="store_true", help="Log embeddings as a W&B artifact")
    return p.parse_args()


class _HFImageDataset(Dataset):
    """Wrap one HF split so image decode + transform run in parallel DataLoader workers.

    Single-threaded decode in the extraction loop leaves the GPU idle between batches (CPU-bound);
    moving decode into worker processes saturates the GPU. Returns (tensor, metadata) per item.
    """

    def __init__(self, hf_split, split_name: str, transform):
        self.split = hf_split
        self.split_name = split_name
        self.transform = transform

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int):
        ex = self.split[idx]
        image = ex["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        meta = {
            "image_id": ex["image_id"],
            "label": ex["label"],
            "ds_name": ex.get("ds_name", "unknown"),
            "split": self.split_name,
            "source": ex.get("source", "unknown"),
        }
        return self.transform(image), meta


def _collate(batch):
    """Stack image tensors; keep metadata dicts as a list (order preserved)."""
    tensors = torch.stack([b[0] for b in batch])
    metas = [b[1] for b in batch]
    return tensors, metas


@torch.no_grad()
def extract(
    extractor, dataset, device: str, batch_size: int = 64, num_workers: int = 8
) -> tuple[np.ndarray, pd.DataFrame]:
    """Run the frozen encoder over every split and collect embeddings + metadata.

    Decode/transform is parallelized across ``num_workers`` DataLoader processes so the GPU stays
    busy. Order is preserved (shuffle=False), so ``embeddings[i]`` aligns with ``df.iloc[i]``.
    """
    transform = extractor.transforms  # torchvision Compose: PIL -> normalized tensor
    all_embeddings: list[np.ndarray] = []
    records: list[dict] = []

    total = sum(len(dataset[s]) for s in dataset.keys())
    bar = tqdm(total=total, desc="Extracting embeddings")
    for split in dataset.keys():
        loader = DataLoader(
            _HFImageDataset(dataset[split], split, transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate,
            pin_memory=(device == "cuda"),
            persistent_workers=False,
        )
        for tensors, metas in loader:
            feats = extractor.model(tensors.to(device))["extracted_features"].float().cpu().numpy()
            all_embeddings.extend(feats)
            records.extend(metas)
            bar.update(len(metas))
    bar.close()
    return np.asarray(all_embeddings, dtype=np.float32), pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    layer: str | int = int(args.layer) if args.layer.lstrip("-").isdigit() else args.layer

    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        print("⚠ CUDA not available, falling back to CPU")

    extractor = EXTRACTOR_CLASSES[args.extractor](args.cache_dir, layer_id_to_extract=layer)
    extractor.freeze()
    extractor.model.to(device).eval()

    dataset = load_split_dataset(args.dataset, Path(args.dataset_dir))
    if args.max_test_samples is not None and "test" in dataset:
        n = min(args.max_test_samples, len(dataset["test"]))
        print(f"Capping test split: {len(dataset['test'])} -> {n} (shuffle seed=42)")
        dataset["test"] = dataset["test"].shuffle(seed=42).select(range(n))
    embeddings, df = extract(
        extractor, dataset, device, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"Extracted embeddings shape: {embeddings.shape}")
    print(f"  Splits: {df['split'].value_counts().to_dict()}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embeddings": embeddings,
        "df": df,
        "identifier": out_path.stem,
        "model": args.extractor,
        "layer": args.layer,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"✓ Saved {out_path}")

    if args.wandb:
        import wandb

        run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "clip-cues"),
            job_type="extract-embeddings",
            config=vars(args),
        )
        art = wandb.Artifact(
            f"embeddings-{args.dataset.split('/')[-1]}-{args.extractor}", type="embeddings"
        )
        art.add_file(str(out_path))
        run.log_artifact(art)
        run.finish()
        print("✓ Logged W&B artifact")


if __name__ == "__main__":
    main()
