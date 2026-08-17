"""Extract projected CLIP embeddings for a dataset and save to disk.

This script extracts CLIP image embeddings AFTER projection (in the shared embedding space).
These projected embeddings can be directly compared with CLIP text embeddings using cosine similarity.

The key difference from extract_clip_embeddings.py:
- Uses CLIPVisionModelWithProjection instead of CLIPVisionModel
- Extracts image_embeds (projected) instead of pooler_output (pre-projection)
- Embeddings are in the shared CLIP embedding space (normalized)
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoProcessor, CLIPVisionModelWithProjection

# HuggingFace dataset ids — used when no local copy exists (e.g. on a Lambda box where
# data/ is not rsynced); mirrors extract_clip_embeddings.py.
HF_DATASET_IDS = {
    "synthclic": "marco-willi/synthclic",
    "synthbuster-plus": "marco-willi/synthbuster-plus",
    "cnnspot": "marco-willi/cnnspot-small",
}


def load_split_dataset(dataset_name: str, dataset_path: Path):
    """Load from local disk if present, else from HuggingFace."""
    if dataset_path.exists():
        print(f"Loading dataset from local disk: {dataset_path}")
        return load_from_disk(str(dataset_path))
    hf_id = HF_DATASET_IDS.get(dataset_name)
    if hf_id is None:
        raise FileNotFoundError(
            f"Dataset not found locally at {dataset_path} and no HF id mapped for '{dataset_name}'."
        )
    print(f"Local dataset not found; loading from HuggingFace: {hf_id}")
    return load_dataset(hf_id)


def extract_projected_embeddings(dataset, output_path: Path, identifier: str, device: str = "cuda"):
    """Extract projected CLIP embeddings for an (already-loaded) dataset.

    Args:
        dataset: A DatasetDict with image/label/image_id/ds_name/source columns.
        output_path: Path to save embeddings pickle file
        identifier: Identifier for this embedding extraction (e.g., dataset name)
        device: Device to use for extraction ("cuda" or "cpu")
    """
    # Initialize CLIP model with projection
    print("Loading CLIP model with projection...")
    model = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-large-patch14-336", cache_dir="data/hf_cache"
    )
    processor = AutoProcessor.from_pretrained(
        "openai/clip-vit-large-patch14-336", cache_dir="data/hf_cache"
    )

    model = model.to(device)
    model.eval()

    # Collect embeddings and metadata
    all_embeddings = []
    metadata_records = []

    # Process all splits
    total_samples = sum(len(dataset[split]) for split in dataset.keys())
    progress_bar = tqdm(total=total_samples, desc="Extracting projected embeddings")

    for split_name in dataset.keys():
        split_data = dataset[split_name]

        for idx, example in enumerate(split_data):
            # Extract image
            image = example["image"]

            # Process image using AutoProcessor
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get projected CLIP embedding
            with torch.inference_mode():
                outputs = model(**inputs)
                # image_embeds are the projected embeddings in the shared space
                embedding = outputs.image_embeds
                embedding = embedding.cpu().numpy()[0]

            all_embeddings.append(embedding)

            # Store metadata
            metadata_records.append(
                {
                    "image_id": example["image_id"],
                    "label": example["label"],
                    "ds_name": example.get("ds_name", "unknown"),
                    "split": split_name,
                    "source": example.get("source", "unknown"),
                }
            )

            progress_bar.update(1)

    progress_bar.close()

    # Convert to numpy array
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    print(f"Extracted embeddings shape: {embeddings_array.shape}")

    # Create metadata dataframe
    import pandas as pd

    df = pd.DataFrame(metadata_records)
    print(f"Metadata shape: {df.shape}")

    # Save to pickle
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "embeddings": embeddings_array,
        "df": df,
        "identifier": identifier,
        "model": "CLIP ViT-L/14@336 (projected)",
        "embedding_type": "projected",  # Mark as projected embeddings
    }

    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)

    print(f"✓ Projected embeddings saved to: {output_path}")
    print(f"  Identifier: {identifier}")
    print("  Embedding type: Projected (shared CLIP space)")
    print(f"  Total samples: {len(df)}")
    print(f"  Splits: {df['split'].value_counts().to_dict()}")
    print(f"  Sources: {df['source'].value_counts().to_dict()}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract projected CLIP embeddings for a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract projected embeddings for SynthCLIC
  python extract_clip_projected_embeddings.py synthclic

  # Extract projected embeddings for SynthBuster+ with custom output
  python extract_clip_projected_embeddings.py synthbuster-plus --output-dir data/embeddings

  # Use CPU instead of GPU
  python extract_clip_projected_embeddings.py synthclic --device cpu

Note:
  These embeddings are in the shared CLIP embedding space and can be directly
  compared with CLIP text embeddings using cosine similarity.
        """,
    )

    parser.add_argument(
        "dataset",
        type=str,
        choices=["synthclic", "synthbuster-plus", "cnnspot"],
        help="Dataset to extract embeddings for",
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/datasets"),
        help="Directory containing the dataset (default: data/datasets)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/embeddings"),
        help="Directory to save embeddings (default: data/embeddings)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for extraction (default: cuda)",
    )

    args = parser.parse_args()

    # Map dataset name to directory
    dataset_dir_map = {
        "synthclic": "synthclic",
        "synthbuster-plus": "synthbuster-plus",
        "cnnspot": "cnnspot",
    }

    # Construct paths
    dataset_path = args.dataset_dir / dataset_dir_map[args.dataset]
    identifier = f"{args.dataset}_projected_embeddings"
    output_path = args.output_dir / f"{identifier}.pkl"

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        args.device = "cpu"

    print(f"Device: {args.device}")

    # Load dataset (local copy if present, else from HuggingFace — e.g. on Lambda)
    dataset = load_split_dataset(args.dataset, dataset_path)

    # Extract embeddings
    extract_projected_embeddings(dataset, output_path, identifier, device=args.device)

    return 0


if __name__ == "__main__":
    exit(main())
