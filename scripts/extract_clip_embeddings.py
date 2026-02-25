"""Extract CLIP embeddings for a dataset and save to disk.

This script loads a dataset, extracts CLIP image embeddings, and saves them
along with metadata to a pickle file for use in training.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm

from clip_cues import CLIPLargePatch14
from clip_cues.transforms import Transforms


def extract_embeddings(
    dataset_path: Path, output_path: Path, identifier: str, device: str = "cuda"
):
    """Extract CLIP embeddings for a dataset.

    Args:
        dataset_path: Path to the dataset directory
        output_path: Path to save embeddings pickle file
        identifier: Identifier for this embedding extraction (e.g., dataset name)
        device: Device to use for extraction ("cuda" or "cpu")
    """
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(str(dataset_path))

    # Initialize CLIP model
    print("Loading CLIP model...")
    extractor = CLIPLargePatch14(cache_dir="./hf_cache")
    extractor.freeze()
    extractor.model = extractor.model.to(device)
    extractor.model.eval()

    # Get transforms
    transforms = Transforms(extractor.transforms)
    inference_transforms = transforms.get_inference_transforms()

    # Collect embeddings and metadata
    all_embeddings = []
    metadata_records = []

    # Process all splits
    total_samples = sum(len(dataset[split]) for split in dataset.keys())
    progress_bar = tqdm(total=total_samples, desc="Extracting embeddings")

    for split_name in dataset.keys():
        split_data = dataset[split_name]

        for idx, example in enumerate(split_data):
            # Extract image
            image = example["image"]

            # Apply transforms
            batch = inference_transforms({"image": [image]})
            pixel_values = torch.stack(batch["pixel_values"]).to(device)

            # Get CLIP embedding
            with torch.no_grad():
                outputs = extractor.model(pixel_values)
                # Extract the feature from the modified forward pass
                embedding = outputs["extracted_features"]
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
        "model": "CLIP ViT-L/14",
    }

    with open(output_path, "wb") as f:
        pickle.dump(output_data, f)

    print(f"✓ Embeddings saved to: {output_path}")
    print(f"  Identifier: {identifier}")
    print(f"  Total samples: {len(df)}")
    print(f"  Splits: {df['split'].value_counts().to_dict()}")
    print(f"  Sources: {df['source'].value_counts().to_dict()}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract CLIP embeddings for a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract embeddings for SynthCLIC
  python extract_clip_embeddings.py synthclic

  # Extract embeddings for SynthBuster+ with custom output
  python extract_clip_embeddings.py synthbuster-plus --output-dir data/embeddings

  # Use CPU instead of GPU
  python extract_clip_embeddings.py synthclic --device cpu
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
    identifier = f"{args.dataset}_embeddings"
    output_path = args.output_dir / f"{identifier}.pkl"

    # Check dataset exists
    if not dataset_path.exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please download the dataset first using:")
        print(f"  python scripts/download_dataset.py {args.dataset}")
        return 1

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        args.device = "cpu"

    print(f"Device: {args.device}")

    # Extract embeddings
    extract_embeddings(dataset_path, output_path, identifier, device=args.device)

    return 0


if __name__ == "__main__":
    exit(main())
