#!/usr/bin/env python3
"""
Download datasets from Hugging Face for CLIP-Cues.

This script downloads datasets used for training and evaluating synthetic image detection models.
Supported datasets:
- marco-willi/synthclic (SynthCLIC)
- marco-willi/synthbuster-plus (synthbuster-plus+)
- marco-willi/cnnspot (CNNSpot)
"""

import argparse
from pathlib import Path

from datasets import load_dataset

AVAILABLE_DATASETS = {
    "synthclic": "marco-willi/synthclic",
    "synthbuster-plus": "marco-willi/synthbuster-plus",
    "cnnspot": "marco-willi/cnnspot",
}


def download_dataset(
    dataset_name: str,
    output_dir: Path,
    cache_dir: Path | None = None,
    streaming: bool = False,
):
    """
    Download a dataset from Hugging Face.

    Args:
        dataset_name: Name of the dataset (e.g., 'synthclic')
        output_dir: Directory to save the dataset
        cache_dir: Optional cache directory for Hugging Face datasets
        streaming: Whether to use streaming mode (doesn't download full dataset)
    """
    if dataset_name not in AVAILABLE_DATASETS:
        available = ", ".join(AVAILABLE_DATASETS.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available datasets: {available}")

    hf_dataset_name = AVAILABLE_DATASETS[dataset_name]

    print(f"Downloading {dataset_name} dataset from Hugging Face...")
    print(f"Dataset: {hf_dataset_name}")
    print(f"Output directory: {output_dir}")
    print(f"Streaming mode: {streaming}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download dataset
    dataset = load_dataset(
        hf_dataset_name,
        cache_dir=cache_dir,
        streaming=streaming,
    )

    print("\nDataset loaded successfully!")

    if not streaming:
        print(f"Splits: {list(dataset.keys())}")

        for split_name, split_data in dataset.items():
            print(f"\n{split_name}:")
            print(f"  Number of examples: {len(split_data)}")
            print(f"  Features: {split_data.features}")

        # Save dataset to disk
        dataset_path = output_dir / dataset_name
        print(f"\nSaving dataset to {dataset_path}...")
        dataset.save_to_disk(str(dataset_path))

        print("\n✓ Dataset downloaded and saved successfully!")
        print(f"  Location: {dataset_path}")

        # Print example
        if "train" in dataset:
            print("\nExample from training set:")
            example = dataset["train"][0]
            for key, value in example.items():
                if key != "image":  # Don't print the full image
                    print(f"  {key}: {value}")
    else:
        print("\nStreaming dataset loaded successfully!")
        print("Note: Dataset is not saved to disk in streaming mode.")

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets from Hugging Face for CLIP-Cues"
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=list(AVAILABLE_DATASETS.keys()),
        help="Name of the dataset to download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/datasets"),
        help="Output directory for the dataset (default: data/datasets)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory for Hugging Face datasets (default: None)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode (doesn't download full dataset)",
    )

    args = parser.parse_args()

    download_dataset(
        args.dataset,
        args.output_dir,
        args.cache_dir,
        args.streaming,
    )


if __name__ == "__main__":
    main()
