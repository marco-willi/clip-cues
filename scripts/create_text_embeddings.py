"""Create CLIP text embeddings for concept vocabulary.

This script reads a vocabulary CSV file (e.g., antonyms.csv) and creates
CLIP text embeddings for each concept, saving them in a format suitable
for training concept bottleneck models.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer


def create_text_embeddings(
    vocab_csv_path: Path, output_path: Path, device: str = "cuda", cache_dir: str = "./hf_cache"
):
    """Create CLIP text embeddings for vocabulary.

    Args:
        vocab_csv_path: Path to vocabulary CSV file
        output_path: Path to save embeddings .pt file
        device: Device to use for extraction
        cache_dir: Directory to cache downloaded models
    """
    print(f"Loading vocabulary from: {vocab_csv_path}")
    df = pd.read_csv(vocab_csv_path)

    # Load CLIP model and tokenizer
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", cache_dir=cache_dir)
    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-large-patch14-336", cache_dir=cache_dir
    )
    model = model.to(device)
    model.eval()

    # Collect all prompts
    prompts = []
    concept_names = []

    for _, row in df.iterrows():
        # Add both positive and negative prompts
        prompts.append(row["positive_prompt"])
        prompts.append(row["negative_prompt"])

        concept_names.append(f"{row['attribute_name']}_positive")
        concept_names.append(f"{row['attribute_name']}_negative")

    print(f"Total concepts: {len(concept_names)}")
    print(f"Processing {len(prompts)} prompts...")

    # Extract embeddings
    all_embeddings = []

    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Extracting text embeddings"):
            inputs = tokenizer([prompt], padding=True, return_tensors="pt").to(device)
            text_features = model.get_text_features(**inputs)
            # Normalize embeddings
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            all_embeddings.append(text_features.cpu())

    # Stack embeddings
    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    print(f"Embeddings shape: {embeddings_tensor.shape}")

    # Save embeddings
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {"embeddings": embeddings_tensor, "vocabulary": concept_names}

    torch.save(output_data, output_path)
    print(f"✓ Text embeddings saved to: {output_path}")
    print(f"  Total concepts: {len(concept_names)}")
    print(f"  Embedding dimension: {embeddings_tensor.shape[1]}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Create CLIP text embeddings for concept vocabulary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create embeddings for antonyms vocabulary
  python create_text_embeddings.py

  # Use CPU instead of GPU
  python create_text_embeddings.py --device cpu

  # Custom paths
  python create_text_embeddings.py \\
      --vocab-csv data/vocabularies/antonyms.csv \\
      --output data/vocabulary/text_embeddings_antonyms.pt
        """,
    )

    parser.add_argument(
        "--vocab-csv",
        type=Path,
        default=Path("data/vocabularies/antonyms.csv"),
        help="Path to vocabulary CSV file (default: data/vocabularies/antonyms.csv)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/vocabulary/text_embeddings_antonym_pairs_v2.pt"),
        help="Output path for embeddings (default: data/vocabulary/text_embeddings_antonym_pairs_v2.pt)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for extraction (default: cuda)",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./hf_cache",
        help="Directory to cache downloaded models (default: ./hf_cache)",
    )

    args = parser.parse_args()

    # Check input exists
    if not args.vocab_csv.exists():
        print(f"❌ Vocabulary file not found: {args.vocab_csv}")
        return 1

    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        args.device = "cpu"

    print(f"Device: {args.device}")

    # Create embeddings
    create_text_embeddings(
        args.vocab_csv, args.output, device=args.device, cache_dir=args.cache_dir
    )

    return 0


if __name__ == "__main__":
    exit(main())
