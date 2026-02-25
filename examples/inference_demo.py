#!/usr/bin/env python3
"""
Inference Demo for CLIP-Cues Synthetic Image Detection.

This script demonstrates how to load a pre-trained model and run inference
on images to detect if they are real or AI-generated.

Usage:
    # Single image
    python examples/inference_demo.py --image path/to/image.jpg

    # Multiple images (batch)
    python examples/inference_demo.py --images img1.jpg img2.jpg img3.jpg

    # Directory of images
    python examples/inference_demo.py --dir path/to/images/

    # With specific model
    python examples/inference_demo.py --image photo.jpg --model clip_orthogonal_combined
"""

import argparse
from pathlib import Path

import torch

from clip_cues import list_available_models, load_clip_classifier


def get_checkpoint_path(model_name: str) -> Path:
    """Get checkpoint path from model name or path.

    Args:
        model_name: Either a model name (e.g., 'clip_orthogonal_synthclic')
                    or a direct path to a checkpoint file.

    Returns:
        Path to the checkpoint file.
    """
    # If it's a direct path
    if Path(model_name).exists():
        return Path(model_name)

    # If it's a model name
    available = list_available_models()
    if model_name in available:
        return Path(available[model_name]["path"])

    # Try with .ckpt extension
    if model_name + ".ckpt" in [Path(m["path"]).name for m in available.values()]:
        for info in available.values():
            if Path(info["path"]).name == model_name + ".ckpt":
                return Path(info["path"])

    raise ValueError(
        f"Model '{model_name}' not found. Available models:\n"
        + "\n".join(f"  - {name}" for name in available.keys())
    )


def main():
    parser = argparse.ArgumentParser(
        description="Detect synthetic images using CLIP-Cues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Selection Guide:
  clip_orthogonal_synthclic    Best for modern AI (SD3, FLUX, Midjourney)
  clip_orthogonal_combined     Best for general/unknown sources
  cm_antonyms_combined         Interpretable (shows which concepts triggered)
  linear_probe_combined        Simple baseline

Examples:
  python examples/inference_demo.py --image photo.jpg
  python examples/inference_demo.py --images *.jpg --model clip_orthogonal_combined
  python examples/inference_demo.py --dir ./test_images/ --device cuda
        """,
    )

    # List models flag (before input group to allow standalone use)
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )

    # Input options (mutually exclusive, required unless --list-models)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--image",
        type=str,
        help="Path to a single image file",
    )
    input_group.add_argument(
        "--images",
        type=str,
        nargs="+",
        help="Paths to multiple image files",
    )
    input_group.add_argument(
        "--dir",
        type=str,
        help="Directory containing images (jpg, png, webp)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="clip_orthogonal_synthclic",
        help="Model name or checkpoint path (default: clip_orthogonal_synthclic)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (cuda/cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing multiple images",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)",
    )

    args = parser.parse_args()

    # List models if requested
    if args.list_models:
        print("\nAvailable Models:")
        print("=" * 60)
        for name, info in list_available_models().items():
            print(f"\n{name}")
            print(f"  Type: {info['type']}")
            print(f"  Training: {info['training_data']}")
            print(f"  Description: {info['description']}")
        return 0

    # Require at least one input option
    if not args.image and not args.images and not args.dir:
        parser.error("one of --image, --images, or --dir is required")

    # Collect image paths
    image_paths = []
    if args.image:
        image_paths = [Path(args.image)]
    elif args.images:
        image_paths = [Path(p) for p in args.images]
    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            print(f"Error: Directory not found: {args.dir}")
            return 1
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_paths = [p for p in dir_path.iterdir() if p.suffix.lower() in extensions]
        image_paths.sort()

    # Validate paths
    missing = [p for p in image_paths if not p.exists()]
    if missing:
        print("Error: Images not found:")
        for p in missing:
            print(f"  - {p}")
        return 1

    if not image_paths:
        print("Error: No images to process")
        return 1

    # Load model
    try:
        checkpoint_path = get_checkpoint_path(args.model)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    print(f"Loading model: {checkpoint_path}")
    print(f"Device: {args.device}")
    model = load_clip_classifier(checkpoint_path, device=args.device)
    model.eval()

    # Run inference
    print(f"\nProcessing {len(image_paths)} image(s)...")
    print("=" * 60)

    if len(image_paths) == 1:
        # Single image - use predict()
        prob = model.predict(image_paths[0])
        prediction = "SYNTHETIC" if prob > args.threshold else "REAL"

        print(f"\nImage: {image_paths[0]}")
        print(f"Probability: {prob:.1%}")
        print(f"Prediction:  {prediction}")

    else:
        # Multiple images - use predict_batch()
        probs = model.predict_batch([str(p) for p in image_paths], batch_size=args.batch_size)

        # Print results
        print(f"\n{'Image':<50} {'Prob':>8} {'Prediction':>12}")
        print("-" * 72)

        n_synthetic = 0
        for path, prob in zip(image_paths, probs):
            prediction = "SYNTHETIC" if prob > args.threshold else "REAL"
            if prob > args.threshold:
                n_synthetic += 1
            print(f"{str(path):<50} {prob:>7.1%} {prediction:>12}")

        # Summary
        print("-" * 72)
        print(f"Total: {len(image_paths)} images")
        print(f"Synthetic: {n_synthetic} ({n_synthetic / len(image_paths):.1%})")
        print(
            f"Real: {len(image_paths) - n_synthetic} ({(len(image_paths) - n_synthetic) / len(image_paths):.1%})"
        )

    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
