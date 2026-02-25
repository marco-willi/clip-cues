#!/usr/bin/env python3
"""
Validate a model against a dataset test split.

This script evaluates a trained model on a test dataset, calculating:
- Overall metrics (Accuracy, AUC, AP, mAP)
- Per-source metrics (each synthetic source vs real)
- Confusion matrices and ROC curves
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)
from tqdm import tqdm

from clip_cues import (
    CLIPLargePatch14,
    LinearHead,
    ActivationOrthogonalityHead,
    SyntheticImageClassifierInference,
)
from clip_cues.transforms import Transforms


def load_model(checkpoint_path: Path, device: torch.device):
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded model ready for inference
    """
    print(f"Loading model from {checkpoint_path}...")

    # Load CLIP feature extractor
    extractor = CLIPLargePatch14(cache_dir="hf_cache")
    extractor.freeze()

    # Load checkpoint to determine head type
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["state_dict"]

    # Determine head type from checkpoint structure
    if "model.classification_head.fc.weight" in state_dict:
        # LinearHead
        head = LinearHead(input_dim=extractor.output_dim, num_classes=1)
        print("  Detected LinearHead")
    elif "model.classification_head.layers.0.weight" in state_dict:
        # ActivationOrthogonalityHead or ClassificationHead
        hidden_dim = state_dict["model.classification_head.layers.0.weight"].shape[0]
        head = ActivationOrthogonalityHead(
            input_dim=extractor.output_dim,
            layer_dims=[hidden_dim],
        )
        print(f"  Detected ActivationOrthogonalityHead (hidden_dim={hidden_dim})")
    else:
        raise ValueError("Unknown head type in checkpoint")

    # Create inference model
    model = SyntheticImageClassifierInference(extractor.model, head)

    # Load weights
    weights = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(weights, strict=False)

    model.eval()
    model.to(device)

    print(f"  Model loaded successfully on {device}")
    return model, extractor.transforms


def run_inference(model, dataset_split, transforms, device: torch.device, batch_size: int = 32):
    """
    Run inference on a dataset split.

    Args:
        model: Model for inference
        dataset_split: Dataset split to evaluate
        transforms: Image transforms
        device: Device to run inference on
        batch_size: Batch size for inference

    Returns:
        predictions: Array of prediction probabilities
        ground_truth: Array of ground truth labels
        sources: List of source names
        image_ids: List of image IDs
    """
    inference_transforms = Transforms(transforms).get_inference_transforms()

    predictions = []
    ground_truth = []
    sources = []
    image_ids = []

    print(f"Running inference on {len(dataset_split)} samples...")

    # Process in batches
    for i in tqdm(range(0, len(dataset_split), batch_size)):
        batch_end = min(i + batch_size, len(dataset_split))
        batch = dataset_split[i:batch_end]

        # Transform images
        images = batch["image"] if isinstance(batch["image"], list) else [batch["image"]]
        batch_transformed = inference_transforms({"image": images})
        pixel_values = torch.stack(batch_transformed["pixel_values"]).to(device)

        # Run inference
        with torch.no_grad():
            probs = model(pixel_values)

        predictions.extend(probs.cpu().numpy().flatten())

        # Get labels
        labels = batch["label"] if isinstance(batch["label"], list) else [batch["label"]]
        ground_truth.extend(labels)

        # Get metadata
        batch_sources = batch["source"] if isinstance(batch["source"], list) else [batch["source"]]
        sources.extend(batch_sources)

        batch_ids = batch["image_id"] if isinstance(batch["image_id"], list) else [batch["image_id"]]
        image_ids.extend(batch_ids)

    return np.array(predictions), np.array(ground_truth), sources, image_ids


def calculate_metrics(predictions, ground_truth, threshold=0.5):
    """
    Calculate classification metrics.

    Args:
        predictions: Prediction probabilities
        ground_truth: Ground truth labels
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    # Binary predictions
    preds_binary = (predictions >= threshold).astype(int)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(ground_truth, preds_binary),
        "auc": roc_auc_score(ground_truth, predictions),
        "ap": average_precision_score(ground_truth, predictions),
        "confusion_matrix": confusion_matrix(ground_truth, preds_binary),
    }

    return metrics


def calculate_per_source_metrics(predictions, ground_truth, sources, dataset_name="unknown"):
    """
    Calculate metrics for each synthetic source vs real.

    This implementation follows the archive/src/detection_via_clip/analyse.py logic:
    - calculate_metrics(): For SynthCLIC/SynthBuster - each synthetic source vs ALL real
    - calculate_metrics_for_cnnspot(): For CNNSpot - synthetic only (no real comparison)

    Args:
        predictions: Prediction probabilities
        ground_truth: Ground truth labels
        sources: Source names
        dataset_name: Name of dataset (to determine which metric calculation to use)

    Returns:
        Dictionary of per-source metrics
    """
    # Group by source
    source_data = defaultdict(lambda: {"preds": [], "labels": []})

    for pred, label, source in zip(predictions, ground_truth, sources):
        source_data[source]["preds"].append(pred)
        source_data[source]["labels"].append(label)

    # Find real sources (label=0)
    real_sources = [src for src, data in source_data.items()
                    if all(label == 0 for label in data["labels"])]

    # Find synthetic sources (label=1)
    synthetic_sources = [src for src, data in source_data.items()
                         if all(label == 1 for label in data["labels"])]

    print(f"\nFound {len(real_sources)} real source(s): {real_sources}")
    print(f"Found {len(synthetic_sources)} synthetic source(s)")

    # Determine which metric calculation to use
    is_cnnspot = "cnnspot" in dataset_name.lower()

    # Calculate metrics for each synthetic source
    per_source_metrics = {}
    all_aps = []

    for synth_source in synthetic_sources:
        if is_cnnspot:
            # CNNSpot style: evaluate synthetic source only (already contains real)
            # From calculate_metrics_for_cnnspot in archive
            combined_preds = np.array(source_data[synth_source]["preds"])
            combined_labels = np.array(source_data[synth_source]["labels"])
        else:
            # SynthCLIC/SynthBuster style: each synthetic source vs ALL real
            # From calculate_metrics in archive
            combined_preds = []
            combined_labels = []

            # Add ALL real samples
            for real_source in real_sources:
                combined_preds.extend(source_data[real_source]["preds"])
                combined_labels.extend(source_data[real_source]["labels"])

            # Add synthetic samples from this source
            combined_preds.extend(source_data[synth_source]["preds"])
            combined_labels.extend(source_data[synth_source]["labels"])

            combined_preds = np.array(combined_preds)
            combined_labels = np.array(combined_labels)

        # Calculate metrics
        metrics = calculate_metrics(combined_preds, combined_labels)

        # Add F1 score (included in archive implementation)
        preds_binary = (combined_preds >= 0.5).astype(int)
        from sklearn.metrics import f1_score
        metrics["f1"] = f1_score(combined_labels, preds_binary)

        per_source_metrics[synth_source] = {
            **metrics,
            "n_real": sum(combined_labels == 0),
            "n_synthetic": sum(combined_labels == 1),
        }

        all_aps.append(metrics["ap"])

    # Calculate mAP (mean Average Precision across all synthetic sources)
    mAP = np.mean(all_aps) if all_aps else 0.0

    return per_source_metrics, mAP, real_sources, synthetic_sources


def print_results(overall_metrics, per_source_metrics, mAP, real_sources, synthetic_sources):
    """
    Print validation results in a formatted table.
    Matches the output format from archive implementation.
    """
    print("\n" + "="*90)
    print("VALIDATION RESULTS")
    print("="*90)

    # Overall metrics
    print("\nOverall Metrics:")
    print("-"*90)
    print(f"  Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"  AUC:      {overall_metrics['auc']:.4f}")
    print(f"  AP:       {overall_metrics['ap']:.4f}")
    print(f"  mAP:      {mAP:.4f}")

    # Confusion matrix
    cm = overall_metrics['confusion_matrix']
    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Real  Synthetic")
    print(f"    Actual Real     {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"    Actual Synth    {cm[1,0]:4d}    {cm[1,1]:4d}")

    # Per-source metrics
    print("\n" + "-"*90)
    print("Per-Source Metrics (Synthetic vs Real):")
    print("-"*90)
    print(f"Real source(s): {', '.join(real_sources)}")
    print()
    print(f"{'Source':<30} {'N_Real':>8} {'N_Synth':>8} {'Accuracy':>10} {'AUC':>10} {'AP':>10} {'F1':>10}")
    print("-"*90)

    # Sort by AP descending
    sorted_sources = sorted(per_source_metrics.items(), key=lambda x: x[1]['ap'], reverse=True)

    for source, metrics in sorted_sources:
        print(f"{source:<30} {metrics['n_real']:8d} {metrics['n_synthetic']:8d} "
              f"{metrics['accuracy']:10.4f} {metrics['auc']:10.4f} {metrics['ap']:10.4f} "
              f"{metrics['f1']:10.4f}")

    print("-"*90)
    print(f"{'mAP (mean across sources)':<59} {mAP:10.4f}")
    print("="*90)


def save_results(output_path: Path, overall_metrics, per_source_metrics, mAP,
                 predictions, ground_truth, sources, image_ids):
    """
    Save results to files (JSON and CSV formats).
    Matches archive implementation which saves both metrics and predictions.
    """
    import json
    import pandas as pd

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare results dictionary for JSON
    results = {
        "overall_metrics": {
            "accuracy": float(overall_metrics["accuracy"]),
            "auc": float(overall_metrics["auc"]),
            "ap": float(overall_metrics["ap"]),
            "mAP": float(mAP),
            "confusion_matrix": overall_metrics["confusion_matrix"].tolist(),
        },
        "per_source_metrics": {},
    }

    for source, metrics in per_source_metrics.items():
        results["per_source_metrics"][source] = {
            "accuracy": float(metrics["accuracy"]),
            "auc": float(metrics["auc"]),
            "ap": float(metrics["ap"]),
            "f1": float(metrics["f1"]),
            "n_real": int(metrics["n_real"]),
            "n_synthetic": int(metrics["n_synthetic"]),
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
        }

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Save metrics in long-form CSV format (matching archive style)
    metrics_csv_path = output_path.parent / f"{output_path.stem}_metrics.csv"
    metrics_records = []
    for source, metrics in per_source_metrics.items():
        for metric_name in ["accuracy", "auc", "ap", "f1"]:
            metrics_records.append({
                "source": source,
                "metric": metric_name,
                "value": float(metrics[metric_name]),
            })

    df_metrics = pd.DataFrame(metrics_records)
    df_metrics.to_csv(metrics_csv_path, index=False)
    print(f"Metrics CSV saved to {metrics_csv_path}")

    # Save detailed predictions in CSV format (matching archive test_results.csv)
    predictions_csv_path = output_path.parent / f"{output_path.stem}_predictions.csv"
    df_predictions = pd.DataFrame({
        "image_id": image_ids,
        "source": sources,
        "label": ground_truth,
        "label_prob": predictions,
        "label_pred": (predictions >= 0.5).astype(int),
    })
    df_predictions.to_csv(predictions_csv_path, index=False)
    print(f"Predictions CSV saved to {predictions_csv_path}")

    # Also save as npz for convenience
    predictions_npz_path = output_path.parent / f"{output_path.stem}_predictions.npz"
    np.savez(
        predictions_npz_path,
        predictions=predictions,
        ground_truth=ground_truth,
        sources=sources,
        image_ids=image_ids,
    )
    print(f"Predictions NPZ saved to {predictions_npz_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate a model on a dataset test split"
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to dataset directory (e.g., data/datasets/synthclic)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (default: test)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for results JSON (default: results/<dataset>_<checkpoint>.json)",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load dataset
    print(f"\nLoading dataset from {args.dataset}...")
    dataset = load_from_disk(str(args.dataset))

    if args.split not in dataset:
        raise ValueError(f"Split '{args.split}' not found in dataset. Available: {list(dataset.keys())}")

    test_split = dataset[args.split]
    print(f"  Loaded {args.split} split: {len(test_split)} samples")

    # Load model
    model, transforms = load_model(args.checkpoint, device)

    # Run inference
    predictions, ground_truth, sources, image_ids = run_inference(
        model, test_split, transforms, device, args.batch_size
    )

    # Calculate overall metrics
    print("\nCalculating metrics...")
    overall_metrics = calculate_metrics(predictions, ground_truth)

    # Calculate per-source metrics
    dataset_name = args.dataset.name
    per_source_metrics, mAP, real_sources, synthetic_sources = calculate_per_source_metrics(
        predictions, ground_truth, sources, dataset_name
    )

    # Print results
    print_results(overall_metrics, per_source_metrics, mAP, real_sources, synthetic_sources)

    # Save results
    if args.output is None:
        dataset_name = args.dataset.name
        checkpoint_name = args.checkpoint.stem
        args.output = Path("results") / f"{dataset_name}_{checkpoint_name}.json"

    save_results(
        args.output, overall_metrics, per_source_metrics, mAP,
        predictions, ground_truth, sources, image_ids
    )

    print("\n✓ Validation complete!")


if __name__ == "__main__":
    main()
