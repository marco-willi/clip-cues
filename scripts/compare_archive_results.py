#!/usr/bin/env python3
"""
Compare archive test results with new validation results.

This script:
1. Reads archive results from data/verify/test_results.csv
2. Calculates metrics using the same method as archive
3. Compares with new validation results
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def calculate_archive_metrics(df):
    """Calculate metrics using archive method (calculate_metrics)."""
    results = []
    all_aps = []

    # Filter for SynthCLIC
    df_synthclic = df[df["ds_name"] == "synthclic"].copy()

    # Identify sources
    real_sources = df_synthclic[df_synthclic["label"] == 0]["source"].unique()
    synthetic_sources = df_synthclic[df_synthclic["label"] == 1]["source"].unique()

    # Per-source metrics (SynthCLIC style)
    for synth_source in sorted(synthetic_sources):
        # Get synthetic samples for this source
        df_synth = df_synthclic[df_synthclic["source"] == synth_source]

        # Get ALL real samples
        df_real = df_synthclic[df_synthclic["label"] == 0]

        # Combine
        df_eval = pd.concat([df_real, df_synth])

        y_true = df_eval["label"].values
        y_prob = df_eval["prediction"].values
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "source": synth_source,
            "n_real": len(df_real),
            "n_synthetic": len(df_synth),
            "accuracy": accuracy_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_prob),
            "ap": average_precision_score(y_true, y_prob),
            "f1": f1_score(y_true, y_pred),
        }

        results.append(metrics)
        all_aps.append(metrics["ap"])

    # Overall metrics
    y_true_all = df_synthclic["label"].values
    y_prob_all = df_synthclic["prediction"].values
    y_pred_all = (y_prob_all >= 0.5).astype(int)

    overall = {
        "accuracy": accuracy_score(y_true_all, y_pred_all),
        "auc": roc_auc_score(y_true_all, y_prob_all),
        "ap": average_precision_score(y_true_all, y_prob_all),
        "mAP": np.mean(all_aps),
    }

    return overall, results, real_sources, synthetic_sources, len(df_synthclic)


def main():
    # Read archive results
    print("Reading archive results...")
    df_archive = pd.read_csv("data/verify/test_results.csv")

    # Calculate archive metrics
    archive_overall, archive_per_source, real_sources, synth_sources, n_samples = (
        calculate_archive_metrics(df_archive)
    )

    print("\n" + "=" * 90)
    print("ARCHIVE RESULTS (from data/verify/test_results.csv)")
    print("=" * 90)
    print("\nDataset: SynthCLIC")
    print(f"Samples: {n_samples}")
    print(f"Real sources: {list(real_sources)}")
    print(f"Synthetic sources: {len(synth_sources)}")

    print("\nOverall Metrics:")
    print("-" * 90)
    for metric, value in archive_overall.items():
        print(f"  {metric:20s}: {value:.4f}")

    print("\nPer-Source Metrics:")
    print("-" * 90)
    print(
        f"{'Source':<30} {'N_Real':>8} {'N_Synth':>8} {'Accuracy':>10} {'AUC':>10} {'AP':>10} {'F1':>10}"
    )
    print("-" * 90)

    for r in sorted(archive_per_source, key=lambda x: x["ap"], reverse=True):
        print(
            f"{r['source']:<30} {r['n_real']:8d} {r['n_synthetic']:8d} "
            f"{r['accuracy']:10.4f} {r['auc']:10.4f} {r['ap']:10.4f} {r['f1']:10.4f}"
        )

    # Check if new results exist
    new_results_path = Path("results/synthclic_linear_probe_combined.json")
    if new_results_path.exists():
        print("\n" + "=" * 90)
        print("COMPARISON WITH NEW RESULTS")
        print("=" * 90)

        with open(new_results_path, "r") as f:
            new_results = json.load(f)

        # Compare
        print("\nOVERALL METRICS:")
        print("-" * 90)
        print(f"{'Metric':<20} {'Archive':>15} {'New':>15} {'Difference':>15}")
        print("-" * 90)

        for metric in ["accuracy", "auc", "ap", "mAP"]:
            archive_val = archive_overall[metric]
            new_val = new_results["overall_metrics"][metric]
            diff = new_val - archive_val
            print(f"{metric:<20} {archive_val:>15.4f} {new_val:>15.4f} {diff:>+15.4f}")

        print("\nPER-SOURCE AP COMPARISON:")
        print("-" * 90)
        print(f"{'Source':<30} {'Archive AP':>15} {'New AP':>15} {'Difference':>15}")
        print("-" * 90)

        for r in archive_per_source:
            source = r["source"]
            archive_ap = r["ap"]
            new_ap = new_results["per_source_metrics"][source]["ap"]
            diff = new_ap - archive_ap
            print(f"{source:<30} {archive_ap:>15.4f} {new_ap:>15.4f} {diff:>+15.4f}")

        print("\n" + "=" * 90)
        print("NOTE: Differences are expected if using different dataset splits.")
        print("Archive used: Combined test results from main code")
        print("New used: Test split from data/datasets/synthclic")
        print("=" * 90)
    else:
        print(f"\nNew results not found at {new_results_path}")
        print(
            "Run: python scripts/validate_model.py data/checkpoints/linear_probe_combined.ckpt data/datasets/synthclic"
        )


if __name__ == "__main__":
    main()
