#!/usr/bin/env python3
"""
Verify that the validation implementation matches the archive calculation method.

This script loads the archive predictions and recalculates metrics using both:
1. Our new implementation's method
2. Direct copy of archive's calculate_metrics function

They should produce identical results.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def archive_calculate_metrics(df):
    """
    Direct copy of calculate_metrics from archive/src/detection_via_clip/analyse.py

    Calculate metrics for the SynthCLIC / SynthBuster datasets
    Args:
        df (pd.DataFrame): DataFrame containing the evaluation results.
         - must contain columns: 'ds_name', 'source', 'label_pred', 'label_prob', 'label'.
    """
    from sklearn import metrics

    results = list()

    snythetic_sources = df.query("label == 1").source.unique()
    ds_name = df.ds_name.unique()[0]

    for synthetic_source in snythetic_sources:
        df_synthetic = df.query(f"source == '{synthetic_source}'")
        df_real = df.query("label == 0")

        df_eval = pd.concat([df_real, df_synthetic])

        collected_metrics = {
            "average_precision": metrics.average_precision_score(
                y_score=df_eval.label_prob, y_true=df_eval.label
            ),
            "roc_auc": metrics.roc_auc_score(y_score=df_eval.label_prob, y_true=df_eval.label),
            "accuracy": metrics.accuracy_score(y_pred=df_eval.label_pred, y_true=df_eval.label),
            "F1": metrics.f1_score(y_pred=df_eval.label_pred, y_true=df_eval.label),
        }

        for metric_name, metric_value in collected_metrics.items():
            results.append(
                {
                    "ds_name": ds_name,
                    "source": synthetic_source,
                    "metric": metric_name,
                    "value": metric_value,
                }
            )
    return results


def new_calculate_metrics(df):
    """
    Calculate metrics using our new implementation's method.
    """
    results = []

    snythetic_sources = df.query("label == 1").source.unique()
    ds_name = df.ds_name.unique()[0]

    for synthetic_source in snythetic_sources:
        # Get synthetic samples for this source
        df_synthetic = df.query(f"source == '{synthetic_source}'")

        # Get ALL real samples
        df_real = df.query("label == 0")

        # Combine
        df_eval = pd.concat([df_real, df_synthetic])

        y_true = df_eval["label"].values
        y_prob = df_eval["prediction"].values  # Note: column name difference
        y_pred = (y_prob >= 0.5).astype(int)

        collected_metrics = {
            "average_precision": average_precision_score(y_true, y_prob),
            "roc_auc": roc_auc_score(y_true, y_prob),
            "accuracy": accuracy_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
        }

        for metric_name, metric_value in collected_metrics.items():
            results.append(
                {
                    "ds_name": ds_name,
                    "source": synthetic_source,
                    "metric": metric_name,
                    "value": metric_value,
                }
            )

    return results


def compare_methods():
    """Compare the two calculation methods."""

    # Read archive results
    print("Loading data/verify/test_results.csv...")
    df = pd.read_csv("data/verify/test_results.csv")

    # Filter for SynthCLIC
    df_sc = df[df["ds_name"] == "synthclic"].copy()

    # Add label_pred and rename prediction to label_prob for archive method
    df_sc["label_pred"] = (df_sc["prediction"] >= 0.5).astype(int)
    df_sc["label_prob"] = df_sc["prediction"]

    print(f"SynthCLIC dataset: {len(df_sc)} samples\n")

    # Calculate using archive method
    print("Calculating metrics using ARCHIVE method...")
    archive_results = archive_calculate_metrics(df_sc)
    df_archive = pd.DataFrame(archive_results)

    # Calculate using new method
    print("Calculating metrics using NEW method...")
    new_results = new_calculate_metrics(df_sc)
    df_new = pd.DataFrame(new_results)

    # Compare
    print("\n" + "=" * 90)
    print("VERIFICATION: Archive Method vs New Method")
    print("=" * 90)

    # Merge on source and metric
    df_comparison = df_archive.merge(df_new, on=["source", "metric"], suffixes=("_archive", "_new"))

    # Calculate differences
    df_comparison["difference"] = df_comparison["value_new"] - df_comparison["value_archive"]
    df_comparison["match"] = df_comparison["difference"].abs() < 1e-10

    # Print results
    print("\nPer-Source Metric Comparison:")
    print("-" * 90)
    print(f"{'Source':<20} {'Metric':<20} {'Archive':>12} {'New':>12} {'Diff':>12} {'Match':>8}")
    print("-" * 90)

    for _, row in df_comparison.iterrows():
        match_symbol = "✓" if row["match"] else "✗"
        print(
            f"{row['source']:<20} {row['metric']:<20} {row['value_archive']:12.6f} "
            f"{row['value_new']:12.6f} {row['difference']:+12.9f} {match_symbol:>8}"
        )

    # Summary
    all_match = df_comparison["match"].all()

    print("\n" + "=" * 90)
    if all_match:
        print("✅ SUCCESS: All metrics match! Implementation is mathematically identical.")
    else:
        print("❌ FAILURE: Some metrics don't match. Check implementation.")
        non_matching = df_comparison[~df_comparison["match"]]
        print(f"\nNon-matching metrics ({len(non_matching)}):")
        print(non_matching[["source", "metric", "value_archive", "value_new", "difference"]])

    print("=" * 90)

    return all_match


if __name__ == "__main__":
    success = compare_methods()
    exit(0 if success else 1)
