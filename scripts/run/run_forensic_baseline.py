#!/usr/bin/env python
"""E1: CNNSpot forensic baseline on SynthCLIC.

Two modes:

  zero_shot  — evaluate the pre-trained ProGAN checkpoint on SynthCLIC without any fine-tuning.
               Shows cross-domain forensic generalisation (or failure thereof on diffusion images).

  retrain    — fine-tune a ResNet-50 (ImageNet init) on SynthCLIC train split using the CNNSpot
               augmentation protocol (JPEG + Gaussian blur, each p=0.5), then evaluate on test.
               This is the apples-to-apples comparison with the published CLIP models.

Usage:
    # Mode A — zero-shot (no GPU needed for inference; CPU feasible for small subsets)
    python scripts/run/run_forensic_baseline.py zero_shot \\
        --checkpoint data/checkpoints/cnnspot/blur_jpg_prob0.5.pth \\
        --wandb

    # Smoke test (CPU, 256 samples)
    python scripts/run/run_forensic_baseline.py zero_shot \\
        --checkpoint data/checkpoints/cnnspot/blur_jpg_prob0.5.pth \\
        --device cpu --max-samples 256

    # Mode B — retrain on SynthCLIC (run on Lambda GPU box)
    python scripts/run/run_forensic_baseline.py retrain --device cuda --wandb
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()  # loads WANDB_API_KEY, WANDB_ENTITY, LAMBDA_* etc. from .env

from clip_cues.dataset import get_dataset  # noqa: E402
from clip_cues_research.forensics.patch_cnn import (  # noqa: E402
    build_model,
    evaluate,
    print_eval_results,
    train,
)
from clip_cues_research.results import make_run_id, save_run_results  # noqa: E402

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "clip-cues")

# The three individual datasets that make up the "combined" training set (parity with the combined
# CLIP linear probe). Order is fixed for reproducibility.
COMBINED_PARTS = ["synthclic", "synthbuster-plus", "cnnspot"]


def build_combined_dataset(cache_dir: str, parts: list[str] = COMBINED_PARTS):
    """Build a combined DatasetDict (train + validation) by concatenating the individual datasets.

    Training only needs ``image`` + ``label``, so we drop the other columns and concatenate. We do
    **not** ``cast_column`` — that rewrites the whole Arrow table (including the large image bytes)
    and overflows the 2 GB offset limit; ``concatenate_datasets`` merely chains chunks. The three
    datasets share identical ``image``/``label`` features (``ClassLabel['real','fake']``), so the
    direct concatenation is a cheap metadata op. There is no combined *test* split — the
    combined-trained model is evaluated on each individual test set (the E1 cross-dataset matrix
    columns), matching how the combined CLIP probe fills its matrix row.
    """
    from datasets import DatasetDict, concatenate_datasets

    per_split: dict[str, list] = {"train": [], "validation": []}
    for name in parts:
        dd = get_dataset(name, cache_dir=cache_dir)
        for split in per_split:
            ds = dd[split]
            keep = [c for c in ds.column_names if c not in ("image", "label")]
            per_split[split].append(ds.remove_columns(keep))
    combined = {s: concatenate_datasets(parts_list) for s, parts_list in per_split.items()}
    for s, ds in combined.items():
        print(f"  combined {s}: {len(ds)} samples (from {', '.join(parts)})")
    return DatasetDict(combined)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("mode", choices=["zero_shot", "retrain"], help="Evaluation mode")

    # Data
    p.add_argument(
        "--dataset",
        default="marco-willi/synthclic",
        help="Train / zero-shot source dataset (HF id or short name)",
    )
    p.add_argument(
        "--eval-datasets",
        default=None,
        help="Comma-separated datasets to evaluate on (default: the --dataset). "
        "Each is a short name or HF id; enables cross-dataset generalization.",
    )
    p.add_argument(
        "--eval-max-samples",
        type=int,
        default=None,
        help="Subsample each eval test split (e.g. 4000) — needed for the huge CNNSpot test split.",
    )
    p.add_argument("--cache-dir", default="data/hf_cache", help="HF dataset cache directory")
    p.add_argument("--test-split", default="test")
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="validation")

    # Model
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to CNNSpot .pth file (required for zero_shot mode)",
    )

    # Inference
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-samples", type=int, default=None, help="Subset size for smoke tests")

    # Training hyperparameters (Mode B only)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-patience", type=int, default=3, help="ReduceLROnPlateau patience")
    p.add_argument("--early-stopping-patience", type=int, default=5)
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("data/checkpoints/cnnspot"),
        help="Where to save the best retrained checkpoint",
    )
    p.add_argument(
        "--checkpoint-name",
        default="cnnspot_synthclic_retrained",
        help="Filename stem for the retrained checkpoint (set per train dataset to avoid clobbering)",
    )

    # Output
    p.add_argument("--output-dir", type=Path, default=Path("results/e1_forensic"))
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Log to W&B (project set via WANDB_PROJECT, default: clip-cues)",
    )

    return p.parse_args()


def save_results(output_dir: Path, run_name: str, results: dict, run_id: str) -> None:
    """Persist one eval's metrics + raw predictions under results/<experiment>/<run>/<run_id>/.

    ``output_dir`` (default results/e1_forensic) is interpreted as ``<base>/<experiment>``, so
    each run lands in results/e1_forensic/<run_name>/<run_id>/{metrics.json,predictions.npz}.
    """
    metrics = {k: v for k, v in results.items() if not isinstance(v, (np.ndarray, list))}
    metrics["per_source_ap"] = results["per_source_ap"]
    # NPZ — predictions + labels + sources for later analysis (E4)
    arrays = {
        "predictions": results["predictions"],
        "labels": results["labels"],
        "sources": np.array(results["sources"]),
    }
    run_dir = save_run_results(
        experiment=output_dir.name,
        run=run_name,
        metrics=metrics,
        arrays=arrays,
        base=output_dir.parent,
        run_id=run_id,
    )
    print(f"Results saved to {run_dir}/")


def log_to_wandb(run, results: dict, eval_label: str | None = None) -> None:
    """Log metrics; when eval_label is given, namespace under xeval/<eval_label>/ for the matrix."""
    ns = f"xeval/{eval_label}" if eval_label else "test"
    flat = {
        f"{ns}/mAP": results["mAP"],
        f"{ns}/auroc": results["auroc"],
        f"{ns}/accuracy": results["accuracy"],
        f"{ns}/overall_ap": results["overall_ap"],
    }
    for src, ap in results["per_source_ap"].items():
        flat[f"{ns}/ap/{src}"] = ap
    run.log(flat)


def evaluate_on_datasets(
    model, eval_datasets, args, device, wandb_run, train_label: str, run_id: str
) -> dict:
    """Evaluate a trained/zero-shot model on each dataset's test split (cross-dataset matrix row).

    Returns {eval_label: results}. Each cell is saved (json+npz) and logged to W&B namespaced by
    the eval dataset, so a downstream export can assemble a train×eval matrix.
    """
    matrix_row: dict[str, dict] = {}
    for eval_name in eval_datasets:
        eval_label = eval_name.split("/")[-1]
        print(f"\nLoading eval dataset: {eval_name} ...")
        eval_ds = get_dataset(eval_name, cache_dir=args.cache_dir)
        test_split = eval_ds[args.test_split]
        print(
            f"Evaluating {train_label} -> {eval_label} "
            f"({len(test_split)} samples, max={args.eval_max_samples}) ..."
        )
        results = evaluate(
            model,
            test_split,
            device=device,
            batch_size=args.batch_size,
            max_samples=args.eval_max_samples,
        )
        print_eval_results(results, title=f"{train_label} -> {eval_label}")
        save_results(args.output_dir, f"{train_label}__to__{eval_label}", results, run_id)
        if wandb_run is not None:
            log_to_wandb(wandb_run, results, eval_label=eval_label)
        matrix_row[eval_label] = {
            "mAP": results["mAP"],
            "auroc": results["auroc"],
            "accuracy": results["accuracy"],
            "overall_ap": results["overall_ap"],
        }
    return matrix_row


def main() -> None:
    args = parse_args()
    run_id = make_run_id()
    device = torch.device(args.device)

    if args.mode == "zero_shot" and args.checkpoint is None:
        raise ValueError("--checkpoint is required for zero_shot mode")

    # Train/source label + which datasets to evaluate on (cross-dataset matrix row).
    is_combined = args.mode == "retrain" and args.dataset == "combined"
    if args.mode == "zero_shot":
        train_label = "cnnspot-progan-zeroshot"
    elif is_combined:
        train_label = "combined"
    else:
        train_label = args.dataset.split("/")[-1]
    # Combined has no single-name test split → evaluate on the individual test sets (matrix columns).
    default_eval = COMBINED_PARTS if is_combined else [args.dataset]
    eval_datasets = (
        [d.strip() for d in args.eval_datasets.split(",") if d.strip()]
        if args.eval_datasets
        else default_eval
    )

    # ── W&B init ──────────────────────────────────────────────────────────────
    wandb_run = None
    run_name = f"e1-cnnspot-{train_label}_{run_id}"
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            group="forensics_xdataset",
            config=vars(args) | {"train_label": train_label},
            tags=["e1", "cnnspot", args.mode, "cross-dataset"],
        )

    # ── Mode A: zero-shot ─────────────────────────────────────────────────────
    if args.mode == "zero_shot":
        print(f"Loading CNNSpot checkpoint: {args.checkpoint}")
        model = build_model(args.checkpoint)
        model.to(device)

    # ── Mode B: retrain on --dataset ──────────────────────────────────────────
    elif args.mode == "retrain":
        wandb_logger = None
        if wandb_run is not None:
            from lightning.pytorch.loggers import WandbLogger

            wandb_logger = WandbLogger(experiment=wandb_run)

        if is_combined:
            print("Building combined train dataset (synthclic + synthbuster-plus + cnnspot) ...")
            dataset = build_combined_dataset(args.cache_dir)
        else:
            print(f"Loading train dataset: {args.dataset} ...")
            dataset = get_dataset(args.dataset, cache_dir=args.cache_dir)
        model, best_ckpt = train(
            hf_dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            lr=args.lr,
            lr_patience=args.lr_patience,
            early_stopping_patience=args.early_stopping_patience,
            checkpoint_dir=args.checkpoint_dir,
            ckpt_filename=args.checkpoint_name,
            max_samples=args.max_samples,
            wandb_logger=wandb_logger,
        )
        print(f"\nBest checkpoint: {best_ckpt}")
        if wandb_run is not None:
            artifact = wandb.Artifact(f"e1-cnnspot-{train_label}", type="model")
            artifact.add_file(str(best_ckpt))
            wandb_run.log_artifact(artifact)

    # ── Cross-dataset evaluation (one matrix row: train_label -> each eval dataset) ──
    matrix_row = evaluate_on_datasets(
        model, eval_datasets, args, device, wandb_run, train_label, run_id
    )
    print("\nMatrix row (mAP):", {k: round(v["mAP"], 4) for k, v in matrix_row.items()})

    if wandb_run is not None:
        wandb_run.summary.update(
            {"train_label": train_label}
            | {f"matrix/{e}/mAP": m["mAP"] for e, m in matrix_row.items()}
            | {f"matrix/{e}/auroc": m["auroc"] for e, m in matrix_row.items()}
        )
        wandb_run.finish()


if __name__ == "__main__":
    main()
