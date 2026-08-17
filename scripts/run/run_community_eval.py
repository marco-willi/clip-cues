#!/usr/bin/env python
"""E7 Option A: evaluate SynthCLIC/CNNSpot detectors on CommunityForensics-Eval (eval-only).

Loads the persisted checkpoints in ``data/checkpoints/`` and scores them on CommunityForensics-Eval
via the CF adapter (no retraining). Writes full-metadata parquet predictions + metrics per detector
under ``results/e7_community_eval/`` (W&B group ``e7_community_eval``), feeding the export step's
per-architecture / per-generator / per-real-source tables.

Usage:
    # smoke test (few samples, CPU) — full run needs the box (CompEval = 413 files)
    python scripts/run/run_community_eval.py --detectors cnnspot_progan --max-samples 64 --device cpu --no-wandb
    # full Option A
    python scripts/run/run_community_eval.py --device cuda
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from clip_cues_research.community_eval import EXPERIMENT, score_cf_split_to_parquet
from clip_cues_research.datasets import CF_EVAL, load_community_forensics
from clip_cues_research.results import make_run_id, save_run_results

# detector name -> (kind, checkpoint). `kind` selects how the scorer is built.
DETECTORS = {
    "clip_orthogonal_synthclic": ("clip", "data/checkpoints/clip_orthogonal_synthclic.ckpt"),
    "cnnspot_synthclic": ("forensic", "data/checkpoints/cnnspot/cnnspot_synthclic_retrained.ckpt"),
    "cnnspot_synthbuster": (
        "forensic",
        "data/checkpoints/cnnspot/cnnspot_synthbuster_retrained.ckpt",
    ),
    "cnnspot_progan": ("forensic", "data/checkpoints/cnnspot/blur_jpg_prob0.5.pth"),
    "cnnspot_combined": (
        "forensic",
        "data/checkpoints/cnnspot/cnnspot_combined_retrained.ckpt",
    ),
}


def build_score_fn(kind: str, checkpoint: str, device: torch.device):
    """Return a ``list[PIL] -> probs(np.ndarray)`` callable for the given detector kind."""
    if kind == "clip":
        from clip_cues import load_clip_classifier

        model = load_clip_classifier(checkpoint)
        model.to(device).eval()
        return lambda pils: model.predict_batch(pils)

    if kind == "forensic":
        from clip_cues_research.forensics.patch_cnn import (
            build_model,
            eval_transform,
            predict_probs,
        )

        model = build_model(checkpoint).to(device).eval()
        return lambda pils: (
            predict_probs(model, torch.stack([eval_transform(p) for p in pils]), device)
            .cpu()
            .numpy()
        )

    raise ValueError(f"unknown detector kind: {kind}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", default=CF_EVAL, help="CommunityForensics HF id")
    p.add_argument("--split", default="CompEval")
    p.add_argument("--cache-dir", default="data/hf_cache")
    p.add_argument("--detectors", nargs="+", default=list(DETECTORS), choices=list(DETECTORS))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N scored batches (0 disables periodic logs).",
    )
    p.add_argument(
        "--max-samples", type=int, default=None, help="smoke-test cap (default: full split)"
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "clip-cues"))
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_id = make_run_id()
    device = torch.device(args.device)

    print(f"Loading {args.dataset} [{args.split}] via CF adapter (run_id={run_id}) ...")
    split = load_community_forensics(
        args.dataset, args.split, cache_dir=args.cache_dir, streaming=False
    )
    print(f"  {len(split)} samples | scoring detectors: {args.detectors}")

    wb = None
    if not args.no_wandb:
        import wandb

        wb = wandb.init(
            project=args.wandb_project,
            group=EXPERIMENT,
            name=f"{EXPERIMENT}_{run_id}",
            config=vars(args) | {"run_id": run_id},
        )

    for name in args.detectors:
        kind, ckpt = DETECTORS[name]
        if not Path(ckpt).exists():
            print(f"!! {name}: checkpoint missing ({ckpt}) — skipping")
            continue
        print(f"\n=== {name} ({kind}) ===")
        score_fn = build_score_fn(kind, ckpt, device)
        pred_path = Path("results") / EXPERIMENT / "predictions" / f"{name}__{run_id}.parquet"
        metrics = score_cf_split_to_parquet(
            split,
            score_fn,
            detector=name,
            out_path=pred_path,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            progress_every=args.progress_every,
        ) | {
            "detector": name,
            "kind": kind,
            "checkpoint": ckpt,
            "run_id": run_id,
        }
        run_dir = save_run_results(EXPERIMENT, name, metrics, run_id=run_id)
        print(
            f"  AP={metrics.get('overall_ap', float('nan')):.4f} "
            f"AUROC={metrics.get('auroc', float('nan')):.4f} "
            f"acc={metrics['accuracy']:.4f} bAcc={metrics['balanced_accuracy']:.4f} "
            f"-> {run_dir}/ (predictions: {pred_path})"
        )
        if wb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    wb.summary[f"{name}/{k}"] = v

    if wb is not None:
        wb.finish()
    print(f"\nDone. Predictions: results/{EXPERIMENT}/predictions/  (export tables next)")


if __name__ == "__main__":
    main()
