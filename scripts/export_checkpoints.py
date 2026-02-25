#!/usr/bin/env python3
"""
Export Classification Head Checkpoints from Full Model Checkpoints.

This script extracts only the classification head weights from full Lightning
checkpoints (which include the CLIP model), creating much smaller checkpoint
files suitable for distribution.

The full checkpoints (~1.2 GB) contain both:
- CLIP feature extractor weights (from HuggingFace, ~1.2 GB)
- Classification head weights (~8 KB)

Since the CLIP model can be downloaded from HuggingFace, we only need to
distribute the tiny classification head weights.

Usage:
    python scripts/export_checkpoints.py --source /path/to/data --output ./checkpoints
"""

import argparse
from pathlib import Path

import torch


# Checkpoint mapping: (relative_path, output_name, model_type)
# model_type: "clip" for CLIP orthogonal, "concept" for concept models, "linear" for linear probes
# Paths are relative to the source_path (DATA_PATH)
CHECKPOINTS = [
    # CLIP Orthogonal models
    ("detection_via_clip/logs/clip/clip_v21/2025-04-13_19-59-30/checkpoints", "clip_orthogonal_cnnspot.ckpt", "clip"),
    ("detection_via_clip/logs/clip/clip_v22/2025-04-13_21-01-13/checkpoints", "clip_orthogonal_synthbuster.ckpt", "clip"),
    ("detection_via_clip/logs/clip/clip_v23/2025-04-13_21-01-22/checkpoints", "clip_orthogonal_synthclic.ckpt", "clip"),
    ("detection_via_clip/logs/clip/clip_v20/2025-04-13_21-01-29/checkpoints", "clip_orthogonal_combined.ckpt", "clip"),
    # Concept models (antonyms vocabulary)
    ("concept_modeling/logs/final_selection/cm_chatgptv4_antonyms_cnnspot_v1/2025-09-13_20-45-56/checkpoints", "cm_antonyms_cnnspot.ckpt", "concept"),
    ("concept_modeling/logs/final_selection/cm_chatgptv4_antonyms_synthbuster_v1/2025-09-11_23-40-01/checkpoints", "cm_antonyms_synthbuster.ckpt", "concept"),
    ("concept_modeling/logs/hyperparam_testing/cm_chatgptv4_antonyms_synthclic_v2/2025-09-11_21-11-48/checkpoints", "cm_antonyms_synthclic.ckpt", "concept"),
    ("concept_modeling/logs/final_selection/cm_chatgptv4_antonyms_combined_v1/2025-09-11_23-41-23/checkpoints", "cm_antonyms_combined.ckpt", "concept"),
    # Linear probe models (baseline)
    ("detection_via_clip/logs/linear_probe/linear_probe_cnnspot/2025-11-29_21-33-35/checkpoints", "linear_probe_cnnspot.ckpt", "linear"),
    ("detection_via_clip/logs/linear_probe/linear_probe_synthbuster/2025-11-29_21-46-16/checkpoints", "linear_probe_synthbuster.ckpt", "linear"),
    ("detection_via_clip/logs/linear_probe/linear_probe_synthclic/2025-11-29_21-46-16/checkpoints", "linear_probe_synthclic.ckpt", "linear"),
    ("detection_via_clip/logs/linear_probe/linear_probe_combined/2025-11-29_21-46-14/checkpoints", "linear_probe_combined.ckpt", "linear"),
]


def find_checkpoint_file(checkpoint_dir: Path) -> Path | None:
    """Find the checkpoint file in a directory."""
    ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
    if len(ckpt_files) > 1:
        print(f"  [WARNING] Multiple checkpoint files found in {checkpoint_dir}: {[f.name for f in ckpt_files]}")
    if not ckpt_files:
        return None
    # Return the most recent one (by name, which includes epoch)
    return sorted(ckpt_files)[-1]


def extract_classification_head(full_checkpoint_path: Path, model_type: str = "clip") -> dict:
    """Extract only classification head weights from a full checkpoint.

    Args:
        full_checkpoint_path: Path to the full Lightning checkpoint
        model_type: One of "clip", "concept", or "linear"

    Returns:
        Dictionary with only classification head weights
    """
    checkpoint = torch.load(full_checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)

    if model_type == "concept":
        # Concept models have model.W_concepts, model.W_classifier, and model.text_embeddings (buffer)
        head_keys = [k for k in state_dict.keys() if "W_concepts" in k or "W_classifier" in k or "text_embeddings" in k]
    else:
        # CLIP orthogonal and linear probe models have model.classification_head.*
        head_keys = [k for k in state_dict.keys() if "classification_head" in k]

    head_state_dict = {k: state_dict[k] for k in head_keys}

    return {
        "state_dict": head_state_dict,
        "model_type": model_type,
    }


def export_checkpoints(source_path: Path, output_path: Path, dry_run: bool = False):
    """Export all checkpoints.

    Args:
        source_path: Root data path containing logs
        output_path: Output directory for exported checkpoints
        dry_run: If True, only print what would be done
    """
    output_path.mkdir(parents=True, exist_ok=True)

    MODEL_TYPE_NAMES = {
        "clip": "CLIP Orthogonal",
        "concept": "Concept Model",
        "linear": "Linear Probe",
    }

    print("\n=== Exporting Checkpoints ===")
    for relative_path, output_name, model_type in CHECKPOINTS:
        ckpt_dir = source_path / relative_path
        ckpt_file = find_checkpoint_file(ckpt_dir)

        type_name = MODEL_TYPE_NAMES.get(model_type, model_type)

        if ckpt_file is None:
            print(f"  [SKIP] {output_name} ({type_name}): No checkpoint found in {ckpt_dir}")
            continue

        output_file = output_path / output_name
        if dry_run:
            print(f"  [DRY RUN] {output_name} ({type_name})")
            print(f"    Source: {ckpt_file}")
            print(f"    Output: {output_file}")
        else:
            print(f"  Exporting {output_name} ({type_name})...")
            head_ckpt = extract_classification_head(ckpt_file, model_type=model_type)
            torch.save(head_ckpt, output_file)
            print(f"    Saved: {output_file} ({output_file.stat().st_size / 1024:.1f} KB)")

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Export classification head checkpoints from full model checkpoints",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="/workspace/data",
        help="Root data path containing logs directories",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/checkpoints",
        help="Output directory for exported checkpoints",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually exporting",
    )

    args = parser.parse_args()

    export_checkpoints(
        source_path=Path(args.source),
        output_path=Path(args.output),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
