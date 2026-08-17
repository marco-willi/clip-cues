#!/usr/bin/env python
"""E8 Step 6a: cross-benchmark interpretation transfer (SynthCLIC -> CommunityForensics).

Reviewer 1's literal ask (point 5) is to validate findings on a larger benchmark. This is the
*interpretability* counterpart: take an **existing SynthCLIC-trained head** and ask whether the
directions/concepts that drive its SynthCLIC decisions stay discriminative on CommunityForensics —
broken down by generator architecture (GAN / LatDiff / PixDiff / Commercial). If the discriminative
directions go *silent* exactly on the families where CLIP detection collapses (E7: GAN 0.37 AP vs
LatDiff 0.88), that is the mechanistic explanation linking E4 + E7 + E8.

Eval-only, no retraining. It reuses the *same cached-embeddings + checkpoint* convention as the rest of
E8: the head weights are read straight from the checkpoint ``state_dict`` and applied to cached
embeddings — so the only new GPU cost upstream is a one-time CF embedding-extraction pass (the head
itself never touches images). Per-image ``architecture`` labels come from the CF embeddings frame (or are
merged from the E7 parquet).

Inputs (all on a box that has run E7 + extraction):
  * ``--checkpoint`` — a SynthCLIC-trained head: ``clip_orthogonal_synthclic.ckpt`` (ortho) or
    ``cm_antonyms_synthclic.ckpt`` (concept).
  * ``--source-embeddings`` — SynthCLIC cached embeddings (the source domain).
  * ``--target-embeddings`` — CommunityForensics cached embeddings (must carry an ``architecture`` column
    in the frame, or pass ``--arch-parquet`` to merge it from the E7 predictions parquet by ``image_id``).
  * concept mode additionally needs ``--text-embeddings`` (the vocabulary used at train time).

Output: ``results/e8_interpretability_stability/transfer/<run_id>/transfer.csv`` — per-architecture
``diagnostic_agreement`` (rank) + ``selection_survival`` (top-K), plus an ``overall`` row.

    python scripts/run/run_interpretation_transfer.py --mode ortho \
        --checkpoint data/checkpoints/clip_orthogonal_synthclic.ckpt \
        --source-embeddings data/embeddings/synthclic_l14_local.pkl \
        --target-embeddings data/embeddings/communityforensics_l14.pkl --top-k 4
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clip_cues.classification_head import ActivationOrthogonalityHead
from clip_cues.concept_modeling.networks import ConceptBottleneckModel
from clip_cues_research.analysis.interpretation_stability import (
    class_mean_difference_importance,
    diagnostic_agreement,
    selection_survival,
    transfer_table,
)
from clip_cues_research.results import make_run_id


def _load_embeddings(path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Load a cached-embeddings pkl ``{"df": frame, "embeddings": array}`` → (embeddings, frame)."""
    with open(path, "rb") as f:
        cache = pickle.load(f)
    return np.asarray(cache["embeddings"], dtype=np.float32), cache["df"].reset_index(drop=True)


def _strip_prefix(state_dict: dict, key_contains: str) -> dict:
    """Sub-dict of params whose key contains ``key_contains``, with everything up to it stripped."""
    out = {}
    for k, v in state_dict.items():
        if key_contains in k:
            out[k.split(key_contains, 1)[1]] = v
    return out


@torch.no_grad()
def _ortho_contributions(checkpoint: dict, emb: np.ndarray) -> np.ndarray:
    """Per-image per-direction logit contributions ``activation * w_logit`` for the orthogonal head."""
    sd = checkpoint["state_dict"]
    w_l1 = sd["model.classification_head.layers.0.weight"]  # (k, d)
    k, d = w_l1.shape
    head = ActivationOrthogonalityHead(input_dim=d, layer_dims=[k], non_linear=False)
    head.load_state_dict(_strip_prefix(sd, "classification_head."))
    head.eval()
    x = torch.from_numpy(emb)
    acts = head(x, output_distilled_representations=True)["distilled_representations"]  # (n, k)
    w_logit = head.to_logits.weight.view(1, -1)  # (1, k)
    return (acts * w_logit).cpu().numpy()


@torch.no_grad()
def _concept_contributions(
    checkpoint: dict, text_embeddings_path: Path, emb: np.ndarray
) -> np.ndarray:
    """Per-image per-concept logit contributions for the concept model (forward output)."""
    text = torch.load(text_embeddings_path)
    model = ConceptBottleneckModel(text["embeddings"])
    sd = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    out = model(torch.from_numpy(emb))
    return out["per_concept_logit_contribution"].cpu().numpy()


def _attach_architecture(df: pd.DataFrame, arch_parquet: Path | None) -> pd.DataFrame:
    """Ensure the frame has an ``architecture`` column (merge from the E7 parquet by image_id if needed)."""
    if "architecture" in df.columns:
        return df
    if arch_parquet is None:
        raise SystemExit(
            "target frame lacks an 'architecture' column; pass --arch-parquet to merge it"
        )
    arch = pd.read_parquet(arch_parquet)[["image_id", "architecture"]].drop_duplicates("image_id")
    return df.merge(arch, on="image_id", how="left")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", required=True, choices=["ortho", "concept"])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--source-embeddings", type=Path, required=True, help="SynthCLIC cached embeddings"
    )
    p.add_argument(
        "--target-embeddings", type=Path, required=True, help="CommunityForensics cached embeddings"
    )
    p.add_argument(
        "--text-embeddings", type=Path, help="concept mode: vocabulary text-embedding .pt"
    )
    p.add_argument(
        "--arch-parquet", type=Path, help="E7 predictions parquet to merge 'architecture' from"
    )
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--label-col", default="label")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    src_emb, src_df = _load_embeddings(args.source_embeddings)
    tgt_emb, tgt_df = _load_embeddings(args.target_embeddings)
    tgt_df = _attach_architecture(tgt_df, args.arch_parquet)

    if args.mode == "ortho":
        src_contrib = _ortho_contributions(ckpt, src_emb)
        tgt_contrib = _ortho_contributions(ckpt, tgt_emb)
    else:
        if args.text_embeddings is None:
            raise SystemExit("--text-embeddings is required for --mode concept")
        src_contrib = _concept_contributions(ckpt, args.text_embeddings, src_emb)
        tgt_contrib = _concept_contributions(ckpt, args.text_embeddings, tgt_emb)

    # Importance on the source domain (SynthCLIC), and on each target architecture group.
    src_imp = class_mean_difference_importance(src_contrib, src_df[args.label_col].to_numpy())
    tgt_labels = tgt_df[args.label_col].to_numpy()
    # Shared-reals convention (mirrors per-generator mAP): each synthetic architecture is diagnosed
    # against {that architecture's fakes} ∪ {all reals}, so the class-mean difference has a real baseline.
    real_rows = np.flatnonzero(tgt_labels == 0)
    by_group = {}
    for arch, idx in tgt_df[tgt_labels == 1].groupby("architecture").groups.items():
        fake_rows = np.asarray(list(idx))
        rows = np.concatenate([fake_rows, real_rows])
        by_group[str(arch)] = class_mean_difference_importance(tgt_contrib[rows], tgt_labels[rows])

    table = transfer_table(src_imp, by_group, k=args.top_k)
    # overall (all target images pooled) row for context
    overall_imp = class_mean_difference_importance(tgt_contrib, tgt_labels)
    overall = pd.DataFrame(
        [
            {
                "group": "overall",
                "diagnostic_agreement": diagnostic_agreement(src_imp, overall_imp),
                "selection_survival": selection_survival(src_imp, overall_imp, args.top_k),
            }
        ]
    )
    table = pd.concat([table, overall], ignore_index=True)

    out_dir = Path("results") / "e8_interpretability_stability" / "transfer" / make_run_id()
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "transfer.csv", index=False)
    print(table.to_string(index=False))
    print(f"\nResults saved to {out_dir}/transfer.csv")


if __name__ == "__main__":
    main()
