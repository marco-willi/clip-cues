#!/usr/bin/env python
"""Faithful end-to-end CLIP-head training (E3 primary + E5 ablation).

Unlike the cached-embedding path (scripts/run/run_linear_probe.py), this re-encodes **augmented**
images through the **frozen** CLIP backbone every epoch — matching the published training protocol
(docs/initial_submission.tex Appendix): RandomResizedCrop(scale=(0.5,1.0))->512 train aug (none at
val/test), **bf16-mixed** precision, Adam(weight_decay=0.01) coupled L2, label_smoothing=0.1, seed
123, max 200 epochs with early stopping (patience 5) on validation cross-entropy. Only the head
trains; the backbone is frozen (features under no_grad).

Head variants (``--head``) share the architecture input -> Linear(in, K) -> logit and differ only
in the auxiliary loss, so this one trainer covers:
  * E3 primary  : ``--head activation_ortho`` (the paper's de-correlated head, K=8, lambda=0.33),
                  trained per dataset and evaluated across the train x eval matrix.
  * E5 ablation : run ``--head`` in {none, activation_ortho, weight_ortho} on one dataset; each run
                  additionally reports weight- and activation-space orthogonality (off-diagonal Gram
                  mass) so the variants can be compared at equal detection performance.

Eval is augmentation-free (test transforms) and reports Convention-A mAP (per-generator mean AP vs
real) plus pooled AP / AUROC for transparency.

Usage:
    # E3 primary (one matrix row)
    python scripts/run/train_clip_head.py --head activation_ortho --backbone clip_large_patch14 \
        --dataset synthclic --eval-datasets synthclic,synthbuster-plus,cnnspot
    # E5 ablation (one variant; the driver runs all three on synthclic)
    python scripts/run/train_clip_head.py --head weight_ortho --dataset synthclic \
        --results-experiment e5_orthogonality --group e5_orthogonality
"""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from clip_cues.classification_head import (
    ActivationOrthogonalityHead,
    ClassificationHead,
    LinearHead,
)
from clip_cues.dataset import get_dataset
from clip_cues.feature_extractor import EXTRACTOR_CLASSES
from clip_cues.transforms import Transforms
from clip_cues_research.analysis.metrics import detection_metrics, pairing_for_dataset
from clip_cues_research.analysis.orthogonality import orthogonality_score
from clip_cues_research.results import make_run_id, save_run_results

# Head variants that carry an orthogonality penalty (E5) — `linear`/`none` carry none.
ORTHO_HEADS = {"activation_ortho", "weight_ortho"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--backbone", default="clip_large_patch14", choices=list(EXTRACTOR_CLASSES))
    p.add_argument("--dataset", default="synthclic", help="Train dataset (short name or HF id)")
    p.add_argument(
        "--head",
        default="activation_ortho",
        choices=["linear", "none", "activation_ortho", "weight_ortho"],
        help="Head/aux-loss variant. Paper main detector = activation_ortho (K=8, lambda=0.33).",
    )
    p.add_argument("--latent-dim", type=int, default=8, help="K: width of the L1 layer (paper: 8)")
    p.add_argument("--ortho-weight", type=float, default=0.33, help="lambda for the ortho penalty")
    p.add_argument(
        "--eval-datasets",
        default=None,
        help="Comma-separated datasets to evaluate on (default: the train dataset)",
    )
    p.add_argument("--cache-dir", default="data/hf_cache")
    p.add_argument("--crop-size", type=int, default=512, help="RandomResizedCrop size (train aug)")
    p.add_argument(
        "--eval-max-test",
        type=int,
        default=None,
        help="Cap each eval test split (FINAL RUN: leave unset = full test, incl. cnnspot 108,310)",
    )
    p.add_argument(
        "--precision",
        default="bf16-mixed",
        choices=["bf16-mixed", "fp32"],
        help="Paper used bf16-mixed; fp32 available for debugging/CPU.",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)  # paper: Adam coupled L2, wd=0.01
    p.add_argument("--label-smoothing", type=float, default=0.1)  # paper: 0.1
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)  # paper: max_epochs 200 (+ early stopping)
    p.add_argument("--early-stopping-patience", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--seed", type=int, default=123)  # paper: 123
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--results-experiment",
        default="e3_xdataset_aug",
        help="results/<this>/ subtree to write into (E5 driver passes e5_orthogonality).",
    )
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "clip-cues"))
    p.add_argument("--group", default="e3_xdataset_aug")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def amp_dtype_for(precision: str, device: torch.device):
    """bf16 autocast dtype on CUDA for 'bf16-mixed'; None (fp32) otherwise."""
    if precision == "bf16-mixed" and device.type == "cuda":
        return torch.bfloat16
    return None


def build_head(head: str, input_dim: int, k: int, lam: float):
    """input -> Linear(in, k) -> logit; only the auxiliary loss differs across variants."""
    if head == "linear":
        return LinearHead(input_dim, num_classes=1)
    if head == "activation_ortho":
        return ActivationOrthogonalityHead(
            input_dim, layer_dims=[k], non_linear=False, loss_weight_ortho=lam
        )
    # `none` and `weight_ortho` share ClassificationHead; `none` zeroes the weight penalty.
    return ClassificationHead(
        input_dim,
        layer_dims=[k],
        non_linear=False,
        loss_weight_ortho=(lam if head == "weight_ortho" else 0.0),
    )


def l1_weight(head) -> torch.Tensor | None:
    """The W_L1 layer weight (k, input_dim) for ortho heads; None for the plain LinearHead."""
    return head.layers[-1].weight if hasattr(head, "layers") else None


class _ImageDataset(Dataset):
    """Apply a per-image transform (train=aug, eval=plain) to a HF split.

    Returns ``(tensor, label, source, image_id)`` — ``source`` (the generator) drives the
    per-generator (Convention A) mAP; ``image_id`` makes dumped predictions traceable.
    """

    def __init__(self, hf_split, transform):
        self.split = hf_split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int):
        ex = self.split[idx]
        return (
            self.transform(ex["image"].convert("RGB")),
            float(ex["label"]),
            str(ex.get("source", "unknown")),
            str(ex.get("image_id", idx)),
        )


def encode(backbone, images, device, amp_dtype) -> torch.Tensor:
    """Frozen-backbone features under bf16 autocast (paper: bf16-mixed). Returned as fp32."""
    ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None
        else nullcontext()
    )
    with torch.no_grad(), ctx:
        feats = backbone(images.to(device))["extracted_features"]
    return feats.float()


@torch.no_grad()
def predict(backbone, head, loader, device, amp_dtype) -> pd.DataFrame:
    """Per-image predictions: frame with ``image_id``, ``label`` (0/1), ``score``, ``source``."""
    head.eval()
    ids: list[str] = []
    ys: list[int] = []
    scores: list[float] = []
    srcs: list[str] = []
    for images, labels, sources, image_ids in loader:
        logits = head(encode(backbone, images, device, amp_dtype))["logits"].view(-1)
        scores.extend(torch.sigmoid(logits.float()).cpu().numpy().tolist())
        ys.extend(labels.view(-1).cpu().numpy().astype(int).tolist())
        srcs.extend(list(sources))
        ids.extend(list(image_ids))
    return pd.DataFrame({"image_id": ids, "label": ys, "score": scores, "source": srcs})


@torch.no_grad()
def activation_orthogonality(backbone, head, loader, device, amp_dtype) -> float:
    """Off-diagonal Gram mass of L2-normalised L1 activations (lower=more orthogonal). E5 metric."""
    head.eval()
    acts = []
    for images, _, _, _ in loader:
        feats = encode(backbone, images, device, amp_dtype)
        out = head(feats, output_distilled_representations=True)
        acts.append(out["distilled_representations"].float())
    a = F.normalize(torch.cat(acts), dim=0)
    gram = a.T @ a
    off = gram - torch.diag(torch.diag(gram))
    return float(torch.linalg.norm(off, ord="fro"))


def _val_cross_entropy(pred_df: pd.DataFrame, label_smoothing: float) -> float:
    """Mean label-smoothed BCE from a predictions frame (the paper's early-stopping metric)."""
    import numpy as np

    p = pred_df["score"].to_numpy().clip(1e-7, 1 - 1e-7)
    y = pred_df["label"].to_numpy().astype(float)
    ys = y * (1 - label_smoothing) + (1 - y) * label_smoothing
    return float(-(ys * np.log(p) + (1 - ys) * np.log(1 - p)).mean())


def score_predictions(pred_df: pd.DataFrame, dataset_label: str) -> dict[str, float]:
    """Convention-A metrics for one eval dataset (mAP) plus pooled AP/AUROC for transparency.

    ``source`` is carried on the predictions frame (attached by position in ``predict``), so no
    ``image_id`` merge is performed — ``image_id`` is non-unique in SynthCLIC/SynthBuster.
    """
    pairing = pairing_for_dataset(dataset_label)
    bundle = detection_metrics(pred_df, real_pairing=pairing)
    y, s = pred_df["label"].to_numpy(), pred_df["score"].to_numpy()
    auroc = float(roc_auc_score(y, s)) if 0 < int(y.sum()) < len(y) else float("nan")
    return {
        "mAP": bundle["mAP"],
        "pooled_ap": bundle["pooled_ap"],
        "auroc": auroc,
        "real_pairing": pairing,
        "n_generators": bundle["n_generators"],
    }


def main() -> None:
    args = parse_args()
    run_id = make_run_id()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = amp_dtype_for(args.precision, device)

    # Frozen backbone + train/eval transforms (aug only on train).
    extractor = EXTRACTOR_CLASSES[args.backbone](args.cache_dir)
    extractor.freeze()
    extractor.model.to(device).eval()
    backbone = extractor.model
    tfm = Transforms(extractor.transforms, random_crop_size=args.crop_size)
    train_tf, eval_tf = tfm._train_transforms, tfm._test_transforms

    train_dd = get_dataset(args.dataset, cache_dir=args.cache_dir)
    train_loader = DataLoader(
        _ImageDataset(train_dd["train"], train_tf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        _ImageDataset(train_dd["validation"], eval_tf),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    eval_names = (
        [d.strip() for d in args.eval_datasets.split(",") if d.strip()]
        if args.eval_datasets
        else [args.dataset]
    )
    train_label = f"{args.head}/{args.backbone}/{args.dataset}"

    import wandb

    wandb.init(
        project=args.wandb_project,
        group=args.group,
        name=f"e2e-{args.head}-{args.backbone}-{args.dataset}_{run_id}",
        config=vars(args) | {"train_label": train_label, "augment": True},
        mode="disabled" if args.no_wandb else None,
    )

    head = build_head(args.head, extractor.output_dim, args.latent_dim, args.ortho_weight).to(
        device
    )
    # Paper: torch.optim.Adam (coupled L2) with weight_decay; lr default 1e-3.
    opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ls = args.label_smoothing

    # Model selection: min val cross-entropy, early stopping patience 5 (paper).
    best_ce, best_state, since_improve = float("inf"), None, 0
    for epoch in range(args.epochs):
        head.train()
        for images, labels, _, _ in train_loader:
            feats = encode(backbone, images, device, amp_dtype)  # frozen backbone, augmented images
            ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_dtype is not None
                else nullcontext()
            )
            with ctx:
                logits = head(feats)["logits"].view(-1)
                y = labels.to(device).view(-1)
                y_smooth = y * (1 - ls) + (1 - y) * ls
                loss = F.binary_cross_entropy_with_logits(logits.float(), y_smooth)
                loss = loss + head.compute_loss().get(
                    "orthogonality", torch.zeros((), device=device)
                )
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_ce = _val_cross_entropy(predict(backbone, head, val_loader, device, amp_dtype), ls)
        print(f"epoch {epoch + 1}/{args.epochs}: val cross_entropy={val_ce:.4f}")
        if not args.no_wandb:
            wandb.log({"epoch": epoch, "val/cross_entropy": val_ce})
        if val_ce < best_ce:
            best_ce, since_improve = val_ce, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            since_improve += 1
            if since_improve >= args.early_stopping_patience:
                print(f"early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        head.load_state_dict(best_state)

    val_metrics = score_predictions(
        predict(backbone, head, val_loader, device, amp_dtype), args.dataset
    )

    # E5 orthogonality measurements on the head (only meaningful for the ortho-capable heads).
    w = l1_weight(head)
    ortho_scores: dict[str, float] = {}
    if args.head in ORTHO_HEADS or args.head == "none":
        if w is not None:
            ortho_scores["weight_ortho_score"] = orthogonality_score(w)
        ortho_scores["activation_ortho_score"] = activation_orthogonality(
            backbone, head, val_loader, device, amp_dtype
        )

    # ── Cross-dataset evaluation (augmentation-free) ──
    preds_dir = Path("results") / args.results_experiment / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    matrix_row: dict[str, dict] = {}
    for eval_name in eval_names:
        eval_label = eval_name.split("/")[-1]
        dd = get_dataset(eval_name, cache_dir=args.cache_dir)
        test_split = dd["test"]
        if args.eval_max_test is not None and len(test_split) > args.eval_max_test:
            test_split = test_split.shuffle(seed=42).select(range(args.eval_max_test))
        loader = DataLoader(
            _ImageDataset(test_split, eval_tf),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        preds = predict(backbone, head, loader, device, amp_dtype)
        test = score_predictions(preds, eval_label)
        matrix_row[eval_label] = {**test, "n": len(test_split)}
        print(
            f"  {train_label} -> {eval_label}: mAP={test['mAP']:.4f} "
            f"(pooled_ap={test['pooled_ap']:.4f}, pairing={test['real_pairing']}) "
            f"auroc={test['auroc']:.4f}"
        )

        # Per-image predictions parquet (label/score/source) — consumed by E4 cross-family analysis.
        preds = preds.assign(
            head=args.head,
            backbone=args.backbone,
            train_dataset=args.dataset,
            eval_dataset=eval_label,
        )
        preds.to_parquet(
            preds_dir
            / f"{args.head}__{args.backbone}__{args.dataset}__to__{eval_label}__{run_id}.parquet"
        )
        save_run_results(
            args.results_experiment,
            f"{args.head}__{args.backbone}__{args.dataset}__to__{eval_label}",
            {
                "head": args.head,
                "backbone": args.backbone,
                "train_dataset": args.dataset,
                "eval_dataset": eval_label,
                "augment": True,
                "precision": args.precision,
                "val/auroc": val_metrics["auroc"],
                **ortho_scores,
                **test,
            },
            arrays={
                "score": preds["score"].to_numpy(),
                "label": preds["label"].to_numpy(),
                "source": preds["source"].to_numpy().astype(str),
            },
            run_id=run_id,
        )

    summary = {
        "train_label": train_label,
        "head": args.head,
        "backbone": args.backbone,
        "dataset": args.dataset,
        "augment": True,
        "precision": args.precision,
        "val/mAP": val_metrics["mAP"],
        "val/pooled_ap": val_metrics["pooled_ap"],
        "val/auroc": val_metrics["auroc"],
        **ortho_scores,
    }
    if args.dataset in matrix_row:
        summary["test/mAP"] = matrix_row[args.dataset]["mAP"]
        summary["test/pooled_ap"] = matrix_row[args.dataset]["pooled_ap"]
        summary["test/auroc"] = matrix_row[args.dataset]["auroc"]
    for e, m in matrix_row.items():
        summary[f"matrix/{e}/mAP"] = m["mAP"]
        summary[f"matrix/{e}/pooled_ap"] = m["pooled_ap"]
        summary[f"matrix/{e}/auroc"] = m["auroc"]
    wandb.log(summary)
    wandb.summary.update(summary)
    print("\nMatrix row (mAP):", {e: round(m["mAP"], 4) for e, m in matrix_row.items()})
    if ortho_scores:
        print("Orthogonality:", {k: round(v, 4) for k, v in ortho_scores.items()})
    wandb.finish()


if __name__ == "__main__":
    main()
