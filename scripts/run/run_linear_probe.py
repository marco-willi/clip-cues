#!/usr/bin/env python
"""E3 (Reviewer 1): train a linear probe on cached CLIP-backbone embeddings, with optional
cross-dataset evaluation.

The protocol is identical to the published linear probe: freeze CLIP -> extract pooler_output once
(scripts/extract/extract_embeddings.py) -> train a single linear layer (clip_cues.classification_head.
LinearHead). This script trains one probe on one (backbone, dataset) and evaluates it on the test
split of one or more datasets' cached embeddings — so it produces one row of a train×eval
cross-dataset matrix (the CLIP-side counterpart to the E1 forensic matrix).

Usage:
    # in-domain only (train + eval on the same dataset's embeddings)
    python scripts/run/run_linear_probe.py \
        --embeddings data/embeddings/synthclic_clip_base_patch16.pkl \
        --backbone clip_base_patch16 --dataset synthclic --epochs 200 --no-wandb

    # cross-dataset: train on synthclic, evaluate on all three (same backbone => same dim)
    python scripts/run/run_linear_probe.py \
        --embeddings data/embeddings/synthclic_clip_base_patch16.pkl \
        --backbone clip_base_patch16 --dataset synthclic \
        --eval-embeddings synthclic=data/embeddings/synthclic_clip_base_patch16.pkl,\
synthbuster-plus=data/embeddings/synthbuster-plus_clip_base_patch16.pkl,\
cnnspot=data/embeddings/cnnspot_clip_base_patch16.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from clip_cues.classification_head import LinearHead
from clip_cues.concept_modeling.dataset import CLIPFeatureDataset
from clip_cues_research.analysis.metrics import detection_metrics, pairing_for_dataset
from clip_cues_research.results import make_run_id, save_run_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings", type=Path, required=True, help="Train dataset embeddings (.pkl)")
    p.add_argument("--backbone", default="clip_large_patch14", help="Backbone label (for logging)")
    p.add_argument("--dataset", default="synthclic", help="Train dataset label")
    p.add_argument(
        "--eval-embeddings",
        default=None,
        help="Comma-separated label=path pairs to cross-evaluate on "
        "(default: in-domain only). Same backbone => same embedding dim.",
    )
    p.add_argument("--train-splits", nargs="+", default=["train"])
    p.add_argument("--val-splits", nargs="+", default=["validation"])
    p.add_argument("--test-splits", nargs="+", default=["test"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)  # paper: Adam coupled L2, wd=0.01
    p.add_argument("--label-smoothing", type=float, default=0.1)  # paper: 0.1
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--early-stopping-patience", type=int, default=5)  # paper: 5 (monitor val CE)
    p.add_argument("--seed", type=int, default=123)  # paper: 123
    p.add_argument("--device", default="cuda")
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "clip-cues"))
    p.add_argument("--group", default="e3_xdataset", help="W&B group")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument(
        "--save-checkpoint",
        type=Path,
        default=None,
        help="Save the trained head as a validate_model-compatible checkpoint (model_type=linear).",
    )
    return p.parse_args()


def load_feature_dataset(
    pkl_path: Path, splits: list[str]
) -> tuple[CLIPFeatureDataset, int, pd.DataFrame]:
    """Build a CLIPFeatureDataset from the given splits of a cached embeddings pkl.

    Each pkl holds a single dataset, so we filter by split only (not ds_name). Also returns the
    per-image metadata (``image_id``/``label``/``source``) so predictions can be scored with the
    per-generator (Convention A) mAP.
    """
    with open(pkl_path, "rb") as f:
        cache = pickle.load(f)
    df = cache["df"]
    emb = cache["embeddings"]
    idx = df["split"].isin(splits).values
    ds = CLIPFeatureDataset(
        torch.from_numpy(emb[idx, :]),
        torch.from_numpy(df.loc[idx, "label"].values).to(torch.float32),
        df.loc[idx, "image_id"].values,
    )
    meta = df.loc[idx, ["image_id", "label", "source"]].reset_index(drop=True)
    return ds, emb.shape[1], meta


@torch.no_grad()
def predict(head, loader, device) -> pd.DataFrame:
    """Per-image predictions: a frame with ``image_id``, ``label`` (0/1), ``score`` (P(synthetic))."""
    head.eval()
    ids: list[str] = []
    ys: list[int] = []
    scores: list[float] = []
    for emb, labels, image_ids in loader:
        logits = head(emb.to(device))["logits"].view(-1)
        scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        ys.extend(labels.view(-1).cpu().numpy().astype(int).tolist())
        ids.extend([str(i) for i in image_ids])
    return pd.DataFrame({"image_id": ids, "label": ys, "score": scores})


def _val_cross_entropy(pred_df: pd.DataFrame, label_smoothing: float) -> float:
    """Mean label-smoothed BCE from a predictions frame (the paper's early-stopping metric)."""
    p = pred_df["score"].to_numpy().clip(1e-7, 1 - 1e-7)
    y = pred_df["label"].to_numpy().astype(float)
    ys = y * (1 - label_smoothing) + (1 - y) * label_smoothing
    return float(-(ys * np.log(p) + (1 - ys) * np.log(1 - p)).mean())


def score_predictions(
    pred_df: pd.DataFrame, meta: pd.DataFrame, dataset_label: str
) -> tuple[dict[str, float], pd.DataFrame]:
    """Attach ``source`` and compute Convention-A metrics for one eval dataset.

    Returns (metrics, predictions_with_source). ``mAP`` is the paper's per-generator mean AP using
    the dataset-appropriate real-pairing rule; ``pooled_ap`` (the old SimpleMetrics quantity) and
    pooled ``auroc`` are kept alongside for transparency.

    NOTE: ``image_id`` is **not unique** in SynthCLIC/SynthBuster (the real + every generator share
    the same id), so ``source`` is attached **by position** — predictions come from an unshuffled
    loader in the same row order as ``meta`` — and the id order is asserted as a safety check. A
    merge on ``image_id`` would explode the frame and collapse per-generator mAP to pooled AP.
    """
    pred_df = pred_df.reset_index(drop=True)
    meta = meta.reset_index(drop=True)
    if (
        len(pred_df) != len(meta)
        or not (pred_df["image_id"].astype(str).values == meta["image_id"].astype(str).values).all()
    ):
        raise AssertionError(
            "prediction/meta row order mismatch — cannot attach source by position"
        )
    merged = pred_df.copy()
    merged["source"] = meta["source"].values
    pairing = pairing_for_dataset(dataset_label)
    bundle = detection_metrics(merged, real_pairing=pairing)
    y, s = merged["label"].to_numpy(), merged["score"].to_numpy()
    auroc = float(roc_auc_score(y, s)) if 0 < int(y.sum()) < len(y) else float("nan")
    vals = {
        "mAP": bundle["mAP"],
        "pooled_ap": bundle["pooled_ap"],
        "auroc": auroc,
        "real_pairing": pairing,
        "n_generators": bundle["n_generators"],
    }
    return vals, merged


def parse_eval_embeddings(
    spec: str | None, default_label: str, default_path: Path
) -> dict[str, Path]:
    if not spec:
        return {default_label: default_path}
    out: dict[str, Path] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        label, _, path = item.partition("=")
        out[label.strip()] = Path(path.strip())
    return out


def main() -> None:
    args = parse_args()
    run_id = make_run_id()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_ds, input_dim, _ = load_feature_dataset(args.embeddings, args.train_splits)
    val_ds, _, val_meta = load_feature_dataset(args.embeddings, args.val_splits)
    print(
        f"Backbone {args.backbone} | train dataset {args.dataset} | input_dim={input_dim} "
        f"| train={len(train_ds)} val={len(val_ds)}"
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    eval_targets = parse_eval_embeddings(args.eval_embeddings, args.dataset, args.embeddings)
    train_label = f"{args.backbone}/{args.dataset}"

    import wandb

    wandb.init(
        project=args.wandb_project,
        group=args.group,
        name=f"linprobe-{args.backbone}-{args.dataset}_{run_id}",
        config=vars(args) | {"input_dim": input_dim, "train_label": train_label},
        mode="disabled" if args.no_wandb else None,
    )

    head = LinearHead(input_dim=input_dim, num_classes=1).to(device)
    # Paper: torch.optim.Adam (coupled L2) with weight_decay; lr default 1e-3.
    opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ls = args.label_smoothing

    # Model selection: min val cross-entropy on the first val set, early stopping patience 5 (paper).
    best_ce, best_state, since_improve = float("inf"), None, 0
    for epoch in range(args.epochs):
        head.train()
        for emb, labels, _ in train_loader:
            opt.zero_grad()
            logits = head(emb.to(device))["logits"].view(-1)
            y = labels.to(device).view(-1)
            y_smooth = y * (1 - ls) + (1 - y) * ls
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_smooth)
            loss.backward()
            opt.step()
        val_ce = _val_cross_entropy(predict(head, val_loader, device), ls)
        if not args.no_wandb:
            wandb.log({"epoch": epoch, "val/cross_entropy": val_ce})
        if val_ce < best_ce:
            best_ce, since_improve = val_ce, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            since_improve += 1
            if since_improve >= args.early_stopping_patience:
                print(f"early stopping at epoch {epoch + 1} (val CE {best_ce:.4f})")
                break

    if best_state is not None:
        head.load_state_dict(best_state)  # evaluate the min-val-CE checkpoint

    # Final val metrics (Convention A) from the restored best checkpoint.
    val_metrics, _ = score_predictions(predict(head, val_loader, device), val_meta, args.dataset)

    # ── Cross-dataset evaluation: one matrix row (train_label -> each eval dataset) ──
    preds_dir = Path("results") / "e3_xdataset" / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    matrix_row: dict[str, dict] = {}
    for eval_label, eval_path in eval_targets.items():
        test_ds, dim, test_meta = load_feature_dataset(eval_path, args.test_splits)
        if dim != input_dim:
            raise ValueError(
                f"Embedding-dim mismatch for '{eval_label}' ({dim}) vs train ({input_dim}); "
                f"cross-eval requires the same backbone."
            )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)
        test, preds = score_predictions(predict(head, test_loader, device), test_meta, eval_label)
        matrix_row[eval_label] = {**test, "n": len(test_ds)}
        print(
            f"  {train_label} -> {eval_label}: mAP={test['mAP']:.4f} "
            f"(pooled_ap={test['pooled_ap']:.4f}, pairing={test['real_pairing']}) "
            f"auroc={test['auroc']:.4f}"
        )

        # Per-image predictions parquet (label/score/source) — consumed by E4 cross-family analysis.
        preds = preds.assign(
            backbone=args.backbone, train_dataset=args.dataset, eval_dataset=eval_label
        )
        preds.to_parquet(
            preds_dir / f"{args.backbone}__{args.dataset}__to__{eval_label}__{run_id}.parquet"
        )
        save_run_results(
            "e3_xdataset",
            f"{args.backbone}__{args.dataset}__to__{eval_label}",
            {
                "backbone": args.backbone,
                "train_dataset": args.dataset,
                "eval_dataset": eval_label,
                "input_dim": input_dim,
                "val/auroc": val_metrics["auroc"],
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
        "backbone": args.backbone,
        "dataset": args.dataset,
        "input_dim": input_dim,
        "val/mAP": val_metrics["mAP"],
        "val/pooled_ap": val_metrics["pooled_ap"],
        "val/auroc": val_metrics["auroc"],
    }
    # In-domain convenience fields (train==eval) for the per-backbone comparison table.
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

    if args.save_checkpoint is not None:
        args.save_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": {
                    f"model.classification_head.{k}": v for k, v in head.state_dict().items()
                },
                "model_type": "linear",
            },
            args.save_checkpoint,
        )
        print(f"saved checkpoint: {args.save_checkpoint}")

    wandb.finish()


if __name__ == "__main__":
    main()
