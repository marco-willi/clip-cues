#!/usr/bin/env python
"""E5: activation- vs weight-orthogonality ablation (small empirical check).

Reviewer 3 asks why the L1 "distilled representation" layer is regularised via a *batch-dependent
activation* alignment loss rather than penalising the weights directly with ``||I - W_L1^T W_L1||``,
and what the trade-off is. The answer is mainly mathematical (paper), but this script provides the
small empirical check: it trains the SAME single-hidden-layer head (input -> Linear(in, K) -> logit)
on cached SynthCLIC embeddings under three regimes that differ ONLY in the auxiliary loss —

    none              : no orthogonality penalty (baseline)
    activation_ortho  : the published loss (Gram of L2-normalised activations -> ||I - G||)
    weight_ortho      : explicit weight penalty ||I - W_L1 W_L1^T||  (what the reviewer names)

and reports, for each: detection mAP/AUROC plus two orthogonality measures on the held-out split —
the weight-space off-diagonal Gram mass and the activation-space off-diagonal Gram mass (both
lower = more orthogonal). This lets us state numerically whether the activation loss already yields
weights that are "roughly orthogonal" (the paper's claim) at comparable detection performance.

Embeddings-only, CPU-friendly, runs in minutes:
    python scripts/run/run_orthogonality_ablation.py \
        --embeddings data/embeddings/synthclic_embeddings.pkl --epochs 200
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader

from clip_cues.classification_head import ActivationOrthogonalityHead, ClassificationHead
from clip_cues.concept_modeling.dataset import CLIPFeatureDataset
from clip_cues_research.analysis.metrics import detection_metrics, pairing_for_dataset
from clip_cues_research.analysis.orthogonality import orthogonality_score
from clip_cues_research.results import make_run_id, save_run_results

VARIANTS = ["none", "activation_ortho", "weight_ortho"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--embeddings", type=Path, default=Path("data/embeddings/synthclic_l14_local.pkl")
    )
    p.add_argument(
        "--dataset",
        default="synthclic",
        help="Dataset label — selects the per-generator real-pairing rule for Convention-A mAP",
    )
    p.add_argument(
        "--latent-dim", type=int, default=8, help="K: width of the L1 layer (paper uses 8)"
    )
    p.add_argument(
        "--ortho-weight", type=float, default=0.33, help="lambda for both ortho penalties"
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)  # paper: Adam coupled L2, wd=0.01
    p.add_argument("--label-smoothing", type=float, default=0.1)  # paper: 0.1
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--early-stopping-patience", type=int, default=5)  # paper: 5 (monitor val CE)
    p.add_argument("--seed", type=int, default=123)  # paper: 123
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--train-splits", nargs="+", default=["train"])
    p.add_argument("--val-splits", nargs="+", default=["validation"])
    p.add_argument("--test-splits", nargs="+", default=["test"])
    p.add_argument("--wandb", action="store_true", help="also log a summary to W&B")
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "clip-cues"))
    return p.parse_args()


def load_feature_dataset(
    pkl_path: Path, splits: list[str]
) -> tuple[CLIPFeatureDataset, int, pd.DataFrame]:
    with open(pkl_path, "rb") as f:
        cache = pickle.load(f)
    df, emb = cache["df"], cache["embeddings"]
    idx = df["split"].isin(splits).values
    ds = CLIPFeatureDataset(
        torch.from_numpy(emb[idx, :]),
        torch.from_numpy(df.loc[idx, "label"].values).to(torch.float32),
        df.loc[idx, "image_id"].values,
    )
    meta = df.loc[idx, ["image_id", "label", "source"]].reset_index(drop=True)
    return ds, emb.shape[1], meta


def build_head(variant: str, input_dim: int, k: int, lam: float):
    """Same architecture for every variant (input -> Linear(in,k) -> logit); only the aux loss differs."""
    if variant == "activation_ortho":
        return ActivationOrthogonalityHead(
            input_dim, layer_dims=[k], non_linear=False, loss_weight_ortho=lam
        )
    # `none` and `weight_ortho` share ClassificationHead; `none` just zeroes the weight penalty.
    return ClassificationHead(
        input_dim,
        layer_dims=[k],
        non_linear=False,
        loss_weight_ortho=(lam if variant == "weight_ortho" else 0.0),
    )


def l1_weight(head) -> torch.Tensor:
    """The W_L1 layer weight (k, input_dim) — same location for both head types."""
    return head.layers[-1].weight


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


def score_predictions(
    pred_df: pd.DataFrame, meta: pd.DataFrame, dataset_label: str
) -> tuple[dict[str, float], pd.DataFrame]:
    """Convention-A metrics for one split (mAP) plus pooled AP/AUROC; returns (metrics, preds+source).

    ``image_id`` is non-unique in SynthCLIC (real + every generator share it), so ``source`` is
    attached **by position** (unshuffled loader ⇒ same order as ``meta``); a merge would explode the
    frame and collapse per-generator mAP to pooled AP.
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
    return {
        "mAP": bundle["mAP"],
        "pooled_ap": bundle["pooled_ap"],
        "auroc": auroc,
        "real_pairing": pairing,
        "n_generators": bundle["n_generators"],
    }, merged


@torch.no_grad()
def activation_orthogonality(head, loader, device) -> float:
    """Off-diagonal Gram mass of the L2-normalised L1 activations on the given split (lower=orthogonal)."""
    head.eval()
    acts = []
    for emb, _, _ in loader:
        out = head(emb.to(device), output_distilled_representations=True)
        acts.append(out["distilled_representations"])
    a = F.normalize(torch.cat(acts), dim=0)
    gram = a.T @ a
    off = gram - torch.diag(torch.diag(gram))
    return float(torch.linalg.norm(off, ord="fro"))


def _val_cross_entropy(pred_df, label_smoothing: float) -> float:
    """Mean label-smoothed BCE from a predictions frame (the paper's early-stopping metric)."""
    import numpy as np

    p = pred_df["score"].to_numpy().clip(1e-7, 1 - 1e-7)
    y = pred_df["label"].to_numpy().astype(float)
    ys = y * (1 - label_smoothing) + (1 - y) * label_smoothing
    return float(-(ys * np.log(p) + (1 - ys) * np.log(1 - p)).mean())


def train_variant(variant, train_loader, val_loader, val_meta, input_dim, args, device) -> dict:
    torch.manual_seed(args.seed)  # identical init across variants for a fair comparison
    head = build_head(variant, input_dim, args.latent_dim, args.ortho_weight).to(device)
    # Paper: torch.optim.Adam (coupled L2) with weight_decay; lr default 1e-3.
    opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ls = args.label_smoothing

    # Model selection: min val cross-entropy, early stopping patience 5 (paper).
    best_ce, best_state, since_improve = float("inf"), None, 0
    for _ in range(args.epochs):
        head.train()
        for emb, labels, _ in train_loader:
            opt.zero_grad()
            logits = head(emb.to(device))["logits"].view(-1)
            y = labels.to(device).view(-1)
            y_smooth = y * (1 - ls) + (1 - y) * ls
            loss = F.binary_cross_entropy_with_logits(logits, y_smooth)
            loss = loss + head.compute_loss().get("orthogonality", torch.zeros((), device=device))
            loss.backward()
            opt.step()
        val_ce = _val_cross_entropy(predict(head, val_loader, device), ls)
        if val_ce < best_ce:
            best_ce, since_improve = val_ce, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            since_improve += 1
            if since_improve >= args.early_stopping_patience:
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    val_metrics, _ = score_predictions(predict(head, val_loader, device), val_meta, args.dataset)
    return {"head": head, "val": val_metrics}


def main() -> None:
    args = parse_args()
    run_id = make_run_id()
    device = torch.device(args.device)

    train_ds, input_dim, _ = load_feature_dataset(args.embeddings, args.train_splits)
    val_ds, _, val_meta = load_feature_dataset(args.embeddings, args.val_splits)
    test_ds, _, test_meta = load_feature_dataset(args.embeddings, args.test_splits)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    pairing = pairing_for_dataset(args.dataset)
    print(
        f"input_dim={input_dim} K={args.latent_dim} lambda={args.ortho_weight} "
        f"| dataset={args.dataset} pairing={pairing} "
        f"| train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} | run_id={run_id}\n"
    )

    preds_dir = Path("results") / "e5_orthogonality" / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for variant in VARIANTS:
        out = train_variant(variant, train_loader, val_loader, val_meta, input_dim, args, device)
        head = out["head"]
        test, test_preds = score_predictions(
            predict(head, test_loader, device), test_meta, args.dataset
        )
        row = {
            "variant": variant,
            "val/mAP": out["val"]["mAP"],
            "val/pooled_ap": out["val"]["pooled_ap"],
            "val/auroc": out["val"]["auroc"],
            "test/mAP": test["mAP"],  # Convention A: per-generator mean AP (the paper's metric)
            "test/pooled_ap": test["pooled_ap"],  # old SimpleMetrics quantity, for transparency
            "test/auroc": test["auroc"],
            "real_pairing": test["real_pairing"],
            "n_generators": test["n_generators"],
            # weight-space orthogonality of W_L1 (off-diagonal Gram mass), lower = more orthogonal
            "weight_ortho_score": orthogonality_score(l1_weight(head)),
            # activation-space orthogonality on the test split, lower = more orthogonal
            "activation_ortho_score": activation_orthogonality(head, test_loader, device),
        }
        rows[variant] = row
        test_preds.assign(variant=variant).to_parquet(preds_dir / f"{variant}__{run_id}.parquet")
        save_run_results(
            "e5_orthogonality",
            variant,
            row,
            arrays={
                "score": test_preds["score"].to_numpy(),
                "label": test_preds["label"].to_numpy(),
                "source": test_preds["source"].to_numpy().astype(str),
            },
            run_id=run_id,
        )
        print(
            f"[{variant:16s}] test mAP={row['test/mAP']:.4f} (pooled_ap={row['test/pooled_ap']:.4f}) "
            f"auroc={row['test/auroc']:.4f} "
            f"| W-ortho={row['weight_ortho_score']:.4f} act-ortho={row['activation_ortho_score']:.4f}"
        )

    # ── comparison table ──
    print(
        "\n=== E5 orthogonality ablation (K={}, lambda={}) ===".format(
            args.latent_dim, args.ortho_weight
        )
    )
    hdr = f"{'variant':16s} {'test_mAP':>9s} {'test_auroc':>11s} {'W_ortho(v)':>11s} {'act_ortho(v)':>13s}"
    print(hdr)
    print("-" * len(hdr))
    for v in VARIANTS:
        r = rows[v]
        print(
            f"{v:16s} {r['test/mAP']:9.4f} {r['test/auroc']:11.4f} "
            f"{r['weight_ortho_score']:11.4f} {r['activation_ortho_score']:13.4f}"
        )
    print("\n(W_ortho / act_ortho: off-diagonal Gram mass, lower = more orthogonal)")

    if args.wandb:
        import wandb

        wb = wandb.init(
            project=args.wandb_project,
            group="e5_orthogonality",
            name=f"e5_orthogonality_{run_id}",
            config=vars(args) | {"run_id": run_id},
        )
        for v in VARIANTS:
            for key, val in rows[v].items():
                if isinstance(val, (int, float)):
                    wb.summary[f"{v}/{key}"] = val
        wb.finish()


if __name__ == "__main__":
    main()
