#!/usr/bin/env python
"""E2: one concept-model training run for the W&B beta-sensitivity sweep.

Reviewer 3 (mandatory) asks how the concept-sparsity weight ``beta`` trades off detection
performance against interpretability. This script trains the concept bottleneck model on cached
SynthCLIC embeddings for a single ``beta`` and logs the two quantities the reviewer wants:

    * ``val/mAP`` (== average precision for binary detection), and
    * ``val/mean_active_concepts`` (mean #concepts with gate > threshold per image).

It runs two ways:

    # standalone (one beta, e.g. a smoke test or a single point)
    python scripts/run/run_beta_sweep.py --beta 1e-3 --epochs 5 --no-wandb

    # under a sweep (reproduction/config/sweep/beta_sensitivity.yaml drives the 5 beta values)
    wandb sweep reproduction/config/sweep/beta_sensitivity.yaml
    wandb agent <entity>/clip-cues/<sweep_id>

The sweep agent passes ``--beta=...`` etc. on the command line; argparse picks them up and they
are mirrored into ``wandb.config`` for logging.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from clip_cues.concept_modeling.train import prepare_datasets, train_concept_model
from clip_cues_research.analysis.concept_activation import mean_active_concepts
from clip_cues_research.analysis.metrics import detection_metrics, pairing_for_dataset
from clip_cues_research.results import make_run_id

# dataset name -> cached image-embedding filename.
# The concept model operates in the shared CLIP space, so it needs the *projected* (768-d)
# image embeddings (image_embeds), not the 1024-d vision pooler_output.
IMAGE_EMBEDDINGS = {"synthclic": "synthclic_projected_embeddings.pkl"}
# vocabulary name -> cached text-embedding filename. ``antonyms`` is the paper's vocabulary: 168
# antonym *difference-directions* (normalize(pos) - normalize(neg), l2-normalized). The 336-pole
# variant (each pole a separate concept) is a regression and kept only for comparison.
TEXT_EMBEDDINGS = {
    "antonyms": "antonyms_diff_embeddings.pt",
    "antonyms_poles": "antonyms_embeddings.pt",
}


def _split_meta(image_path: Path, ds_names: list[str], splits: list[str]) -> pd.DataFrame:
    """Ordered (image_id, label, source) for a split — same filter/order as ``prepare_datasets``.

    Source is attached to predictions **by position** (not by image_id) because SynthCLIC's
    ``image_id`` is shared by the real + every generator; a map/merge on it collapses per-generator
    mAP to pooled AP.
    """
    with open(image_path, "rb") as f:
        cache = pickle.load(f)
    df = cache["df"]
    idx = df["ds_name"].isin(ds_names) & df["split"].isin(splits)
    return df.loc[idx, ["image_id", "label", "source"]].reset_index(drop=True)


@torch.no_grad()
def _predict(model, loader, device) -> pd.DataFrame:
    """Per-image predictions: frame with ``image_id``, ``label`` (0/1), ``score`` (P(synthetic))."""
    model.eval()
    ids: list[str] = []
    ys: list[int] = []
    scores: list[float] = []
    for emb, labels, image_ids in loader:
        out = model(emb.to(device))
        scores.extend(torch.sigmoid(out["class_logits"].view(-1)).cpu().numpy().tolist())
        ys.extend(labels.view(-1).cpu().numpy().astype(int).tolist())
        ids.extend([str(i) for i in image_ids])
    return pd.DataFrame({"image_id": ids, "label": ys, "score": scores})


def _conv_a_metrics(model, loader, device, meta, dataset_label):
    """Convention-A metrics (per-generator mean AP) + pooled AP/AUROC for one split.

    ``meta`` is the ordered (image_id, label, source) frame for this split; ``source`` is attached
    by position (loaders are unshuffled), with an image_id order assertion as a safety check.
    """
    pred = _predict(model, loader, device).reset_index(drop=True)
    meta = meta.reset_index(drop=True)
    if (
        len(pred) != len(meta)
        or not (pred["image_id"].astype(str).values == meta["image_id"].astype(str).values).all()
    ):
        raise AssertionError(
            "prediction/meta row order mismatch — cannot attach source by position"
        )
    pred["source"] = meta["source"].values
    pairing = pairing_for_dataset(dataset_label)
    bundle = detection_metrics(pred, real_pairing=pairing)
    y, s = pred["label"].to_numpy(), pred["score"].to_numpy()
    auroc = float(roc_auc_score(y, s)) if 0 < int(y.sum()) < len(y) else float("nan")
    return {
        "mAP": bundle["mAP"],
        "pooled_ap": bundle["pooled_ap"],
        "auroc": auroc,
        "real_pairing": pairing,
        "n_generators": bundle["n_generators"],
    }, pred


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # swept / fixed hyperparameters (names match reproduction/config/sweep/beta_sensitivity.yaml)
    p.add_argument("--beta", type=float, default=1e-4)
    p.add_argument("--alpha", type=float, default=1e-4)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)  # concept cfg: 1e-4 (W_classifier)
    p.add_argument("--label-smoothing", type=float, default=0.0)  # concept cfg: 0.0 (NOT 0.1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--epochs", type=int, default=4000
    )  # concept cfg cap; early stopping ends it sooner
    p.add_argument("--early-stopping-patience", type=int, default=10)  # concept cfg: 10
    p.add_argument("--check-val-every-n-epoch", type=int, default=40)  # concept cfg: check every 40
    p.add_argument(
        "--selection", default="val_loss", choices=["val_loss", "composite", "auroc"]
    )  # concept cfg early-stop = val/loss
    p.add_argument("--seed", type=int, default=123)  # paper: 123
    # data
    p.add_argument("--dataset", default="synthclic", choices=list(IMAGE_EMBEDDINGS))
    p.add_argument("--vocabulary", default="antonyms", choices=list(TEXT_EMBEDDINGS))
    p.add_argument("--embeddings-dir", type=Path, default=Path("data/embeddings"))
    p.add_argument("--train-splits", nargs="+", default=["train"])
    p.add_argument("--val-splits", nargs="+", default=["validation"])
    p.add_argument("--test-splits", nargs="+", default=["test"])
    p.add_argument("--active-threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    # W&B
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "clip-cues"))
    p.add_argument("--no-wandb", action="store_true", help="disable W&B (local smoke test)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_id = make_run_id()

    import wandb

    wandb.init(
        project=args.wandb_project,
        group="e2_beta_sweep",
        name=f"e2_beta_sweep_{run_id}",
        config=vars(args),
        mode="disabled" if args.no_wandb else None,
    )
    cfg = wandb.config  # under a sweep, the agent's overrides land here

    image_path = args.embeddings_dir / IMAGE_EMBEDDINGS[cfg["dataset"]]
    text_path = args.embeddings_dir / TEXT_EMBEDDINGS[cfg["vocabulary"]]

    # Checkpoint dir for the selected model (min composite (1-sparsity_rel)+(1-auroc), not final epoch).
    ckpt_dir = Path("results") / "e2_beta_sweep" / "checkpoints" / run_id / f"beta_{cfg['beta']:g}"

    result = train_concept_model(
        image_embeddings_path=image_path,
        text_embeddings_path=text_path,
        ds_names=[cfg["dataset"]],
        train_splits=args.train_splits,
        val_splits=args.val_splits,
        tau=cfg["tau"],
        beta=cfg["beta"],
        alpha=cfg["alpha"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        label_smoothing=cfg["label_smoothing"],
        batch_size=cfg["batch_size"],
        epochs=cfg["epochs"],
        early_stopping_patience=cfg["early_stopping_patience"],
        check_val_every_n_epoch=cfg["check_val_every_n_epoch"],
        selection=cfg["selection"],
        device=args.device,
        seed=cfg["seed"],
        output_dir=ckpt_dir,
        epoch_callback=None if args.no_wandb else _wandb_epoch_logger,
    )

    model = result["model"]
    device = result["device"]
    val_loader = result["val_loader"]

    # `train_concept_model` already restored the best (min composite) checkpoint into `model`;
    # reload from disk only as a safety net if the in-memory restore was skipped.
    ckpt_path = ckpt_dir / "best_model.pt"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])

    # Ordered (image_id,label,source) per split — for positional source attach (image_id non-unique).
    val_meta = _split_meta(image_path, [cfg["dataset"]], args.val_splits)
    test_meta = _split_meta(image_path, [cfg["dataset"]], args.test_splits)

    # Convention-A metrics (per-generator mean AP, the paper's metric) on val + active-concept stats.
    val, _ = _conv_a_metrics(model, val_loader, device, val_meta, cfg["dataset"])
    val_active = mean_active_concepts(
        model, val_loader, device=device, threshold=args.active_threshold
    )

    # Test split (built from the same cached embeddings) for the final table.
    _, test_dataset = prepare_datasets(
        image_path, [cfg["dataset"]], args.train_splits, args.test_splits
    )
    from torch.utils.data import DataLoader

    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False)
    test, test_preds = _conv_a_metrics(model, test_loader, device, test_meta, cfg["dataset"])
    test_active = mean_active_concepts(
        model, test_loader, device=device, threshold=args.active_threshold
    )

    summary = {
        "beta": cfg["beta"],
        "alpha": cfg["alpha"],
        # mAP = Convention A (per-generator mean AP); pooled_ap kept for transparency.
        "val/mAP": val["mAP"],
        "val/pooled_ap": val["pooled_ap"],
        "val/auroc": val["auroc"],
        "val/mean_active_concepts": val_active["mean_active_concepts"],
        "val/mean_gate_mass": val_active["mean_gate_mass"],
        "val/best_auroc": result["best_val_auroc"],
        "test/mAP": test["mAP"],
        "test/pooled_ap": test["pooled_ap"],
        "test/auroc": test["auroc"],
        "test/mean_active_concepts": test_active["mean_active_concepts"],
        "test/mean_gate_mass": test_active["mean_gate_mass"],
        # max |W_classifier| — at high beta the gates collapse but the classifier rescales (concentrates
        # large weights on the few surviving concepts), which is why mAP holds at ~0 active concepts.
        "test/max_w_classifier": float(model.W_classifier.weight.abs().max().item()),
        "seed": cfg["seed"],  # for multi-seed aggregation (mean +/- std over seeds)
        "real_pairing": test["real_pairing"],
        "n_generators": test["n_generators"],
        "num_concepts": val_active["num_concepts"],
        "vocabulary": cfg["vocabulary"],
    }
    wandb.log(summary)
    wandb.summary.update(summary)

    # Always persist raw results locally: results/e2_beta_sweep/<dataset>__beta_<beta>/metrics.json
    from clip_cues_research.results import save_run_results

    run_dir = save_run_results(
        "e2_beta_sweep", f"{cfg['dataset']}__beta_{cfg['beta']:g}", summary, run_id=run_id
    )
    # Per-image test predictions (label/score/source) for downstream re-scoring / auditing.
    preds_dir = Path("results") / "e2_beta_sweep" / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)
    test_preds.assign(beta=cfg["beta"], vocabulary=cfg["vocabulary"]).to_parquet(
        preds_dir / f"{cfg['dataset']}__beta_{cfg['beta']:g}__{run_id}.parquet"
    )
    print(f"Results saved to {run_dir}/")

    print("\n=== E2 run summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    wandb.finish()


def _wandb_epoch_logger(epoch, train_loss, train_metrics, val_loss, val_metrics):
    import wandb

    wandb.log(
        {
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "train/auroc": float(train_metrics["auroc"]),
            "val/auroc": float(val_metrics["auroc"]),
            "val/ap": float(val_metrics["ap"]),
        }
    )


if __name__ == "__main__":
    main()
