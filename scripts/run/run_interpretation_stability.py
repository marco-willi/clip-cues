#!/usr/bin/env python
"""E8: multi-seed interpretation-stability harness for both interpretable heads.

Reviewer 1 asks for more generalization analysis; E8 measures whether the *interpretations* (not just
detection mAP) are stable. This script **imports** the existing train engines and drives its own seed
loop + artifact persistence — it does NOT edit ``run_orthogonality_ablation.py`` /
``run_beta_sweep.py`` or write into their E2/E5 result namespaces (see PLAN_E8 isolation constraints).

Per the 2026-06-26 scope decision:
  * orthogonal head — axes: seed (init x data-shuffle), lambda, k, backbone. The init/data split is the
    direct test of the paper's "directions are init-driven" hypothesis: ``--regime vary-init`` fixes the
    data order and varies init; ``--regime vary-shuffle`` fixes init and varies the data order.
  * concept model — axes: seed and beta only.
Both run on SynthCLIC and CNNSpot (pass the matching embeddings + ``--dataset``).

Artifacts land under ``results/e8_interpretability_stability/{ortho,concept}/<run_id>/``:
  * ``fit_<i>.npz`` — per-fit interpretation artifacts (W_L1 + importance, or W_classifier);
  * ``fits.csv`` — per-fit detection metrics (sanity that training wasn't broken);
  * ``stability.json`` — the aggregated E8 metrics from ``analysis.interpretation_stability``.

Embeddings-only, CPU-friendly. Examples:
    # orthogonal head, vary init (the init-hypothesis test), SynthCLIC L/14
    python scripts/run/run_interpretation_stability.py --mode ortho \
        --embeddings data/embeddings/synthclic_l14_local.pkl --dataset synthclic \
        --regime vary-init --seeds 0 1 2 3 4 5 6 7 8 9

    # concept model, seed stability at fixed beta, CNNSpot
    python scripts/run/run_interpretation_stability.py --mode concept \
        --image-embeddings data/embeddings/cnnspot_projected_embeddings.pkl \
        --text-embeddings data/embeddings/antonyms_diff_embeddings.pt \
        --dataset cnnspot --beta 1e-4 --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from clip_cues_research.analysis.interpretation_stability import (
    direction_stability,
    importance_rank_correlation,
    sign_agreement,
    top_k_jaccard,
)
from clip_cues_research.results import make_run_id

_SCRIPTS = Path(__file__).resolve().parent


def _load_script_module(name: str):
    """Import a sibling script as a module *without* modifying it (honors the isolation constraint)."""
    spec = importlib.util.spec_from_file_location(f"_e8_{name}", _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── orthogonal head ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _ortho_importance(head, loader, device) -> np.ndarray:
    """Signed per-direction importance = mean logit contribution(synthetic) - mean(real).

    Mirrors the paper's "difference in logit contribution" (Fig. in §5.1): contribution
    ``c_i = a_i * w_logit_i`` per direction; the class-mean difference is how strongly (and toward which
    class) each direction drives the decision. |value| is the interpretable strength ranking.
    """
    head.eval()
    w_logit = head.to_logits.weight.view(-1).to(device)  # (k,)
    sums = {0: None, 1: None}
    counts = {0: 0, 1: 0}
    for emb, labels, _ in loader:
        acts = head(emb.to(device), output_distilled_representations=True)[
            "distilled_representations"
        ]
        contrib = acts * w_logit  # (batch, k)
        y = labels.view(-1).to(device)
        for cls in (0, 1):
            m = y == cls
            if m.any():
                s = contrib[m].sum(0)
                sums[cls] = s if sums[cls] is None else sums[cls] + s
                counts[cls] += int(m.sum())
    mean1 = (sums[1] / counts[1]) if counts[1] else torch.zeros_like(w_logit)
    mean0 = (sums[0] / counts[0]) if counts[0] else torch.zeros_like(w_logit)
    return (mean1 - mean0).cpu().numpy()


def fit_orthogonal(ortho_mod, args, init_seed: int, shuffle_seed: int, device) -> dict:
    """One orthogonal-head fit with separable init/shuffle seeds. Returns artifacts + val metrics.

    init_seed drives ``torch.manual_seed`` inside ``train_variant`` (weight init); shuffle_seed drives an
    explicit DataLoader generator (data order) so the two sources of randomness are independent.
    """
    train_ds, input_dim, _ = ortho_mod.load_feature_dataset(args.embeddings, args.train_splits)
    val_ds, _, val_meta = ortho_mod.load_feature_dataset(args.embeddings, args.val_splits)
    gen = torch.Generator().manual_seed(shuffle_seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # args namespace for train_variant (init_seed -> args.seed reseeds global RNG before head build)
    targs = SimpleNamespace(
        seed=init_seed,
        latent_dim=args.latent_dim,
        ortho_weight=args.ortho_weight,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        dataset=args.dataset,
    )
    out = ortho_mod.train_variant(
        "activation_ortho", train_loader, val_loader, val_meta, input_dim, targs, device
    )
    head = out["head"]
    return {
        "W_L1": ortho_mod.l1_weight(head).detach().cpu().numpy(),  # (k, d)
        "importance": _ortho_importance(head, val_loader, device),  # (k,)
        "metrics": {"init_seed": init_seed, "shuffle_seed": shuffle_seed, **out["val"]},
    }


def run_ortho(args) -> dict:
    ortho_mod = _load_script_module("run_orthogonality_ablation")
    device = torch.device(args.device)
    out_dir = Path("results") / "e8_interpretability_stability" / "ortho" / make_run_id()
    out_dir.mkdir(parents=True, exist_ok=True)

    fits, importances, w_l1s = [], [], []
    for i, seed in enumerate(args.seeds):
        # vary-init: data order fixed (shuffle_seed const), init varies with the seed.
        # vary-shuffle: init fixed (--base-seed), data order varies with the seed.
        if args.regime == "vary-init":
            init_seed, shuffle_seed = seed, args.base_seed
        else:
            init_seed, shuffle_seed = args.base_seed, seed
        res = fit_orthogonal(ortho_mod, args, init_seed, shuffle_seed, device)
        np.savez(out_dir / f"fit_{i}.npz", W_L1=res["W_L1"], importance=res["importance"])
        fits.append(res["metrics"])
        importances.append(res["importance"])
        w_l1s.append(res["W_L1"])
        print(
            f"[ortho] fit {i}: init={init_seed} shuffle={shuffle_seed} val/mAP={res['metrics']['mAP']:.4f}"
        )

    stability = {
        "directions": direction_stability(w_l1s),
        "importance_top_k_jaccard": top_k_jaccard(importances, k=args.top_k),
        "importance_rank_correlation": importance_rank_correlation(importances),
        "sign_agreement_per_direction": sign_agreement(importances).tolist(),
    }
    return _finalize(out_dir, args, fits, stability)


# ── concept model ──────────────────────────────────────────────────────────────────────────────
def fit_concept(beta_mod, args, seed: int, beta: float, device: str) -> dict:
    """One concept-model fit (train_concept_model) → W_classifier + val metrics + #active concepts."""
    train_concept_model = beta_mod.train_concept_model
    mean_active_concepts = beta_mod.mean_active_concepts

    result = train_concept_model(
        image_embeddings_path=args.image_embeddings,
        text_embeddings_path=args.text_embeddings,
        ds_names=[args.dataset],
        train_splits=args.train_splits,
        val_splits=args.val_splits,
        beta=beta,
        alpha=args.alpha,
        tau=args.tau,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        selection="val_loss",
        device=device,
        seed=seed,
        output_dir=None,
        verbose=False,
    )
    model, dev, val_loader = result["model"], result["device"], result["val_loader"]
    val_meta = beta_mod._split_meta(args.image_embeddings, [args.dataset], args.val_splits)
    val, _ = beta_mod._conv_a_metrics(model, val_loader, dev, val_meta, args.dataset)
    active = mean_active_concepts(model, val_loader, device=dev, threshold=args.active_threshold)
    return {
        "W_classifier": model.W_classifier.weight.detach().cpu().view(-1).numpy(),  # (n_concepts,)
        "metrics": {
            "seed": seed,
            "beta": beta,
            "mAP": val["mAP"],
            "auroc": val["auroc"],
            "mean_active_concepts": active["mean_active_concepts"],
        },
    }


def run_concept(args) -> dict:
    beta_mod = _load_script_module("run_beta_sweep")
    out_dir = Path("results") / "e8_interpretability_stability" / "concept" / make_run_id()
    out_dir.mkdir(parents=True, exist_ok=True)

    betas = args.betas if args.betas else [args.beta]
    fits, importances = [], []
    i = 0
    # Stability is measured across seeds *within* each beta (identity stability of the selected concepts).
    per_beta_stability = {}
    for beta in betas:
        beta_importances = []
        for seed in args.seeds:
            res = fit_concept(beta_mod, args, seed, beta, args.device)
            np.savez(out_dir / f"fit_{i}.npz", W_classifier=res["W_classifier"])
            fits.append(res["metrics"])
            importances.append(res["W_classifier"])
            beta_importances.append(res["W_classifier"])
            print(
                f"[concept] fit {i}: beta={beta:g} seed={seed} val/mAP={res['metrics']['mAP']:.4f} "
                f"active={res['metrics']['mean_active_concepts']:.1f}"
            )
            i += 1
        if len(beta_importances) >= 2:
            per_beta_stability[f"beta_{beta:g}"] = {
                "top_k_jaccard": top_k_jaccard(beta_importances, k=args.top_k),
                "rank_correlation": importance_rank_correlation(beta_importances),
            }

    stability = {"per_beta_seed_stability": per_beta_stability}
    if len(importances) >= 2:
        stability["across_all_fits"] = {
            "top_k_jaccard": top_k_jaccard(importances, k=args.top_k),
            "rank_correlation": importance_rank_correlation(importances),
            "sign_agreement_per_concept": sign_agreement(importances).tolist(),
        }
    return _finalize(out_dir, args, fits, stability)


# ── shared ──────────────────────────────────────────────────────────────────────────────────────
def _finalize(out_dir: Path, args, fits: list[dict], stability: dict) -> dict:
    pd.DataFrame(fits).to_csv(out_dir / "fits.csv", index=False)
    payload = {
        "mode": args.mode,
        "dataset": args.dataset,
        "regime": args.regime if args.mode == "ortho" else None,
        "seeds": args.seeds,
        "n_fits": len(fits),
        "stability": stability,
    }
    (out_dir / "stability.json").write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved to {out_dir}/")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", required=True, choices=["ortho", "concept"])
    p.add_argument("--dataset", default="synthclic")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument(
        "--top-k", type=int, default=8, help="K for top-K selection-overlap (concept: e.g. 30)"
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # shared optimization
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--early-stopping-patience", type=int, default=5)
    p.add_argument("--train-splits", nargs="+", default=["train"])
    p.add_argument("--val-splits", nargs="+", default=["validation"])
    # orthogonal head
    p.add_argument("--embeddings", type=Path, help="ortho: CLIP hidden-state (1024-d) pkl")
    p.add_argument("--regime", default="vary-init", choices=["vary-init", "vary-shuffle"])
    p.add_argument(
        "--base-seed", type=int, default=123, help="fixed seed for the non-varied factor"
    )
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--ortho-weight", type=float, default=0.33)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    # concept model
    p.add_argument("--image-embeddings", type=Path, help="concept: projected (p-d) image pkl")
    p.add_argument("--text-embeddings", type=Path, help="concept: vocabulary text-embedding .pt")
    p.add_argument("--beta", type=float, default=1e-4)
    p.add_argument(
        "--betas",
        type=float,
        nargs="*",
        help="concept: sweep these betas (seed stability per beta)",
    )
    p.add_argument("--alpha", type=float, default=1e-4)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--check-val-every-n-epoch", type=int, default=40)
    p.add_argument("--active-threshold", type=float, default=0.5)
    args = p.parse_args()
    # concept defaults differ from ortho (match run_beta_sweep): override only if user left ortho defaults
    if args.mode == "concept":
        if args.weight_decay == 0.01:
            args.weight_decay = 1e-4
        if args.label_smoothing == 0.1:
            args.label_smoothing = 0.0
        if args.batch_size == 64:
            args.batch_size = 256
        if args.epochs == 200:
            args.epochs = 4000
        if args.early_stopping_patience == 5:
            args.early_stopping_patience = 10
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "ortho":
        if args.embeddings is None:
            raise SystemExit("--embeddings is required for --mode ortho")
        run_ortho(args)
    else:
        if args.image_embeddings is None or args.text_embeddings is None:
            raise SystemExit(
                "--image-embeddings and --text-embeddings are required for --mode concept"
            )
        run_concept(args)


if __name__ == "__main__":
    main()
