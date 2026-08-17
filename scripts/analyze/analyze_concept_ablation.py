#!/usr/bin/env python
"""Explainability follow-up Step 3 — concept-model causal faithfulness (PLAN_EXPLAINABILITY_FOLLOWUPS).

Mirrors the E8 head-ablation logic at the *concept* level for the published concept bottleneck
models: are the top-ranked concepts causally load-bearing, or just where the signal reads most
strongly? E2's rescaling caveat (gate mass and decision capacity decouple) predicts weak necessity.

For each dataset's ``cm_antonyms_*`` checkpoint (eval-mode, deterministic gates = sigmoid):
  - rank concepts by mean |per-concept logit contribution| on the TRAIN split;
  - NECESSITY: zero the top-k concepts' contributions at inference, k in KS;
  - SUFFICIENCY: keep ONLY the top-k concepts' contributions;
  - RANDOM-k baselines (20 draws) for both, to calibrate the drop;
  - metrics: Convention-A mAP (paper pairing rule) + pooled AP + AUROC on the eval split.

Ablation is exact: class_logit = masked_similarity @ W_classifier^T + b, so zeroing a concept's
column removes exactly its contribution (bias is never ablated).

Append-only: writes outputs/explain/concept_ablation/ only.

Run (local, CPU):
    uv run python scripts/analyze/analyze_concept_ablation.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from clip_cues_research.analysis.metrics import pairing_for_dataset, per_generator_map

MODELS = {
    "synthclic": (
        "data/checkpoints/cm_antonyms_synthclic.ckpt",
        "data/embeddings/synthclic_projected_embeddings.pkl",
    ),
    "cnnspot": (
        "data/checkpoints/cm_antonyms_cnnspot.ckpt",
        "data/embeddings/cnnspot_projected_embeddings.pkl",
    ),
}
KS = [1, 2, 3, 5, 10, 20, 40]
N_RANDOM = 20
SEED = 123
OUT = Path("outputs/explain/concept_ablation")


def unit_rows(M: np.ndarray) -> np.ndarray:
    return M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-12, None)


def load_cm(ckpt: str) -> dict[str, np.ndarray]:
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
    return {
        "T": unit_rows(sd["model.text_embeddings"].numpy().astype(np.float64)),
        "Wc": sd["model.W_concepts.weight"].numpy().astype(np.float64),
        "bc": sd["model.W_concepts.bias"].numpy().astype(np.float64),
        "Wclf": sd["model.W_classifier.weight"].numpy().astype(np.float64).ravel(),
        "bclf": float(sd["model.W_classifier.bias"].numpy().ravel()[0]),
    }


def contributions(m: dict, X: np.ndarray) -> np.ndarray:
    """Per-image per-concept logit contributions (eval-mode ConceptBottleneckModel forward)."""
    Z = unit_rows(X)
    sim = Z @ m["T"].T
    gates = 1.0 / (1.0 + np.exp(-(Z @ m["Wc"].T + m["bc"])))
    return sim * gates * m["Wclf"][None, :]  # (n, num_concepts); logit = sum + bclf


def metrics(meta: pd.DataFrame, scores: np.ndarray, pairing: str) -> dict:
    y = meta["label"].to_numpy().astype(int)
    return {
        "mAP": per_generator_map(meta.assign(score=scores), real_pairing=pairing),
        "pooled_ap": float(average_precision_score(y, scores)),
        "auroc": float(roc_auc_score(y, scores)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for ds, (ckpt, pkl) in MODELS.items():
        m = load_cm(ckpt)
        d = pickle.load(open(pkl, "rb"))
        df, emb = d["df"].reset_index(drop=True), np.asarray(d["embeddings"], dtype=np.float64)
        ev = "test" if (df["split"] == "test").any() else "validation"
        tr = (df["split"] == "train").to_numpy()
        te = (df["split"] == ev).to_numpy()
        pairing = pairing_for_dataset(ds)

        C_tr = contributions(m, emb[tr])
        C_te = contributions(m, emb[te])
        meta_te = df.loc[te].reset_index(drop=True)
        order = np.argsort(-np.abs(C_tr).mean(0))  # rank on TRAIN, evaluate on eval split

        vocab = torch.load("data/embeddings/antonyms_diff_embeddings.pt")["vocabulary"]
        gates_mean_active = float(
            (1.0 / (1.0 + np.exp(-(unit_rows(emb[tr]) @ m["Wc"].T + m["bc"]))) > 0.5).sum(1).mean()
        )

        def score_with(cols_zeroed: np.ndarray | None = None, cols_kept: np.ndarray | None = None):
            C = C_te.copy()
            if cols_zeroed is not None:
                C[:, cols_zeroed] = 0.0
            if cols_kept is not None:
                mask = np.ones(C.shape[1], bool)
                mask[cols_kept] = False
                C[:, mask] = 0.0
            return C.sum(1) + m["bclf"]

        base = metrics(meta_te, score_with(), pairing)
        rows = [{"mode": "baseline", "k": 0, **base}]
        rng = np.random.default_rng(SEED)
        for k in KS:
            rows.append(
                {
                    "mode": "ablate_top_k",
                    "k": k,
                    **metrics(meta_te, score_with(cols_zeroed=order[:k]), pairing),
                }
            )
            rows.append(
                {
                    "mode": "keep_only_top_k",
                    "k": k,
                    **metrics(meta_te, score_with(cols_kept=order[:k]), pairing),
                }
            )
            rand_abl, rand_keep = [], []
            for _ in range(N_RANDOM):
                cols = rng.choice(C_te.shape[1], size=k, replace=False)
                rand_abl.append(metrics(meta_te, score_with(cols_zeroed=cols), pairing)["mAP"])
                rand_keep.append(metrics(meta_te, score_with(cols_kept=cols), pairing)["mAP"])
            rows.append(
                {
                    "mode": "ablate_random_k",
                    "k": k,
                    "mAP": float(np.mean(rand_abl)),
                    "mAP_std": float(np.std(rand_abl)),
                }
            )
            rows.append(
                {
                    "mode": "keep_only_random_k",
                    "k": k,
                    "mAP": float(np.mean(rand_keep)),
                    "mAP_std": float(np.std(rand_keep)),
                }
            )

        top = [
            {
                "rank": r + 1,
                "concept": vocab[i],
                "W_classifier": float(m["Wclf"][i]),
                "mean_abs_contribution_train": float(np.abs(C_tr[:, i]).mean()),
            }
            for r, i in enumerate(order[:10])
        ]
        report[ds] = {
            "checkpoint": ckpt,
            "eval_split": ev,
            "real_pairing": pairing,
            "n_eval": int(te.sum()),
            "mean_active_gates_train": gates_mean_active,
            "baseline": base,
            "top_concepts": top,
            "ablation": rows,
        }
        tbl = pd.DataFrame(rows)
        tbl.to_csv(OUT / f"{ds}_ablation.csv", index=False)
        print(f"\n=== {ds} (eval={ev}, pairing={pairing}, baseline mAP {base['mAP']:.3f}) ===")
        print("top concepts:", [t["concept"] for t in top[:5]])
        with pd.option_context("display.width", 150):
            print(tbl.round(4).to_string(index=False))

    (OUT / "concept_ablation.json").write_text(json.dumps(report, indent=2))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
