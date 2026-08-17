#!/usr/bin/env python
"""Explainability follow-up Step 1 — geometry of exposure (PLAN_EXPLAINABILITY_FOLLOWUPS).

Explains the exposure/intervention result (combined probe: CF-Eval GAN 0.357 -> 0.944) at the
level of the learned directions:

  1. Pairwise cosines between the checkpoint linear-probe directions
     (synthclic / synthbuster / cnnspot / combined) in the 1024-d pooler space.
  2. Least-squares decomposition of w_combined onto span{w_synthclic, w_synthbuster, w_cnnspot}
     (coefficients + captured variance).
  3. Span-sufficiency on CF-Eval: fit a tiny logistic on probe-score features (x . w_hat_i) using
     ONLY the combined training data (the same data the combined probe saw), then score CF-Eval
     zero-shot. If span{w_synthclic, w_cnnspot} recovers the combined probe's per-architecture
     APs, combined training *composes* existing cue directions rather than discovering new ones.
     A full-rank (1024-d) logistic on the same data is the retrained upper bound.

Append-only: reads checkpoints/embeddings, writes outputs/explain/probe_geometry/ only.

Run (local, CPU):
    uv run python scripts/analyze/analyze_probe_geometry.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from clip_cues_research.analysis.metrics import (
    map_by_architecture,
    pairing_for_dataset,
    per_generator_map,
)

CKPTS = {
    "synthclic": "data/checkpoints/linear_probe_synthclic.ckpt",
    "synthbuster": "data/checkpoints/linear_probe_synthbuster.ckpt",
    "cnnspot": "data/checkpoints/linear_probe_cnnspot.ckpt",
    "combined": "data/checkpoints/linear_probe_combined.ckpt",
}
POOLER = {
    "synthclic": "data/embeddings/synthclic_clip_large_patch14.pkl",
    "synthbuster": "data/embeddings/synthbuster-plus_clip_large_patch14.pkl",
    "cnnspot": "data/embeddings/cnnspot_clip_large_patch14.pkl",
}
CF_EMB = "data/embeddings/communityforensics_l14_eval.pkl"
OUT = Path("outputs/explain/probe_geometry")
SEED = 123


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v), 1e-12, None)


def probe_direction(ckpt: str) -> np.ndarray:
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
    return sd["model.classification_head.fc.weight"].numpy().astype(np.float64).ravel()


def load_pkl(path: str) -> tuple[pd.DataFrame, np.ndarray]:
    d = pickle.load(open(path, "rb"))
    return d["df"].reset_index(drop=True), np.asarray(d["embeddings"], dtype=np.float64)


def split(df: pd.DataFrame, emb: np.ndarray, sp: str) -> tuple[np.ndarray, pd.DataFrame]:
    m = (df["split"] == sp).to_numpy()
    return emb[m], df.loc[m].reset_index(drop=True)


def eval_scores(name: str, meta: pd.DataFrame, scores: np.ndarray, pairing: str) -> dict:
    y = meta["label"].to_numpy().astype(int)
    frame = meta.assign(score=scores)
    row = {
        "eval": name,
        "auroc": float(roc_auc_score(y, scores)),
        "pooled_ap": float(average_precision_score(y, scores)),
        "mAP_by_gen": per_generator_map(frame, real_pairing=pairing),
    }
    if "architecture" in meta.columns:
        arch = map_by_architecture(frame, real_pairing=pairing)
        row.update({f"{a}_mAP": float(v) for a, v in zip(arch["architecture"], arch["mAP"])})
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    w = {k: probe_direction(p) for k, p in CKPTS.items()}
    names = list(w)

    # 1) pairwise cosines
    cos = pd.DataFrame(
        [[float(unit(w[a]) @ unit(w[b])) for b in names] for a in names],
        index=names,
        columns=names,
    )
    cos.to_csv(OUT / "probe_direction_cosines.csv")
    print("pairwise direction cosines:\n", cos.round(3))

    # 2) decompose w_combined on the single-domain directions
    B = np.stack([w["synthclic"], w["synthbuster"], w["cnnspot"]], axis=1)  # 1024 x 3
    coef, *_ = np.linalg.lstsq(B, w["combined"], rcond=None)
    resid = w["combined"] - B @ coef
    decomp = {
        "coefficients": dict(zip(["synthclic", "synthbuster", "cnnspot"], coef.round(4).tolist())),
        "captured_norm_fraction": float(np.linalg.norm(B @ coef) / np.linalg.norm(w["combined"])),
        "captured_variance_r2": float(
            1 - np.linalg.norm(resid) ** 2 / np.linalg.norm(w["combined"]) ** 2
        ),
    }
    print("w_combined decomposition:", json.dumps(decomp, indent=2))

    # 3) span-sufficiency: fit on combined TRAIN, score CF-Eval zero-shot + in-domain tests
    train_X, train_y = [], []
    tests = {}
    for ds in POOLER:
        df, emb = load_pkl(POOLER[ds])
        Xtr, mtr = split(df, emb, "train")
        train_X.append(Xtr)
        train_y.append(mtr["label"].to_numpy().astype(int))
        ev = "test" if (df["split"] == "test").any() else "validation"
        tests[ds] = split(df, emb, ev)
    Xtr = np.concatenate(train_X)
    ytr = np.concatenate(train_y)
    cf = pickle.load(open(CF_EMB, "rb"))
    cf_meta, cf_X = cf["df"].reset_index(drop=True), np.asarray(cf["embeddings"], dtype=np.float64)

    variants: dict[str, list[str] | None] = {
        "span_synthclic_only": ["synthclic"],
        "span_cnnspot_only": ["cnnspot"],
        "span_2d_synthclic+cnnspot": ["synthclic", "cnnspot"],
        "span_3d_all_single_domain": ["synthclic", "synthbuster", "cnnspot"],
        "full_1024d_retrained": None,  # upper bound: full-rank logistic on the same data
    }
    rows = []

    def feats(X: np.ndarray, dirs: list[str] | None) -> np.ndarray:
        if dirs is None:
            return X
        return np.stack([X @ unit(w[d]) for d in dirs], axis=1)

    for vname, dirs in variants.items():
        F = feats(Xtr, dirs)
        sc = StandardScaler().fit(F)
        lr = LogisticRegression(C=1.0, max_iter=5000, random_state=SEED).fit(sc.transform(F), ytr)

        def score(X: np.ndarray) -> np.ndarray:
            return lr.predict_proba(sc.transform(feats(X, dirs)))[:, 1]

        r = {"variant": vname, "n_dims": F.shape[1]}
        r.update(
            {
                f"cf_{k}": v
                for k, v in eval_scores("cf_eval", cf_meta, score(cf_X), "matched").items()
                if k != "eval"
            }
        )
        for ds, (Xe, me) in tests.items():
            r[f"{ds}_test_mAP"] = per_generator_map(
                me.assign(score=score(Xe)), real_pairing=pairing_for_dataset(ds)
            )
        rows.append(r)
        print(f"{vname}: cf mAP-by-gen {r['cf_mAP_by_gen']:.4f}")

    # reference: the actual combined checkpoint probe on the same CF embeddings (sanity check
    # against the published row) and on the in-domain tests
    with torch.no_grad():
        pass  # scores are a plain affine map; no torch needed
    for pname in ["combined", "synthclic", "cnnspot"]:
        s_cf = cf_X @ w[pname]  # monotone in P(fake); AP/AUROC are rank metrics
        r = {"variant": f"checkpoint_probe_{pname}", "n_dims": 1024}
        r.update(
            {
                f"cf_{k}": v
                for k, v in eval_scores("cf_eval", cf_meta, s_cf, "matched").items()
                if k != "eval"
            }
        )
        for ds, (Xe, me) in tests.items():
            r[f"{ds}_test_mAP"] = per_generator_map(
                me.assign(score=Xe @ w[pname]), real_pairing=pairing_for_dataset(ds)
            )
        rows.append(r)

    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT / "span_sufficiency_eval.csv", index=False)
    (OUT / "geometry.json").write_text(
        json.dumps(
            {
                "pairwise_cosines": {
                    f"{a}~{b}": float(cos.loc[a, b])
                    for i, a in enumerate(names)
                    for b in names[i + 1 :]
                },
                "combined_decomposition": decomp,
                "seed": SEED,
                "train_n": int(len(ytr)),
            },
            indent=2,
        )
    )
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(tbl.round(4).to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
