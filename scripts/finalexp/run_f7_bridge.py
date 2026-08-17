#!/usr/bin/env python
"""F7 (addition, not in the spec): the bridge from the matched heads to everything they replace.

The consolidation retrains the detector inventory. That risks orphaning the existing interpretation
record, which targets either the **deployed checkpoints** (E12/N23-N25) or the **CV-tuned sklearn
proxies** P768t/P1024t (N7-N14, N18). F7 measures how closely the new matched heads reproduce each,
so downstream numbers either transfer by citation or come with a scoped re-run list.

Battery (the N19 bridge statistics): data-metric (Sigma) cosine — which *is* N19's "whitened
cosine", see stability.whitened_cosine — logit Pearson/Spearman, decision agreement at logit
threshold 0, error Jaccard, and cue-profile Spearman.

Decision rule, stated in advance:
    Sigma cos >= 0.9 AND cue-profile rho >= 0.9 AND decision agreement >= 0.95
    => downstream N-numbers transfer by citation; the consolidation is documentation-only.

Also reports the **augmentation effect** as a by-product: AUROC(deployed) - AUROC(matched D_h). E12
attributed the deployed-vs-proxy gap to the augmented training protocol; if that difference is small
here, the gap is a regularization effect instead and E12's sentence needs correcting.

    uv run python scripts/finalexp/run_f7_bridge.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from clip_cues_research.finalexp import data as D
from clip_cues_research.finalexp import profiles, spaces, stability
from clip_cues_research.finalexp.runner import EXPERIMENTS_ROOT, Run

EXPERIMENT = "F7-bridge"
PRIMARY_SEED = 123
# The CV-tuned proxies as defined in EXTERNAL_VALIDATION_PROTOCOL.md / INTERPRETATION.md §0.
PROXY_C = {"P1024t": 0.03, "P768t": 0.01}
PASS_RULE = {"sigma_cosine": 0.9, "cue_profile_spearman": 0.9, "decision_agreement": 0.95}


def deployed_linear(name: str) -> tuple[np.ndarray, float]:
    """(w, b) of a deployed k=1 probe checkpoint."""
    sd = D.get_checkpoint(f"ckpt/{name}")
    w = np.asarray(sd["model.classification_head.fc.weight"], dtype=np.float64).ravel()
    b = float(np.asarray(sd["model.classification_head.fc.bias"]).ravel()[0])
    return w, b


def deployed_ortho(name: str) -> tuple[np.ndarray, float]:
    """Effective (w, b) of the deployed k=8 head — linear, so w_eff = w2 @ W1."""
    sd = D.get_checkpoint(f"ckpt/{name}")
    w1 = np.asarray(sd["model.classification_head.layers.0.weight"], dtype=np.float64)
    b1 = np.asarray(sd["model.classification_head.layers.0.bias"], dtype=np.float64)
    w2 = np.asarray(sd["model.classification_head.to_logits.weight"], dtype=np.float64)
    b2 = float(np.asarray(sd["model.classification_head.to_logits.bias"]).ravel()[0])
    return (w2 @ w1).ravel(), float((w2 @ b1).ravel()[0] + b2)


def fit_proxy(
    x_tr: np.ndarray, y_tr: np.ndarray, C: float
) -> tuple[StandardScaler, LogisticRegression]:
    sc = StandardScaler().fit(x_tr)
    return sc, LogisticRegression(C=C, max_iter=5000).fit(sc.transform(x_tr), y_tr)


def matched_logits(experiment: str, seed: int) -> np.ndarray:
    p = EXPERIMENTS_ROOT / experiment / f"runs/seed{seed}" / "logits_test.csv"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    return pd.read_csv(p)["logit"].to_numpy()


def matched_weight(experiment: str, seed: int) -> tuple[np.ndarray, float]:
    w = np.load(EXPERIMENTS_ROOT / experiment / f"runs/seed{seed}" / "weights.npz")
    return w["weight"], float(w["bias"][0])


def error_jaccard(za: np.ndarray, zb: np.ndarray, y: np.ndarray) -> float:
    """Jaccard of the two scorers' error sets at logit threshold 0."""
    ea, eb = (za > 0).astype(int) != y, (zb > 0).astype(int) != y
    union = (ea | eb).sum()
    return float((ea & eb).sum() / union) if union else float("nan")


def bridge(
    name_a: str,
    za: np.ndarray,
    wa: np.ndarray,
    name_b: str,
    zb: np.ndarray,
    wb: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    cues: np.ndarray,
) -> dict:
    """The full N19-style battery for one pair, plus a pass/fail against the stated rule.

    Direction metrics are only defined when both weights live in the **same** space. The
    ``D_h ~ D_e`` pair spans 1024-d and 768-d, so its direction columns are NaN by construction and
    the comparison rests on the score-level statistics — which is the correct treatment, not a
    limitation: there is no basis-free cosine between vectors in different spaces.
    """
    same_space = len(np.asarray(wa)) == len(np.asarray(wb)) == x.shape[1]
    direction = (
        stability.direction_agreement(wa, wb, x)
        if same_space
        else {"sigma_cosine": float("nan"), "raw_cosine": float("nan")}
    )
    row = {
        "pair": f"{name_a} ~ {name_b}",
        "target_a": name_a,
        "target_b": name_b,
        "same_space": same_space,
        **direction,
        **stability.score_agreement(za, zb),
        "error_jaccard": round(error_jaccard(za, zb, y), 6),
        "cue_profile_spearman": stability.profile_agreement(
            profiles.col_corr(za, cues), profiles.col_corr(zb, cues)
        ),
        "auroc_a": round(float(roc_auc_score(y, za)), 6),
        "auroc_b": round(float(roc_auc_score(y, zb)), 6),
    }
    # A cross-space pair cannot satisfy a direction criterion, so it is judged on the score-level
    # criteria alone rather than being failed on a metric that does not apply to it.
    row["passes_rule"] = bool(
        (np.isnan(row["sigma_cosine"]) or row["sigma_cosine"] >= PASS_RULE["sigma_cosine"])
        and row["cue_profile_spearman"] >= PASS_RULE["cue_profile_spearman"]
        and row["decision_agreement"] >= PASS_RULE["decision_agreement"]
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="synthclic")
    ap.add_argument("--seed", type=int, default=PRIMARY_SEED)
    args = ap.parse_args()

    inputs = [
        f"pooler/{args.dataset}",
        f"projected/{args.dataset}",
        f"cue_scores/{args.dataset}__antonyms",
        "ckpt/linear_probe_synthclic",
        "ckpt/clip_orthogonal_synthclic",
        "vocab/antonyms",
    ]
    print(f"F7: bridge on {args.dataset} (matched seed {args.seed})")

    # Both 768-d objects must live in ONE representation or the direction metrics mix units.
    # The proxies are standardized, hence scale-invariant, so using the same scale-matched space
    # F3 trains in leaves them identical models while making every weight comparable.
    pooler_space = spaces.load(args.dataset, "pooler")
    proj_space = spaces.load(args.dataset, "projected")
    split = pooler_space.df["split"].to_numpy()
    te, tr = split == "test", split == "train"
    y = pooler_space.df["label"].to_numpy().astype(int)
    yte, ytr = y[te], y[tr]

    h_te, h_tr = pooler_space.x[te], pooler_space.x[tr]
    e_te, e_tr = proj_space.x[te], proj_space.x[tr]
    cues_te = D.get_npz(f"cue_scores/{args.dataset}__antonyms")["scores"][te]

    # ── the four targets ─────────────────────────────────────────────────────────────────────
    w_dep, b_dep = deployed_linear("linear_probe_synthclic")
    w_k8, b_k8 = deployed_ortho("clip_orthogonal_synthclic")
    sc1024, lr1024 = fit_proxy(h_tr, ytr, PROXY_C["P1024t"])
    sc768, lr768 = fit_proxy(e_tr, ytr, PROXY_C["P768t"])

    z = {
        "matched_Dh": matched_logits("F1-canonical-stability", args.seed),
        "matched_De": matched_logits("F3-projected-head", args.seed),
        "deployed_k1": h_te @ w_dep + b_dep,
        "deployed_k8": h_te @ w_k8 + b_k8,
        "proxy_P1024t": lr1024.decision_function(sc1024.transform(h_te)),
        "proxy_P768t": lr768.decision_function(sc768.transform(e_te)),
    }
    # Effective raw-space normals, so direction metrics are comparable across targets.
    w = {
        "matched_Dh": matched_weight("F1-canonical-stability", args.seed)[0],
        "matched_De": matched_weight("F3-projected-head", args.seed)[0],
        "deployed_k1": w_dep,
        "deployed_k8": w_k8,
        "proxy_P1024t": lr1024.coef_.ravel() / sc1024.scale_,
        "proxy_P768t": lr768.coef_.ravel() / sc768.scale_,
    }
    space = {
        "matched_Dh": h_te,
        "deployed_k1": h_te,
        "deployed_k8": h_te,
        "proxy_P1024t": h_te,
        "matched_De": e_te,
        "proxy_P768t": e_te,
    }

    pairs = [
        ("matched_De", "proxy_P768t"),  # primary: does N7-N14/N18 transfer?
        ("matched_Dh", "proxy_P1024t"),  # primary
        ("matched_Dh", "deployed_k1"),  # keeps E12/N23-N25 attached
        ("matched_Dh", "deployed_k8"),  # connects F2 to the original k=8 checkpoint
        ("matched_Dh", "matched_De"),  # for context: the projection gap itself
        ("deployed_k1", "deployed_k8"),  # E12/N23 replication check (expect rho ~0.99)
    ]
    rows = [bridge(a, z[a], w[a], b, z[b], w[b], space[a], yte, cues_te) for a, b in pairs]
    df = pd.DataFrame(rows)

    run = Run(EXPERIMENT, "artifacts", inputs)
    run.save_csv("bridge.csv", df)

    aug = round(
        float(roc_auc_score(yte, z["deployed_k1"]) - roc_auc_score(yte, z["matched_Dh"])), 6
    )
    summary = {
        "experiment": EXPERIMENT,
        "spec_id": "addition",
        "dataset": args.dataset,
        "matched_seed": args.seed,
        "decision_rule": PASS_RULE,
        "pairs": rows,
        "auroc_by_target": {k: round(float(roc_auc_score(yte, v)), 6) for k, v in z.items()},
        "augmentation_effect": {
            "auroc_deployed_k1_minus_matched_Dh": aug,
            "interpretation": (
                "E12 attributed the deployed-vs-proxy AUROC gap (~0.02-0.04) to the deployed "
                "models' augmented training protocol. This is the direct measurement of that "
                "augmentation effect under otherwise identical features and evaluation."
            ),
        },
        "verdict": {
            "all_primary_pairs_pass": bool(
                all(
                    r["passes_rule"]
                    for r in rows
                    if r["pair"] in ("matched_De ~ proxy_P768t", "matched_Dh ~ proxy_P1024t")
                )
            ),
            "deployed_bridge_passes": bool(
                next(r["passes_rule"] for r in rows if r["pair"] == "matched_Dh ~ deployed_k1")
            ),
        },
        "proxy_definitions": {
            "P1024t": "standardized logistic, C=0.03, pooler",
            "P768t": "standardized logistic, C=0.01, derived projected",
        },
        "spaces": {"pooler": pooler_space.as_dict(), "projected": proj_space.as_dict()},
    }
    run.note(summary=summary)
    run.save_json("summary.json", summary)
    run.finish()

    print(
        "\n  pair                              sigma-cos  logit-rho  agree  errJacc  cue-rho  pass"
    )
    for r in rows:
        sig = "   n/a" if np.isnan(r["sigma_cosine"]) else f"{r['sigma_cosine']:+.3f}"
        print(
            f"  {r['pair']:33s} {sig}    {r['logit_spearman']:.3f}   "
            f"{r['decision_agreement']:.3f}   {r['error_jaccard']:.3f}   "
            f"{r['cue_profile_spearman']:.3f}   {'YES' if r['passes_rule'] else 'no'}"
        )
    print(
        "\n  AUROC by target: "
        + ", ".join(f"{k} {v:.4f}" for k, v in summary["auroc_by_target"].items())
    )
    print(f"  augmentation effect (deployed k=1 - matched D_h): {aug:+.4f}")


if __name__ == "__main__":
    main()
