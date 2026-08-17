#!/usr/bin/env python
"""E8 deep-dive: is the *relevant* ortho direction (geometry + semantics) stable across seeds?

Refines the headline (all-8 averaged) stability with three analyses over the saved per-fit artifacts
in results/e8_interpretability_stability/ortho/<run_id>/fit_*.npz (each holds W_L1 (8,1024) + importance (8)):

  1. GEOMETRY  — top-1<->top-1 cosine, top-1 best-match (position-agnostic), top-2/3 important-subspace
     principal-angle alignment, vs the all-8 reference. Answers "is the direction that matters preserved
     more than the unused ones?".
  2. SEMANTICS — project each fit's most-relevant direction through CLIP visual_projection, rank the 168
     antonym concepts, Jaccard of top-k across seeds + a consensus list (freq-across-seeds, mean signed cos).
     Answers "are the interpreted concepts stable even if the vector rotates?" (chance |cos| ~ 1/sqrt(768) = 0.036).
  3. PER-SEED  — the actual ranked top-k antonym terms for the most-relevant direction, one line per seed.

Conclusions (2026-06-26 run) are recorded in memory [[e8-interpretability-stability]].
Run: python scripts/analyze/analyze_direction_interpretation_stability.py
"""

from __future__ import annotations

import glob
import itertools
import json

import numpy as np
import torch
from scipy.linalg import subspace_angles
from transformers import CLIPVisionModelWithProjection


def load_runs():
    out = {}
    for sj in glob.glob("results/e8_interpretability_stability/ortho/*/stability.json"):
        m = json.load(open(sj))
        rd = sj.rsplit("/", 1)[0]
        fits = sorted(
            glob.glob(rd + "/fit_*.npz"), key=lambda p: int(p.split("fit_")[1].split(".")[0])
        )
        Ws = [
            np.load(f)["W_L1"]
            / np.clip(np.linalg.norm(np.load(f)["W_L1"], axis=1, keepdims=True), 1e-12, None)
            for f in fits
        ]
        imps = [np.abs(np.load(f)["importance"]) for f in fits]
        sgn = [np.sign(np.load(f)["importance"]) for f in fits]
        out[(m["dataset"], m["regime"])] = (Ws, imps, sgn)
    return out


def geometry(runs):
    def sub(Wa, ia, Wb, ib, k):
        A = Wa[np.argsort(-ia)[:k]]
        B = Wb[np.argsort(-ib)[:k]]
        return float(np.mean(np.cos(subspace_angles(A.T, B.T))))

    print("\n## 1. GEOMETRY — preservation of the most-relevant direction(s)")
    print(f"{'dataset':9} {'regime':12} | top1<->top1  top1-bestmatch  top2-subspace  all8")
    for key in sorted(runs):
        Ws, imps, _ = runs[key]
        P = list(itertools.combinations(range(len(Ws)), 2))
        t11 = [abs(Ws[j][np.argmax(imps[j])] @ Ws[i][np.argmax(imps[i])]) for i, j in P]
        t1b = [
            0.5
            * (
                np.max(np.abs(Ws[j] @ Ws[i][np.argmax(imps[i])]))
                + np.max(np.abs(Ws[i] @ Ws[j][np.argmax(imps[j])]))
            )
            for i, j in P
        ]
        s2 = [sub(Ws[i], imps[i], Ws[j], imps[j], 2) for i, j in P]
        a8 = [sub(Ws[i], imps[i], Ws[j], imps[j], 8) for i, j in P]
        print(
            f"{key[0]:9} {key[1]:12} |  {np.mean(t11):.3f}        {np.mean(t1b):.3f}          {np.mean(s2):.3f}        {np.mean(a8):.3f}"
        )


def _vocab():
    av = torch.load("data/embeddings/antonyms_diff_embeddings.pt")
    A = av["embeddings"].numpy().astype(np.float64)
    A /= np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    Wp = (
        CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14-336", cache_dir="data/hf_cache"
        )
        .visual_projection.weight.detach()
        .numpy()
        .astype(np.float64)
    )
    return A, list(av["vocabulary"]), Wp


def semantics(runs, K=10):
    A, names, Wp = _vocab()

    def terms(d, s):
        p = (d * s) @ Wp.T
        p /= np.clip(np.linalg.norm(p), 1e-12, None)
        cos = A @ p
        idx = np.argsort(-np.abs(cos))[:K]
        return idx, cos

    print(f"\n## 2. SEMANTICS — top-{K} antonym concept stability (chance |cos|~0.036)")
    for key in sorted(runs):
        Ws, imps, sgn = runs[key]
        sets = []
        freq = {}
        for W, imp, sg in zip(Ws, imps, sgn):
            t1 = int(np.argmax(imp))
            idx, cos = terms(W[t1], sg[t1])
            sets.append(set(idx.tolist()))
            for i in idx:
                freq.setdefault(i, []).append(float(cos[i]))
        J = [len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)]
        cons = sorted(freq.items(), key=lambda kv: -len(kv[1]))[:6]
        print(
            f"  {key[0]:9} {key[1]:12} Jaccard={np.mean(J):.3f}±{np.std(J):.3f} | consensus: "
            + ", ".join(f"{names[i]}({len(v)}/{len(Ws)})" for i, v in cons)
        )


def per_seed(runs, K=8, which=(("synthclic", "vary-init"), ("cnnspot", "vary-init"))):
    A, names, Wp = _vocab()
    print("\n## 3. PER-SEED ranked terms (most-relevant direction)")
    for key in which:
        Ws, imps, sgn = runs[key]
        print(f"  --- {key[0]} {key[1]} ---")
        for s, (W, imp, sg) in enumerate(zip(Ws, imps, sgn)):
            t1 = int(np.argmax(imp))
            p = (W[t1] * sg[t1]) @ Wp.T
            p /= np.clip(np.linalg.norm(p), 1e-12, None)
            cos = A @ p
            top = np.argsort(-np.abs(cos))[:K]
            print(f"    seed {s}: " + ", ".join(f"{names[i]}({cos[i]:+.2f})" for i in top))


if __name__ == "__main__":
    runs = load_runs()
    geometry(runs)
    semantics(runs)
    per_seed(runs)
