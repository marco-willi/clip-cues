#!/usr/bin/env python
"""E8 causal capstone: forward-pass MEAN-ABLATION of attention heads (Gandelsman compute_ablations style).

Replace an ablated head's output (its slice of the out_proj input, all tokens) with its dataset per-position
MEAN, recompute the embedding, re-apply the deterministic detector, measure AUROC:
  - keep-only-top-k : ablate ALL heads except the top-k -> CAUSAL SUFFICIENCY (do the top-k alone detect?)
  - ablate-top-k    : ablate only the top-k             -> NECESSITY (how much does removing them hurt?)
  - ablate-single   : ablate the single best head       -> e.g. GAN L8/H11
Images are pre-tokenized ONCE and reused across all forward passes (8x speedup).
Run: python scripts/analyze/analyze_head_ablation.py --dataset cnnspot --eval-split validation
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoProcessor, CLIPVisionModel

from clip_cues_research.analysis.metrics import per_generator_map

MODEL = "openai/clip-vit-large-patch14-336"
OUT = Path("outputs/e8/head_decomp")
POOLER = {
    "synthclic": "data/embeddings/synthclic_clip_large_patch14.pkl",
    "cnnspot": "data/embeddings/cnnspot_clip_large_patch14.pkl",
}
HF = {"synthclic": "marco-willi/synthclic", "cnnspot": "marco-willi/cnnspot-small"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eval-split", default=None)
    ap.add_argument("--max-images", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--topk", type=int, nargs="+", default=[1, 5])
    a = ap.parse_args()
    ds = a.dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = pickle.load(open(POOLER[ds], "rb"))
    df = d["df"]
    emb = d["embeddings"].astype(np.float64)
    mtr = (df.split == "train").to_numpy()
    sc = StandardScaler().fit(emb[mtr])
    lr = LogisticRegression(C=1, max_iter=5000).fit(
        sc.transform(emb[mtr]), df.label[mtr].to_numpy()
    )

    def detect(pooler):
        return lr.decision_function(sc.transform(pooler))

    summ = json.load(open(OUT / f"{ds}_summary.json"))
    top = [(t["layer"], t["head"]) for t in summ["top_heads"]]
    M = CLIPVisionModel.from_pretrained(MODEL, cache_dir="data/hf_cache").to(dev).eval()
    cfg = M.config
    proc = AutoProcessor.from_pretrained(MODEL, cache_dir="data/hf_cache")
    H = cfg.num_attention_heads
    L = cfg.num_hidden_layers
    dh = cfg.hidden_size // H
    g = M.post_layernorm.weight.detach()
    b = M.post_layernorm.bias.detach()
    eps = M.post_layernorm.eps
    ev = a.eval_split or ("test" if (df.split == "test").any() else "validation")
    dsi = load_dataset(HF[ds])[ev]
    n = min(a.max_images, len(dsi))
    idx = np.linspace(0, len(dsi) - 1, n).astype(int)
    y = np.array([int(dsi[int(i)]["label"]) for i in idx])
    src = np.array([str(dsi[int(i)]["source"]) for i in idx])  # for Convention-A mAP
    pair = "matched" if ds == "cnnspot" else "shared"
    # PRE-TOKENIZE ONCE -> cpu pixel batches (reused across all configs)
    pv = []
    for s in range(0, n, a.batch_size):
        imgs = [dsi[int(i)]["image"].convert("RGB") for i in idx[s : s + a.batch_size]]
        pv.append(proc(images=imgs, return_tensors="pt")["pixel_values"])
    print(f"[{ds}] pre-tokenized {n} images into {len(pv)} batches", flush=True)

    ABLATE = {}
    MEANZ = {}
    CAP = {}

    def prehook(lyr):
        def f(mod, inp):
            CAP[lyr] = inp[0].detach()
            ab = ABLATE.get(lyr)
            if ab:
                z = inp[0].clone()
                for h in ab:
                    z[:, :, h * dh : (h + 1) * dh] = MEANZ[lyr][:, h * dh : (h + 1) * dh].to(
                        z.dtype
                    )
                return (z,) + tuple(inp[1:])

        return f

    for lyr in range(L):
        M.encoder.layers[lyr].self_attn.out_proj.register_forward_pre_hook(prehook(lyr))

    @torch.no_grad()
    def forward_pooler(pix):
        out = M(pixel_values=pix.to(dev))
        cls = out.last_hidden_state[:, 0, :]
        mu = cls.mean(-1, keepdim=True)
        sig = torch.sqrt(cls.var(-1, unbiased=False, keepdim=True) + eps)
        return (g * (cls - mu) / sig + b).cpu().numpy()

    # pass 1: per-position mean z (no ablation)
    ABLATE.clear()
    zsum = {lyr: None for lyr in range(L)}
    cnt = 0
    for pix in pv:
        forward_pooler(pix)
        for lyr in range(L):
            s = CAP[lyr].sum(0)
            zsum[lyr] = s if zsum[lyr] is None else zsum[lyr] + s
        cnt += pix.shape[0]
    for lyr in range(L):
        MEANZ[lyr] = (zsum[lyr] / cnt).detach()

    def metrics(ablate_map):
        """Forward-pass with the given heads mean-ablated -> (mAP Convention-A, AUROC)."""
        ABLATE.clear()
        ABLATE.update({lyr: set(hs) for lyr, hs in ablate_map.items()})
        pool = np.concatenate([forward_pooler(pix) for pix in pv], 0)
        ABLATE.clear()
        s = detect(pool)
        mp = float(
            per_generator_map(
                pd.DataFrame({"label": y, "score": s, "source": src}), real_pairing=pair
            )
        )
        return mp, float(roc_auc_score(y, s))

    base_map, base_auroc = metrics({})
    print(f"[{ds}] baseline mAP={base_map:.3f} AUROC={base_auroc:.3f}", flush=True)
    res = {
        "dataset": ds,
        "eval": ev,
        "n": int(n),
        "metric": "per_generator_mAP (Convention A); auroc kept for reference",
        "baseline_map": base_map,
        "baseline_auroc": base_auroc,
        "keep_only_topk": {},  # mAP
        "ablate_topk": {},  # mAP
        "keep_only_topk_auroc": {},
        "ablate_topk_auroc": {},
    }
    for k in a.topk:
        keep = set(top[:k])
        ko_map, ko_auroc = metrics(
            {lyr: set(h for h in range(H) if (lyr, h) not in keep) for lyr in range(L)}
        )
        abt = {}
        [abt.setdefault(lyr, set()).add(h) for (lyr, h) in top[:k]]
        ab_map, ab_auroc = metrics(abt)
        res["keep_only_topk"][k], res["keep_only_topk_auroc"][k] = ko_map, ko_auroc
        res["ablate_topk"][k], res["ablate_topk_auroc"][k] = ab_map, ab_auroc
        print(
            f"  k={k}: keep-only mAP={ko_map:.3f}  ablate-top{k} mAP={ab_map:.3f}",
            flush=True,
        )
    (l0, h0) = top[0]
    sb_map, _ = metrics({l0: {h0}})
    res["ablate_single_best"] = {f"L{l0}H{h0}": sb_map}
    print(f"  ablate single best L{l0}/H{h0}: mAP={sb_map:.3f}", flush=True)
    (OUT / f"{ds}_ablation.json").write_text(json.dumps(res, indent=2))
    print("wrote", OUT / f"{ds}_ablation.json")


if __name__ == "__main__":
    main()
