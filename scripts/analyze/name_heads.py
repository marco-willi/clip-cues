#!/usr/bin/env python
"""E8: name the discriminative attention heads (Gandelsman text interpretation).

For the top heads from analyze_head_decomposition, capture each head's contribution vector c_{l,h}[CLS]
(1024-d) per image, map it to CLIP text space via the FULL ln_post linearization
(gamma*(c-mean(c))/sigma then visual_projection — faithful to Gandelsman), take the synthetic-minus-real
mean, and report the nearest antonym + TextSpan descriptions. Names WHAT each discriminative head encodes.
Run: python scripts/analyze/name_heads.py --dataset synthclic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoProcessor, CLIPVisionModel, CLIPVisionModelWithProjection

MODEL = "openai/clip-vit-large-patch14-336"
HF = {"synthclic": "marco-willi/synthclic", "cnnspot": "marco-willi/cnnspot-small"}
OUT = Path("outputs/e8/head_decomp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthclic")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--ntop", type=int, default=10)
    ap.add_argument("--max-images", type=int, default=1600)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    ds = a.dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    summ = json.load(open(OUT / f"{ds}_summary.json"))
    heads = [(t["layer"], t["head"]) for t in summ["top_heads"]][: a.ntop]
    model = CLIPVisionModel.from_pretrained(MODEL, cache_dir="data/hf_cache").to(dev).eval()
    cfg = model.config
    proc = AutoProcessor.from_pretrained(MODEL, cache_dir="data/hf_cache")
    H = cfg.num_attention_heads
    dh = cfg.hidden_size // H
    gamma = model.post_layernorm.weight.detach()
    Wo = {
        layer: model.encoder.layers[layer].self_attn.out_proj.weight.detach() for layer, _ in heads
    }
    # text bases
    av = torch.load("data/embeddings/antonyms_diff_embeddings.pt")
    A = av["embeddings"].numpy().astype(np.float64)
    An = list(av["vocabulary"])
    A = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    ts = torch.load("data/embeddings/textspan_embeddings.pt")
    T = ts["embeddings"].numpy().astype(np.float64)
    Tn = list(ts["vocabulary"])
    T = T / np.clip(np.linalg.norm(T, axis=1, keepdims=True), 1e-12, None)
    Wp = (
        CLIPVisionModelWithProjection.from_pretrained(MODEL, cache_dir="data/hf_cache")
        .visual_projection.weight.detach()
        .numpy()
        .astype(np.float64)
    )
    cap = {}
    for layer, _ in heads:

        def mk(layer):
            def hook(m, i):
                cap[layer] = i[0].detach()

            return hook

        model.encoder.layers[layer].self_attn.out_proj.register_forward_pre_hook(mk(layer))
    dsimg = load_dataset(HF[ds])[a.eval_split]
    n = min(a.max_images, len(dsimg))
    idx = np.linspace(0, len(dsimg) - 1, n).astype(int)
    labels = np.array([int(dsimg[int(i)]["label"]) for i in idx])
    # accumulate in the 768-d PROJECTED space, applying the FULL ln_post linearization per image
    # (gamma*(c-mean(c))/sigma) then visual_projection — faithful to Gandelsman (was: gamma*d only).
    eps = model.post_layernorm.eps
    Wp_t = torch.tensor(Wp, device=dev, dtype=torch.float32)  # (768,1024)
    sums = {(layer, h): {0: np.zeros(768), 1: np.zeros(768)} for layer, h in heads}
    cnt = {0: 0, 1: 0}
    for s in range(0, n, a.batch_size):
        b = [dsimg[int(i)]["image"].convert("RGB") for i in idx[s : s + a.batch_size]]
        lb = labels[s : s + len(b)]
        inp = proc(images=b, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**inp)
            cls = out.last_hidden_state[:, 0, :]  # (bsz,1024)
            sig = torch.sqrt(cls.var(-1, unbiased=False, keepdim=True) + eps)  # (bsz,1)
            for layer, h in heads:
                z = cap[layer][:, 0, h * dh : (h + 1) * dh]  # (bsz,dh)
                c = z @ Wo[layer][:, h * dh : (h + 1) * dh].T  # (bsz,1024)
                u = gamma * (c - c.mean(-1, keepdim=True)) / sig  # ln_post linearization
                proj = (u @ Wp_t.T).cpu().numpy()  # (bsz,768) cross-modal
                for cls_ in (0, 1):
                    msk = lb == cls_
                    if msk.any():
                        sums[(layer, h)][cls_] += proj[msk].sum(0)
        for cls_ in (0, 1):
            cnt[cls_] += int((lb == cls_).sum())
        if s % (a.batch_size * 10) == 0:
            print(f"  {ds} {s + len(b)}/{n}", flush=True)
    res = []
    for layer, h in heads:
        d = (
            sums[(layer, h)][1] / cnt[1] - sums[(layer, h)][0] / cnt[0]
        )  # synth-real PROJECTED direction (768)
        p = d / np.clip(np.linalg.norm(d), 1e-12, None)
        ca = (
            A @ p
        )  # cosine of the (unit) head synth-real direction with each (unit) antonym direction
        ct = T @ p
        oa = np.argsort(-np.abs(ca))[:6]
        ot = np.argsort(-np.abs(ct))[:6]
        top_ant = [An[i] for i in oa]
        top_ts = [Tn[i] for i in ot]
        res.append(
            {
                "layer": layer,
                "head": h,
                "top_antonyms": top_ant,
                "antonym_cos": [float(ca[i]) for i in oa],  # signed cosine alignment (≈0.1 ⇒ weak)
                "max_abs_cos_antonym": float(np.abs(ca).max()),
                "top_textspan": top_ts,
                "textspan_cos": [float(ct[i]) for i in ot],
                "max_abs_cos_textspan": float(np.abs(ct).max()),
            }
        )
        print(f"head (L{layer},H{h}) maxcos={np.abs(ca).max():.3f}: antonyms={top_ant}")
    (OUT / f"{ds}_head_names.json").write_text(json.dumps(res, indent=2))
    print("wrote", OUT / f"{ds}_head_names.json")


if __name__ == "__main__":
    main()
