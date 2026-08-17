#!/usr/bin/env python
"""E8 capstone: per-attention-head PAIRED shift — which heads consistently encode the real->synthetic
change with CONTENT held fixed (SynthCLIC pairing) + Gandelsman head decomposition + the detector direction.

For each (real, synthetic-counterpart) pair, run CLIP on both, compute each head's contribution to the
deterministic detector logit (LN-linearized, as in analyze_head_decomposition), and take the paired
difference delta = contrib(syn) - contrib(real). Aggregate across pairs: effect size = mean/std. Heads with
large consistent +delta consistently push toward 'synthetic' when only the real->synth transformation changes.

P-Q1.a (head x cue attribution map): alongside the scalar logit delta, accumulate each head's 768-d CLIP-text-
space contribution (full ln_post linearization, faithful to name_heads.py) and take the paired synth-real mean
direction, then cosine it to the 168 antonym difference cues -> a head x cue matrix that names WHICH cue each
head carries, content-controlled. Fuses thread 12 (heads) and thread 13C (cues) in one frame.

P-Q1.b (per-generator): --gen all loops imagen3 / SD3-medium / FLUX.1-dev / FLUX.1-schnell and reports whether
the same heads encode the shift across diffusion families (Spearman rank-corr of per-head effect-size vectors).

Run: python scripts/analyze/analyze_paired_heads.py --gen all --n-pairs 400
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
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoProcessor, CLIPVisionModel, CLIPVisionModelWithProjection

MODEL = "openai/clip-vit-large-patch14-336"
OUT = Path("outputs/e8/paired")
OUT.mkdir(parents=True, exist_ok=True)
GENS = ["imagen3", "SD3-medium", "FLUX.1-dev", "FLUX.1-schnell"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gen", default="all", help="a generator name, or 'all' for the 4 diffusion families"
    )
    ap.add_argument("--n-pairs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ntop-cue", type=int, default=6, help="top antonym cues per head in the map")
    a = ap.parse_args()
    gens = GENS if a.gen == "all" else [a.gen]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # detector direction (pooler logreg) — the deterministic head we interpret
    d = pickle.load(open("data/embeddings/synthclic_clip_large_patch14.pkl", "rb"))
    df = d["df"]
    emb = d["embeddings"].astype(np.float64)
    m = (df["split"] == "train").to_numpy()
    sc = StandardScaler().fit(emb[m])
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(
        sc.transform(emb[m]), df.loc[m, "label"].to_numpy()
    )
    w = torch.tensor(lr.coef_.ravel() / sc.scale_, dtype=torch.float32, device=dev)

    model = CLIPVisionModel.from_pretrained(MODEL, cache_dir="data/hf_cache").to(dev).eval()
    cfg = model.config
    proc = AutoProcessor.from_pretrained(MODEL, cache_dir="data/hf_cache")
    H = cfg.num_attention_heads
    L = cfg.num_hidden_layers
    dh = cfg.hidden_size // H
    gamma = model.post_layernorm.weight.detach()
    eps = model.post_layernorm.eps
    Wo = [lyr.self_attn.out_proj.weight.detach() for lyr in model.encoder.layers]
    Wp = torch.tensor(
        CLIPVisionModelWithProjection.from_pretrained(MODEL, cache_dir="data/hf_cache")
        .visual_projection.weight.detach()
        .numpy(),
        dtype=torch.float32,
        device=dev,
    )  # (768,1024) CLIP-text-aligned projection (P-Q1.a)

    # antonym difference-cue basis (768), for the head x cue attribution map
    av = torch.load("data/embeddings/antonyms_diff_embeddings.pt")
    An = list(av["vocabulary"])
    A = av["embeddings"].numpy().astype(np.float64)
    A = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)

    cap = {}
    for layer, lyr in enumerate(model.encoder.layers):

        def mk(layer):
            def hook(mod, inp):
                cap[layer] = inp[0].detach()

            return hook

        lyr.self_attn.out_proj.register_forward_pre_hook(mk(layer))

    ds = load_dataset("marco-willi/synthclic")["train"]
    # index real + each synthetic generator by image_id WITHOUT decoding images (metadata columns only)
    meta = ds.select_columns(["source", "image_id"])
    real_by = {}
    syn_by = {g: {} for g in gens}
    for i, row in enumerate(meta):
        src, iid = row["source"], row["image_id"]
        if src == "clic2020":
            real_by[iid] = i
        elif src in syn_by:
            syn_by[src][iid] = i

    Wor = [Wo[layer].view(cfg.hidden_size, H, dh) for layer in range(L)]

    @torch.no_grad()
    def per_image(imgs):
        """Return (scalar per-head logit contrib (bs,L*H), projected per-head dir (bs,L*H,768))."""
        inp = proc(images=imgs, return_tensors="pt").to(dev)
        out = model(**inp)
        cls = out.last_hidden_state[:, 0, :]
        sig = torch.sqrt(cls.var(-1, unbiased=False, keepdim=True) + eps)  # (bs,1)
        weff = w * gamma / sig  # (bs,1024) detector-through-LN linearization
        bs = cls.shape[0]
        scal = np.zeros((bs, L * H))
        proj = np.zeros((bs, L * H, 768))
        for layer in range(L):
            z = cap[layer][:, 0, :].view(bs, H, dh)
            c = torch.einsum("bhd,ohd->bho", z, Wor[layer])  # (bs,H,1024) per-head contribution
            cc = c - c.mean(-1, keepdim=True)  # ln_post: subtract mean
            scal[:, layer * H : (layer + 1) * H] = (
                torch.einsum("bho,bo->bh", cc, weff).cpu().numpy()
            )
            u = gamma * cc / sig.unsqueeze(1)  # (bs,H,1024) ln_post-linearized, gamma applied
            proj[:, layer * H : (layer + 1) * H, :] = (
                torch.einsum("bho,po->bhp", u, Wp).cpu().numpy()
            )
        return scal, proj

    summ = {"n_pairs_requested": a.n_pairs, "generators": {}}
    eff_by_gen = {}
    for g in gens:
        iids = [k for k in real_by if k in syn_by[g]][: a.n_pairs]
        print(f"\n=== {g}: {len(iids)} pairs ===", flush=True)
        real_s = np.zeros((len(iids), L * H))
        syn_s = np.zeros((len(iids), L * H))
        sum_proj_delta = np.zeros((L * H, 768))  # accumulate paired synth-real projected direction
        for s in range(0, len(iids), a.batch_size):
            chunk = iids[s : s + a.batch_size]
            rim = [ds[real_by[k]]["image"].convert("RGB") for k in chunk]
            sim = [ds[syn_by[g][k]]["image"].convert("RGB") for k in chunk]
            rs, rp = per_image(rim)
            ss, sp = per_image(sim)
            real_s[s : s + len(chunk)] = rs
            syn_s[s : s + len(chunk)] = ss
            sum_proj_delta += (sp - rp).sum(0)
            if s % (a.batch_size * 5) == 0:
                print(f"  {s + len(chunk)}/{len(iids)}", flush=True)

        delta = syn_s - real_s  # (n_pairs, L*H) paired per-head logit shift
        mean = delta.mean(0)
        std = delta.std(0)
        eff = mean / np.clip(std, 1e-9, None)
        eff_by_gen[g] = eff
        rows = [
            {
                "layer": i // H,
                "head": i % H,
                "mean_delta": float(mean[i]),
                "effect_size": float(eff[i]),
                "frac_same_sign": float(max((delta[:, i] > 0).mean(), (delta[:, i] < 0).mean())),
            }
            for i in range(L * H)
        ]
        tab = pd.DataFrame(rows).sort_values("effect_size", key=np.abs, ascending=False)

        # ---- P-Q1.a: head x cue attribution map (cosine of each head's mean paired dir to antonym cues) ----
        head_dir = sum_proj_delta / max(
            len(iids), 1
        )  # (L*H,768) mean paired synth-real dir per head
        hd_unit = head_dir / np.clip(np.linalg.norm(head_dir, axis=1, keepdims=True), 1e-12, None)
        cue_cos = hd_unit @ A.T  # (L*H, n_cues) signed alignment per head x cue
        map_rows = []
        for i in range(L * H):
            o = np.argsort(-np.abs(cue_cos[i]))[: a.ntop_cue]
            map_rows.append(
                {
                    "layer": i // H,
                    "head": i % H,
                    "effect_size": float(eff[i]),
                    "max_abs_cue_cos": float(np.abs(cue_cos[i]).max()),
                    "top_cues": ";".join(An[j] for j in o),
                    "top_cue_cos": ";".join(f"{cue_cos[i][j]:.3f}" for j in o),
                }
            )
        cue_map = pd.DataFrame(map_rows).sort_values("effect_size", key=np.abs, ascending=False)

        gtag = g.replace(".", "").replace("-", "")
        tab.to_csv(OUT / f"paired_head_shifts_{gtag}.csv", index=False)
        cue_map.to_csv(OUT / f"paired_head_cue_map_{gtag}.csv", index=False)
        # full head x cue matrix (for figures / downstream)
        np.savez_compressed(
            OUT / f"paired_head_cue_matrix_{gtag}.npz",
            cue_cos=cue_cos,
            effect_size=eff,
            cues=np.array(An),
            layer=np.array([i // H for i in range(L * H)]),
            head=np.array([i % H for i in range(L * H)]),
        )
        share = np.cumsum(np.sort(np.abs(mean))[::-1])
        share /= share[-1]
        summ["generators"][g] = {
            "n_pairs": len(iids),
            "top_heads": tab.head(12).to_dict("records"),
            "abs_mean_delta_share_top5": float(share[4]),
            "abs_mean_delta_share_top10": float(share[9]),
            "total_mean_paired_logit_shift": float(mean.sum()),
            "top_head_cue_attribution": cue_map.head(8)[
                ["layer", "head", "effect_size", "top_cues", "top_cue_cos"]
            ].to_dict("records"),
        }
        print(f"\n[{g}] top heads by paired effect size (+=toward synthetic):")
        print(tab.head(8).to_string(index=False))
        print(f"\n[{g}] head x cue attribution (top heads):")
        print(
            cue_map.head(8)[
                ["layer", "head", "effect_size", "max_abs_cue_cos", "top_cues"]
            ].to_string(index=False)
        )

    # ---- P-Q1.b: do the same heads encode the shift across generators? (rank-corr of effect vectors) ----
    if len(gens) > 1:
        agree = {
            f"{gi}|{gj}": float(spearmanr(eff_by_gen[gi], eff_by_gen[gj]).statistic)
            for i, gi in enumerate(gens)
            for gj in gens[i + 1 :]
        }
        # pooled effect = mean across generators; report the consensus top heads
        pooled = np.mean([eff_by_gen[g] for g in gens], axis=0)
        order = np.argsort(-np.abs(pooled))[:12]
        summ["per_generator_head_effect_rank_corr"] = agree
        summ["consensus_top_heads"] = [
            {"layer": int(i // H), "head": int(i % H), "mean_effect_size": float(pooled[i])}
            for i in order
        ]
        print(
            "\nP-Q1.b head-effect rank-corr across generators:",
            {k: round(v, 3) for k, v in agree.items()},
        )
        print(
            "consensus top heads (mean |effect| across gens):",
            [(int(i // H), int(i % H), round(float(pooled[i]), 2)) for i in order[:8]],
        )

    (OUT / "paired_heads_summary.json").write_text(json.dumps(summ, indent=2))
    print("\nwrote", OUT / "paired_heads_summary.json")


if __name__ == "__main__":
    main()
