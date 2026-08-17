#!/usr/bin/env python
"""E8: Gandelsman-style per-attention-head decomposition of the SID detector logit.

"Interpreting CLIP's Image Representation via Text-Based Decomposition" (Gandelsman et al.) decomposes the
CLIP [CLS] representation into additive contributions from individual attention heads. We apply it to the
DETECTOR: the final-LayerNorm is affine PER SAMPLE, so the detector logit decomposes EXACTLY into a sum of
per-head terms. For each (layer,head) we get its contribution to the real-vs-synth logit per image, then ask:
does the discriminative signal LOCALIZE to a few heads (interpretable) or spread over all 384 (diffuse)?

Math (post_layernorm: pooler = gamma*(cls-mu)/sigma + beta; logit = w.pooler):
  logit - w.beta = sum_components (w*gamma/sigma) . (component - mean_dims(component))
  attention head (l,h) contributes c_{l,h}[CLS] = z_l[CLS, head_slice] @ W_o_l[:, head_slice].T to cls.
  -> head logit term = (w*gamma/sigma) . (c_{l,h} - mean_dims(c_{l,h}))   [exact]

Needs a GPU pass over images (internal attention). Writes outputs/e8/head_decomp/.
Run: python scripts/analyze/analyze_head_decomposition.py --dataset synthclic [--max-images 1600]
"""

from __future__ import annotations

import argparse
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
POOLER = {
    "synthclic": "data/embeddings/synthclic_clip_large_patch14.pkl",
    "cnnspot": "data/embeddings/cnnspot_clip_large_patch14.pkl",
}
HF = {"synthclic": "marco-willi/synthclic", "cnnspot": "marco-willi/cnnspot-small"}
OUT = Path("outputs/e8/head_decomp")
OUT.mkdir(parents=True, exist_ok=True)


def detector_direction(ds):
    """Deterministic logreg direction in raw pooler space: logit = emb.(coef/scale) + const."""
    d = pickle.load(open(POOLER[ds], "rb"))
    df = d["df"]
    emb = d["embeddings"].astype(np.float64)
    m = (df["split"] == "train").to_numpy()
    sc = StandardScaler().fit(emb[m])
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(
        sc.transform(emb[m]), df.loc[m, "label"].to_numpy()
    )
    return (lr.coef_.ravel() / sc.scale_), df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(POOLER))
    ap.add_argument("--max-images", type=int, default=1600)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--eval-split",
        default=None,
        help="override eval split (cnnspot HF test=108k → use validation)",
    )
    a = ap.parse_args()
    ds = a.dataset
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w_np, df = detector_direction(ds)
    w = torch.tensor(w_np, dtype=torch.float32, device=dev)  # (1024,)

    model = CLIPVisionModel.from_pretrained(MODEL, cache_dir="data/hf_cache").to(dev).eval()
    proc = AutoProcessor.from_pretrained(MODEL, cache_dir="data/hf_cache")
    vm = model
    cfg = (
        model.config
    )  # transformers 5.x: CLIPVisionModel children are embeddings/encoder/post_layernorm
    H = cfg.num_attention_heads
    L = cfg.num_hidden_layers
    dh = cfg.hidden_size // H
    gamma = vm.post_layernorm.weight.detach()
    beta = vm.post_layernorm.bias.detach()
    eps = vm.post_layernorm.eps
    Wo = [lyr.self_attn.out_proj.weight.detach() for lyr in vm.encoder.layers]  # each (embed,embed)

    cap = {}

    def mk(layer):
        def hook(mod, inp):
            cap[layer] = inp[0].detach()  # input to out_proj: (bsz,seq,embed)

        return hook

    for layer, lyr in enumerate(vm.encoder.layers):
        lyr.self_attn.out_proj.register_forward_pre_hook(mk(layer))

    ev = a.eval_split or ("test" if (df["split"] == "test").any() else "validation")
    dsimg = load_dataset(HF[ds])[ev]
    n = min(a.max_images, len(dsimg))
    idx = np.linspace(0, len(dsimg) - 1, n).astype(int)
    labels = np.array([int(dsimg[int(i)]["label"]) for i in idx])
    sources = np.array([str(dsimg[int(i)]["source"]) for i in idx])  # for Convention-A mAP
    pair = "matched" if ds == "cnnspot" else "shared"  # cnnspot carries per-source reals

    head_logits = np.zeros((n, L * H), dtype=np.float64)
    recon = np.zeros(n)
    true_logit = np.zeros(n)
    for s in range(0, n, a.batch_size):
        b = [dsimg[int(i)]["image"].convert("RGB") for i in idx[s : s + a.batch_size]]
        inp = proc(images=b, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = vm(**inp)
            cls = out.last_hidden_state[:, 0, :]  # (bsz,1024) pre post-LN
            mu = cls.mean(-1, keepdim=True)
            sig = torch.sqrt(cls.var(-1, unbiased=False, keepdim=True) + eps)
            pooler = gamma * (cls - mu) / sig + beta
            tl = pooler @ w  # detector logit (no intercept)
            weff = w * gamma / sig  # (bsz,1024) per-image eff direction
            bs = cls.shape[0]
            per = np.zeros((bs, L * H))
            for layer in range(L):
                z = cap[layer][:, 0, :]  # (bsz,1024) out_proj input @ CLS
                zr = z.view(bs, H, dh)
                Wor = Wo[layer].view(cfg.hidden_size, H, dh)  # (out,H,dh)
                c = torch.einsum("bhd,ohd->bho", zr, Wor)  # (bsz,H,1024) per-head contribution
                cc = c - c.mean(-1, keepdim=True)
                contrib = torch.einsum("bho,bo->bh", cc, weff)  # (bsz,H)
                per[:, layer * H : (layer + 1) * H] = contrib.cpu().numpy()
            head_logits[s : s + bs] = per
            recon[s : s + bs] = per.sum(1)
            true_logit[s : s + bs] = (tl - pooler @ torch.zeros_like(w)).cpu().numpy()
        if s % (a.batch_size * 10) == 0:
            print(f"  {ds} {s + bs}/{n}", flush=True)

    # sanity: attention heads' share of the (centered) logit; recon vs (logit - w.beta)
    centered_logit = true_logit - float(beta @ w)
    recon_corr = float(np.corrcoef(recon, centered_logit)[0, 1])
    head_share = float(
        np.sum(recon * centered_logit) / np.sum(centered_logit**2)
    )  # LS share of logit from heads
    print(
        f"[{ds}] attention-head LS share of centered logit ~{head_share:.2f}; "
        f"corr(sum_heads, centered_logit)={recon_corr:.3f} (heads explain ~{recon_corr**2:.0%} of logit variance)"
    )

    # per-head discriminativeness
    rows = []
    for layer in range(L):
        for h in range(H):
            col = head_logits[:, layer * H + h]
            au = roc_auc_score(labels, col)
            rows.append(
                {
                    "layer": layer,
                    "head": h,
                    "auroc": au,
                    "abs_dev": abs(au - 0.5),
                    "mean_synth_minus_real": float(
                        col[labels == 1].mean() - col[labels == 0].mean()
                    ),
                }
            )
    hd = pd.DataFrame(rows).sort_values("abs_dev", ascending=False)
    hd.to_csv(OUT / f"{ds}_head_auroc.csv", index=False)
    # localization: cumulative |contribution| share of top-k heads, and AUROC of a logreg on top-k head logits
    order = hd["layer"].to_numpy() * H + hd["head"].to_numpy()  # head indices ranked by |AUROC-0.5|
    cum = np.cumsum(np.sort(np.abs(head_logits).mean(0))[::-1])
    cum /= cum[-1]
    # AUROC of a logreg on the top-k heads' direct logit contributions. Use 5-fold CROSS-VALIDATED
    # out-of-fold scores (not in-sample): an in-sample refit overfits as k grows and spuriously exceeds
    # the detector's own AUROC (it is a *fresh* classifier on k features scored on its training data).
    # The CV version measures generalizable linear *decodability* from those heads — comparable to the
    # detector and consistent with the causal ablation. In-sample kept alongside for transparency.
    from sklearn.model_selection import cross_val_predict

    def _map(scores):  # Convention-A per-generator mAP on the subset
        return per_generator_map(
            pd.DataFrame({"label": labels, "score": np.asarray(scores), "source": sources}),
            real_pairing=pair,
        )

    topk_map, topk_auroc, topk_auroc_insample = {}, {}, {}
    for k in [1, 3, 5, 10, 20]:
        Xk = head_logits[:, order[:k]]
        cv_scores = cross_val_predict(
            LogisticRegression(C=1, max_iter=5000), Xk, labels, cv=5, method="decision_function"
        )
        topk_map[k] = float(_map(cv_scores))  # headline metric (mAP, 5-fold CV)
        topk_auroc[k] = float(roc_auc_score(labels, cv_scores))
        topk_auroc_insample[k] = float(
            roc_auc_score(
                labels, LogisticRegression(C=1, max_iter=5000).fit(Xk, labels).decision_function(Xk)
            )
        )
    detector_map_subset = float(
        _map(true_logit)
    )  # the actual detector's mAP on this subset (baseline)
    detector_auroc_subset = float(roc_auc_score(labels, true_logit))
    print(f"[{ds}] top heads by |AUROC-0.5|:")
    print(hd.head(10).to_string(index=False))
    print(
        f"[{ds}] AUROC from top-k heads' logits: "
        + ", ".join(f"k={k}:{v:.3f}" for k, v in topk_auroc.items())
        + f"  | full-detector~{0.888 if ds == 'synthclic' else 0.962}"
    )
    print(
        f"[{ds}] |contrib| share: top-5 heads={cum[4]:.2f}, top-10={cum[9]:.2f}, top-20={cum[19]:.2f} of 384"
    )
    summary = {
        "dataset": ds,
        "eval": ev,
        "n": int(n),
        "head_share_of_logit": head_share,
        "recon_corr": recon_corr,
        "topk_head_map": topk_map,  # Convention-A per-generator mAP, 5-fold CV (headline)
        "detector_map_subset": detector_map_subset,  # actual detector mAP on this subset (baseline)
        "detector_auroc_subset": detector_auroc_subset,
        "topk_head_auroc": topk_auroc,  # 5-fold CV AUROC (reference)
        "topk_head_auroc_insample": topk_auroc_insample,  # in-sample refit (optimistic; reference)
        "contrib_share_top5": float(cum[4]),
        "contrib_share_top10": float(cum[9]),
        "contrib_share_top20": float(cum[19]),
        "top_heads": hd.head(12)[["layer", "head", "auroc", "mean_synth_minus_real"]].to_dict(
            "records"
        ),
    }
    import json

    (OUT / f"{ds}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT}/{ds}_*")


if __name__ == "__main__":
    main()
