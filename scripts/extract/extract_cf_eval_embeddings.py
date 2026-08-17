#!/usr/bin/env python
"""T0/E7: extract the FULL CommunityForensics-Eval L/14 pooler embeddings (1024-d) for local head inference.

E7 scored images through whole detectors (``predict_batch``) and never cached embeddings. To compute a
canonical *linear-probe* (k=1/logistic) CF-Eval number with the EXACT E7 protocol — and to keep CF-Eval
embeddings reusable for any future head — we extract ``pooler_output`` once on a GPU box and cache it.
Inference (linear probe / orthogonal head) is then done locally on these embeddings; applying the
orthogonal head to them must reproduce the persisted E7 ortho scores per-image
(validation: ``scripts/utils/eval_heads_on_cf_embeddings.py``).

The extractor + transforms are the SAME ``CLIPLargePatch14`` used inside ``load_clip_classifier``, so
``head(pooler_emb)`` is identical by construction to ``predict_batch(image)``.

Run (Lambda A10):
    python scripts/extract/extract_cf_eval_embeddings.py --device cuda --cache-dir data/hf_cache
Output:
    data/embeddings/communityforensics_l14_eval.pkl  -> {"df": <meta>, "embeddings": float32[N,1024]}
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clip_cues.feature_extractor import EXTRACTOR_CLASSES
from clip_cues_research.datasets import CF_EVAL, load_community_forensics

# Same metadata columns E7 persists (scripts/run/run_community_eval.py -> community_eval._META), so the
# downstream export pairing (per-source == per-generator) is reproduced byte-for-byte.
META = ["image_id", "label", "source", "architecture", "real_source", "subset"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="CompEval")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--cache-dir", default="data/hf_cache")
    ap.add_argument(
        "--out", type=Path, default=Path("data/embeddings/communityforensics_l14_eval.pkl")
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-samples", type=int, default=None, help="cap (smoke tests only)")
    a = ap.parse_args()

    ext = EXTRACTOR_CLASSES["clip_large_patch14"](a.cache_dir, layer_id_to_extract="pooler_output")
    ext.freeze()
    ext.model.to(a.device).eval()
    tf = ext.transforms

    ds = load_community_forensics(
        CF_EVAL, a.split, cache_dir=a.cache_dir
    )  # map-style (lazy decode)
    n = len(ds) if a.max_samples is None else min(a.max_samples, len(ds))
    print(f"CF-Eval {a.split}: extracting {n} samples on {a.device}", flush=True)

    embs: list[np.ndarray] = []
    metas: dict[str, list] = {k: [] for k in META}
    for s in range(0, n, a.batch_size):
        b = ds[s : min(s + a.batch_size, n)]
        imgs = [im if im.mode == "RGB" else im.convert("RGB") for im in b["image"]]
        t = torch.stack([tf(im) for im in imgs]).to(a.device)
        with torch.inference_mode():
            f = ext.model(t)["extracted_features"].float().cpu().numpy().astype(np.float32)
        embs.append(f)
        for k in META:
            metas[k].extend(b[k])
        if (s // a.batch_size) % 50 == 0:
            print(f"  {min(s + a.batch_size, n)}/{n}", flush=True)

    emb = np.concatenate(embs, 0)
    df = pd.DataFrame(metas)
    assert len(df) == emb.shape[0] == n, (len(df), emb.shape, n)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as fh:
        pickle.dump({"df": df, "embeddings": emb}, fh)
    print(
        f"WROTE {a.out} emb={emb.shape} "
        f"labels={df.label.value_counts().to_dict()} "
        f"n_source={df.source.nunique()} n_arch={df.architecture.nunique()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
