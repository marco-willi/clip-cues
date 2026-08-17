#!/usr/bin/env python
"""E9: embed a candidate-vocabulary CSV into CLIP's 768-d cross-modal space.

Accepts two CSV shapes (auto-detected):
  bipolar  — columns ``attribute_name, positive_prompt, negative_prompt`` (the antonyms.csv
             format): each row becomes a unit DIFF direction normalize(pos) - normalize(neg),
             matching data/embeddings/antonyms_diff_embeddings.pt. With --poles, the raw
             normalized poles are emitted instead (two rows per attribute, _positive/_negative).
  unipolar — columns ``name, prompt``: each row becomes the unit text embedding of ``prompt``.

Output: a ``{embeddings, vocabulary}`` .pt (the repo's standard vocab format). HF caching per
the repo rule: cache_dir="data/hf_cache".

Usage:
    uv run python scripts/interpret/embed_vocab.py --csv data/vocabularies/antonyms.csv \
        --out data/embeddings/vocab_pool/antonyms_rebuilt.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoProcessor, CLIPModel

MODEL = "openai/clip-vit-large-patch14-336"


def embed_texts(texts: list[str], device: str | None = None) -> torch.Tensor:
    """Unit-normalized 768-d CLIP text embeddings (same recipe as embed_textspan.py)."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = CLIPModel.from_pretrained(MODEL, cache_dir="data/hf_cache").to(dev).eval()
    proc = AutoProcessor.from_pretrained(MODEL, cache_dir="data/hf_cache")
    embs = []
    with torch.inference_mode():
        for i in range(0, len(texts), 256):
            tok = proc(
                text=texts[i : i + 256], return_tensors="pt", padding=True, truncation=True
            ).to(dev)
            f = m.get_text_features(**tok)
            if not torch.is_tensor(f):
                # transformers 5.x: the output object's pooler_output ALREADY IS the projected
                # text feature (== text_projection(text_model pooled)); do NOT project again.
                # The old recipe applied text_projection once more (double projection) — every
                # vocabulary .pt embedded before 2026-07-17 lives in that W^2 space (verified:
                # stored antonym poles cos 1.000 to the double-projected recipe, cos 0.03 to
                # canonical). See docs/revision_state/INTERPRETATION.md.
                f = f.pooler_output
            embs.append(f.cpu().float())  # float32: the CBM registers this buffer as-is
    E = torch.cat(embs)
    return E / E.norm(dim=1, keepdim=True)


def embed_csv(csv: str | Path, poles: bool = False) -> dict:
    """CSV -> {embeddings, vocabulary} dict (see module docstring for accepted shapes)."""
    df = pd.read_csv(csv)
    if {"positive_prompt", "negative_prompt"} <= set(df.columns):
        name_col = "attribute_name" if "attribute_name" in df.columns else "name"
        pos = embed_texts(df["positive_prompt"].tolist())
        neg = embed_texts(df["negative_prompt"].tolist())
        if poles:
            E = torch.cat([pos, neg])
            names = [f"{n}_positive" for n in df[name_col]] + [
                f"{n}_negative" for n in df[name_col]
            ]
        else:
            E = pos - neg
            E = E / E.norm(dim=1, keepdim=True)
            names = df[name_col].tolist()
    elif "prompt" in df.columns:
        E = embed_texts(df["prompt"].tolist())
        names = df["name"].tolist()
    else:
        raise ValueError(f"{csv}: need positive_prompt/negative_prompt or prompt columns")
    return {"embeddings": E, "vocabulary": names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True, help=".pt output path")
    ap.add_argument("--poles", action="store_true", help="emit raw poles instead of diffs")
    a = ap.parse_args()
    d = embed_csv(a.csv, poles=a.poles)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(d, a.out)
    print(f"wrote {a.out}  {tuple(d['embeddings'].shape)}  ({len(d['vocabulary'])} terms)")


if __name__ == "__main__":
    main()
