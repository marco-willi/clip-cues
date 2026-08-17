#!/usr/bin/env python
"""Embed the Gandelsman TextSpan descriptions into CLIP's 768-d cross-modal space (image_embeds space),
to match the antonym cue basis. Output: data/embeddings/textspan_embeddings.pt {embeddings,vocabulary}."""

from pathlib import Path

import torch
from transformers import AutoProcessor, CLIPModel

src = Path(
    "/tmp/claude-0/-workspace/68803085-c0a3-4829-addc-0fb21784bb5f/scratchpad/ts_image_descriptions_general.txt"
)
terms = [line.strip() for line in src.read_text().splitlines() if line.strip()]
print("textspan terms:", len(terms))
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = (
    CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", cache_dir="data/hf_cache")
    .to(dev)
    .eval()
)
proc = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14-336", cache_dir="data/hf_cache")
embs = []
with torch.inference_mode():
    for i in range(0, len(terms), 256):
        b = terms[i : i + 256]
        tok = proc(text=b, return_tensors="pt", padding=True, truncation=True).to(dev)
        f = m.get_text_features(**tok)  # transformers 5.x returns an output object, not a tensor
        if not torch.is_tensor(
            f
        ):  # project pooler_output -> 768 (see concept-model-projected-embeddings)
            f = m.text_projection(f.pooler_output)
        embs.append(f.cpu())
E = torch.cat(embs)
E = E / E.norm(dim=1, keepdim=True)
torch.save({"embeddings": E, "vocabulary": terms}, "data/embeddings/textspan_embeddings.pt")
print("WROTE data/embeddings/textspan_embeddings.pt", tuple(E.shape))
