"""E6: CommunityForensics official detector (out-of-the-box external baseline), revision-only.

The released checkpoints (`OwensLab/commfor-model-{384,224}`) are **not** transformers models — they
are `PyTorchModelHubMixin` weights of a plain timm ViT (params prefixed ``vit.``, a single binary
logit head). We therefore rebuild the architecture with timm and load the `vit.`-stripped state_dict
(verified: zero missing/unexpected keys). Preprocessing **must match the official eval transform**
(Community-Forensics ``dataloader.get_transform(mode="test")``): Resize(shortest→440 for the 384
model / 256 for 224) → CenterCrop(input_size) → ToTensor([0,1]) → Normalize(**ImageNet** mean/std),
default (bilinear) interpolation. (Using timm's default — 0.5/0.5 norm, bicubic, crop_pct 1.0 —
silently degrades CF-Eval mAP; this was a bug.)

The wrapper exposes ``predict_batch(images) -> probs`` (P(fake), higher = synthetic) — the same
contract as ``clip_cues``'s CLIP classifier and ``patch_cnn`` — so it drops straight into the E7
scoring spine (``clip_cues_research.community_eval.score_cf_split``).
"""

from __future__ import annotations

import json

import numpy as np
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

DEFAULT_REPO = "OwensLab/commfor-model-384"

# Official Community-Forensics eval preprocessing (dataloader.get_transform, mode="test"):
# resize shortest side to RESIZE_FOR[input_size], center-crop to input_size, ImageNet normalize.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_RESIZE_FOR = {224: 256, 384: 440}


def official_eval_transform(input_size: int) -> transforms.Compose:
    """Replicates Community-Forensics' test-mode transform (torchvision, default bilinear resize)."""
    resize_size = _RESIZE_FOR.get(input_size, round(input_size * 256 / 224))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def _timm_name(model_size: str, patch_size: int, input_size: int) -> str:
    """Map the CommunityForensics config to a timm model id, e.g. vit_small_patch16_384."""
    return f"vit_{model_size}_patch{patch_size}_{input_size}"


class CommunityForensicsDetector:
    """Out-of-the-box CommunityForensics ViT detector. ``predict_batch(list[PIL]) -> P(fake)``."""

    def __init__(
        self,
        model_repo: str = DEFAULT_REPO,
        device: str | None = None,
        batch_size: int = 32,
        cache_dir: str = "data/hf_cache",
    ):
        self.model_repo = model_repo
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        cfg = json.load(open(hf_hub_download(model_repo, "config.json", cache_dir=cache_dir)))
        self.input_size = int(cfg["input_size"])
        timm_name = _timm_name(cfg["model_size"], int(cfg["patch_size"]), self.input_size)

        self.model = timm.create_model(
            timm_name, num_classes=1, img_size=self.input_size, pretrained=False
        )
        state = load_file(hf_hub_download(model_repo, "model.safetensors", cache_dir=cache_dir))
        stripped = {k[len("vit.") :]: v for k, v in state.items() if k.startswith("vit.")}
        missing, unexpected = self.model.load_state_dict(
            stripped, strict=True
        )  # raises on mismatch
        self.model.to(self.device).eval()

        # Official Community-Forensics eval preprocessing (NOT timm's default — see module docstring).
        self.transform = official_eval_transform(self.input_size)

    @torch.no_grad()
    def predict_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Probability of synthetic/fake per image (higher = fake), shape (len(images),)."""
        probs: list[np.ndarray] = []
        for i in range(0, len(images), self.batch_size):
            chunk = images[i : i + self.batch_size]
            x = torch.stack([self.transform(im.convert("RGB")) for im in chunk]).to(self.device)
            logits = self.model(x).reshape(-1)  # (n,) single binary logit
            probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probs) if probs else np.empty(0)
