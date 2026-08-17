"""Feature Extractors for Synthetic Image Detection.

This module provides CLIP-based feature extractors that convert images
into embeddings suitable for classification.
"""

import types
from typing import Callable

import torch
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)
from transformers import CLIPImageProcessor, CLIPVisionModel


class FeatureExtractor:
    """Base class for feature extractors."""

    def __init__(self, cache_dir: str):
        self.model: torch.nn.Module
        self.transforms: Compose
        self.output_dim: int

    def freeze(self):
        """Freeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False


def modify_forward_to_return_named_output(model, relevant_output_name: str):
    """Modify the model's forward method to return a dictionary with extracted features.

    Args:
        model: The model whose forward method will be modified.
        relevant_output_name: The name of the relevant output in the model's output dictionary.
    """
    original_forward = model.forward

    def new_forward(self, *args, **kwargs):
        outputs = original_forward(*args, **kwargs)

        if not isinstance(outputs, dict):
            raise TypeError("Model outputs must be a dictionary with named outputs.")

        outputs["extracted_features"] = outputs[relevant_output_name]
        return outputs

    model.forward = types.MethodType(new_forward, model)


def modify_forward_to_return_hidden_states(model, layer_id: int = 3):
    """Modify the model's forward method to return hidden states from a specific layer.

    Args:
        model: The model whose forward method will be modified.
        layer_id: Which hidden state to return (can be negative for counting from end)
    """
    original_forward = model.forward

    def new_forward(self, *args, **kwargs):
        kwargs["output_hidden_states"] = True
        outputs = original_forward(*args, **kwargs)

        if not isinstance(outputs, dict):
            raise TypeError("Model outputs must be a dictionary with named outputs.")

        # Extract CLS token from specified layer
        outputs["extracted_features"] = outputs.hidden_states[layer_id][:, 0, :].squeeze(1)
        return outputs

    model.forward = types.MethodType(new_forward, model)


def modify_forward(model, layer_id: str | int = "pooler_output"):
    """Configure model to return named output or hidden state.

    Args:
        model: The model to modify
        layer_id: Either a string key (e.g., "pooler_output") or int layer index
    """
    if isinstance(layer_id, int):
        modify_forward_to_return_hidden_states(model, layer_id)
    elif isinstance(layer_id, str):
        modify_forward_to_return_named_output(model, layer_id)


class CLIPVisionFeatureExtractor(FeatureExtractor):
    """Frozen CLIP vision encoder for an arbitrary HuggingFace CLIP backbone.

    Parameterized by HF model id and output dimension so any CLIP variant (ViT-L/14-336,
    ViT-B/16, ViT-B/32, ...) can be used under the same protocol. The transform and
    forward-hook conventions are identical across variants, so cached embeddings remain
    drop-in compatible with the downstream classification / concept-modeling heads.
    """

    hf_id: str = "openai/clip-vit-large-patch14-336"
    default_output_dim: int = 1024

    def __init__(
        self,
        cache_dir: str,
        layer_id_to_extract: str | int = "pooler_output",
        hf_id: str | None = None,
        output_dim: int | None = None,
    ):
        """Initialize the CLIP feature extractor.

        Args:
            cache_dir: Directory to cache downloaded model weights
            layer_id_to_extract: Which layer to extract features from.
                - "pooler_output": Use the pooled output (default)
                - int: Use hidden state from specified layer index
            hf_id: HuggingFace CLIP vision model id (defaults to the class' ``hf_id``)
            output_dim: Embedding dimensionality (defaults to the class' ``default_output_dim``)
        """
        super().__init__(cache_dir)
        self.layer_id_to_extract = layer_id_to_extract
        self.hf_id = hf_id or self.hf_id
        self.output_dim = output_dim or self.default_output_dim

        self.model = self.load_model(cache_dir)
        self.transforms = self.get_transforms(cache_dir)

    def load_model(self, cache_dir: str) -> torch.nn.Module:
        """Load the CLIP model from HuggingFace."""
        model = CLIPVisionModel.from_pretrained(self.hf_id, cache_dir=cache_dir)
        modify_forward(model, self.layer_id_to_extract)
        return model

    def get_transforms(self, cache_dir: str) -> Callable:
        """Define and return the image transformation pipeline."""
        processor = CLIPImageProcessor.from_pretrained(self.hf_id, cache_dir=cache_dir)
        size = (
            processor.size["shortest_edge"]
            if "shortest_edge" in processor.size
            else (processor.size["height"], processor.size["width"])
        )
        normalize = Normalize(mean=processor.image_mean, std=processor.image_std)
        return Compose([Resize(size), CenterCrop(size), ToTensor(), normalize])


class CLIPLargePatch14(CLIPVisionFeatureExtractor):
    """CLIP ViT-Large/14 @ 336px — the primary feature extractor used in the paper.

    Extracts 1024-dimensional image embeddings. Kept as a named subclass for backward
    compatibility; equivalent to ``CLIPVisionFeatureExtractor`` with its default backbone.
    """

    hf_id = "openai/clip-vit-large-patch14-336"
    default_output_dim = 1024


class CLIPBasePatch16(CLIPVisionFeatureExtractor):
    """CLIP ViT-Base/16 @ 224px — cheaper backbone for cross-variant validation."""

    hf_id = "openai/clip-vit-base-patch16"
    default_output_dim = 768


class CLIPBasePatch32(CLIPVisionFeatureExtractor):
    """CLIP ViT-Base/32 @ 224px — cheaper backbone for cross-variant validation."""

    hf_id = "openai/clip-vit-base-patch32"
    default_output_dim = 768


# Dictionary to map model names to their respective feature extractor classes
EXTRACTOR_CLASSES = {
    "clip_large_patch14": CLIPLargePatch14,
    "clip_base_patch16": CLIPBasePatch16,
    "clip_base_patch32": CLIPBasePatch32,
}
