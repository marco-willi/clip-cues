"""Synthetic Image Detection Models."""

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .classification_head import (
    ActivationOrthogonalityHead,
    LinearHead,
)
from .concept_modeling.networks import ConceptBottleneckModel
from .feature_extractor import CLIPLargePatch14
from .transforms import Transforms

# Available pre-trained checkpoints
AVAILABLE_MODELS = {
    # CLIP Orthogonal models (recommended)
    "clip_orthogonal_synthclic": {
        "path": "data/checkpoints/clip_orthogonal_synthclic.ckpt",
        "type": "orthogonal",
        "training_data": "SynthCLIC",
        "description": "Best for detecting modern diffusion models (SD3, FLUX, Imagen3)",
    },
    "clip_orthogonal_synthbuster": {
        "path": "data/checkpoints/clip_orthogonal_synthbuster.ckpt",
        "type": "orthogonal",
        "training_data": "SynthBuster+",
        "description": "Trained on diverse generators including older models",
    },
    "clip_orthogonal_cnnspot": {
        "path": "data/checkpoints/clip_orthogonal_cnnspot.ckpt",
        "type": "orthogonal",
        "training_data": "CNNSpot",
        "description": "Trained on ProGAN images",
    },
    "clip_orthogonal_combined": {
        "path": "data/checkpoints/clip_orthogonal_combined.ckpt",
        "type": "orthogonal",
        "training_data": "Combined",
        "description": "Trained on all datasets combined",
    },
    # Linear probe models
    "linear_probe_synthclic": {
        "path": "data/checkpoints/linear_probe_synthclic.ckpt",
        "type": "linear",
        "training_data": "SynthCLIC",
        "description": "Simple linear probe baseline",
    },
    "linear_probe_synthbuster": {
        "path": "data/checkpoints/linear_probe_synthbuster.ckpt",
        "type": "linear",
        "training_data": "SynthBuster+",
        "description": "Simple linear probe baseline",
    },
    "linear_probe_cnnspot": {
        "path": "data/checkpoints/linear_probe_cnnspot.ckpt",
        "type": "linear",
        "training_data": "CNNSpot",
        "description": "Simple linear probe baseline",
    },
    "linear_probe_combined": {
        "path": "data/checkpoints/linear_probe_combined.ckpt",
        "type": "linear",
        "training_data": "Combined",
        "description": "Simple linear probe baseline",
    },
    # Concept bottleneck models
    "cm_antonyms_synthclic": {
        "path": "data/checkpoints/cm_antonyms_synthclic.ckpt",
        "type": "concept",
        "training_data": "SynthCLIC",
        "description": "Interpretable concept bottleneck model",
    },
    "cm_antonyms_synthbuster": {
        "path": "data/checkpoints/cm_antonyms_synthbuster.ckpt",
        "type": "concept",
        "training_data": "SynthBuster+",
        "description": "Interpretable concept bottleneck model",
    },
    "cm_antonyms_cnnspot": {
        "path": "data/checkpoints/cm_antonyms_cnnspot.ckpt",
        "type": "concept",
        "training_data": "CNNSpot",
        "description": "Interpretable concept bottleneck model",
    },
    "cm_antonyms_combined": {
        "path": "data/checkpoints/cm_antonyms_combined.ckpt",
        "type": "concept",
        "training_data": "Combined",
        "description": "Interpretable concept bottleneck model",
    },
}


def list_available_models() -> dict:
    """List all available pre-trained checkpoints.

    Returns:
        Dictionary mapping model names to their metadata including:
        - path: Relative path to checkpoint file
        - type: Model type (orthogonal, linear, concept)
        - training_data: Dataset used for training
        - description: Brief description of the model

    Example:
        >>> models = list_available_models()
        >>> for name, info in models.items():
        ...     print(f"{name}: {info['description']}")
    """
    return AVAILABLE_MODELS.copy()


def load_checkpoint(checkpoint_path: str | Path) -> dict:
    """Load a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dictionary with 'state_dict' and 'model_type' keys
    """
    return torch.load(checkpoint_path, map_location="cpu", weights_only=True)


def load_clip_classifier(
    checkpoint_path: str | Path,
    cache_dir: str = "~/.cache/huggingface",
    device: str = "cpu",
) -> "SyntheticImageClassifierInference":
    """Load a pre-trained CLIP-based synthetic image classifier.

    Args:
        checkpoint_path: Path to the checkpoint file (.ckpt)
        cache_dir: Directory to cache the CLIP model weights
        device: Device to load the model on ("cpu" or "cuda")

    Returns:
        SyntheticImageClassifierInference model ready for inference

    Example:
        >>> model = load_clip_classifier("data/checkpoints/clip_orthogonal_combined.ckpt")
        >>> model.eval()
        >>> pred = model(image_tensor)  # Returns probability of being synthetic
    """
    cache_dir = str(Path(cache_dir).expanduser())
    ckpt = load_checkpoint(checkpoint_path)
    model_type = ckpt.get("model_type", "clip")

    # Create feature extractor
    feature_extractor = CLIPLargePatch14(cache_dir=cache_dir)
    feature_extractor.freeze()

    # Create classification head based on model type
    state_dict = ckpt["state_dict"]

    if model_type == "linear":
        # Linear probe model
        head = LinearHead(input_dim=1024, num_classes=1)
        clean_state_dict = {
            k.replace("model.classification_head.", ""): v for k, v in state_dict.items()
        }
    elif model_type == "clip":
        # CLIP orthogonal model - infer layer_dims from checkpoint
        weight_key = "model.classification_head.layers.0.weight"
        layer_dim = state_dict[weight_key].shape[0]
        head = ActivationOrthogonalityHead(input_dim=1024, layer_dims=[layer_dim])
        clean_state_dict = {
            k.replace("model.classification_head.", ""): v for k, v in state_dict.items()
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}. Expected 'clip' or 'linear'.")

    head.load_state_dict(clean_state_dict)

    # Create the classifier
    model = SyntheticImageClassifierInference(
        feature_extractor=feature_extractor.model,
        classification_head=head,
    )
    model.transforms = feature_extractor.transforms
    model.to(device)

    return model


def load_concept_model(
    checkpoint_path: str | Path,
    cache_dir: str = "~/.cache/huggingface",
    device: str = "cpu",
) -> tuple[ConceptBottleneckModel, CLIPLargePatch14]:
    """Load a pre-trained Concept Bottleneck Model.

    Note: Concept models use CLIP text embeddings (768-dim) for concept matching,
    which requires the CLIP text encoder. The returned feature extractor should
    be used to extract 768-dim features from a CLIP model that includes the
    text encoder.

    Args:
        checkpoint_path: Path to the checkpoint file (.ckpt)
        cache_dir: Directory to cache model weights
        device: Device to load the model on ("cpu" or "cuda")

    Returns:
        Tuple of (ConceptBottleneckModel, feature_extractor)

    Example:
        >>> model, extractor = load_concept_model("data/checkpoints/cm_antonyms_combined.ckpt")
        >>> model.eval()
    """
    cache_dir = str(Path(cache_dir).expanduser())
    ckpt = load_checkpoint(checkpoint_path)

    if ckpt.get("model_type") != "concept":
        raise ValueError(f"Expected concept model, got: {ckpt.get('model_type')}")

    # Get text embeddings from checkpoint
    text_embeddings = ckpt["state_dict"]["model.text_embeddings"]

    # Create concept model
    model = ConceptBottleneckModel(text_embeddings=text_embeddings)

    # Load state dict
    clean_state_dict = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(clean_state_dict)
    model.to(device)

    # Create feature extractor for reference
    feature_extractor = CLIPLargePatch14(cache_dir=cache_dir)
    feature_extractor.freeze()

    return model, feature_extractor


class SyntheticImageClassifierInference(nn.Module):
    """Synthetic Image Classifier for Inference Only.

    A simple wrapper that combines a frozen feature extractor with a
    trained classification head for detecting synthetic images.

    Attributes:
        transforms: Image preprocessing transforms (set by load_clip_classifier)
    """

    transforms: callable = None

    def __init__(self, feature_extractor: nn.Module, classification_head: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classification_head = classification_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning probability of being synthetic.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Tensor of shape (batch_size, 1) with values in [0, 1]
            where 0 = real, 1 = synthetic
        """
        x = self.feature_extractor(x)
        if isinstance(x, dict):
            x = x["extracted_features"]
        output = self.classification_head(x)
        logits = output["logits"]
        preds = nn.functional.sigmoid(logits)
        return preds

    def predict(self, image: Union[str, Path, Image.Image]) -> float:
        """Predict probability that an image is synthetic.

        Args:
            image: Path to image file or PIL Image object

        Returns:
            Float probability in [0, 1] where 0=real, 1=synthetic

        Example:
            >>> model = load_clip_classifier("data/checkpoints/clip_orthogonal_synthclic.ckpt")
            >>> prob = model.predict("photo.jpg")
            >>> print(f"Synthetic probability: {prob:.1%}")
        """
        if self.transforms is None:
            raise RuntimeError("Transforms not set. Use load_clip_classifier() to load the model.")

        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Apply transforms
        transforms_wrapper = Transforms(self.transforms)
        inference_transforms = transforms_wrapper.get_inference_transforms()
        batch = inference_transforms({"image": [image]})
        pixel_values = torch.stack(batch["pixel_values"])

        # Get device from model parameters
        device = next(self.parameters()).device
        pixel_values = pixel_values.to(device)

        # Run inference
        with torch.no_grad():
            prob = self(pixel_values)

        return prob.item()

    def predict_batch(
        self, images: list[Union[str, Path, Image.Image]], batch_size: int = 32
    ) -> np.ndarray:
        """Predict probability that images are synthetic (batch processing).

        Args:
            images: List of image paths or PIL Image objects
            batch_size: Number of images to process at once

        Returns:
            NumPy array of probabilities, shape (n_images,)

        Example:
            >>> model = load_clip_classifier("data/checkpoints/clip_orthogonal_synthclic.ckpt")
            >>> probs = model.predict_batch(["img1.jpg", "img2.jpg", "img3.jpg"])
            >>> for path, prob in zip(paths, probs):
            ...     print(f"{path}: {prob:.1%}")
        """
        if self.transforms is None:
            raise RuntimeError("Transforms not set. Use load_clip_classifier() to load the model.")

        # Load all images
        pil_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_images.append(Image.open(img).convert("RGB"))
            else:
                pil_images.append(img)

        # Get device
        device = next(self.parameters()).device

        # Process in batches
        transforms_wrapper = Transforms(self.transforms)
        inference_transforms = transforms_wrapper.get_inference_transforms()

        all_probs = []
        for i in range(0, len(pil_images), batch_size):
            batch_images = pil_images[i : i + batch_size]
            batch = inference_transforms({"image": batch_images})
            pixel_values = torch.stack(batch["pixel_values"]).to(device)

            with torch.no_grad():
                probs = self(pixel_values)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs).flatten()


class SyntheticImageClassifier(nn.Module):
    """Synthetic Image Classifier for Training.

    Combines a frozen feature extractor with a trainable classification head.
    Supports label smoothing and auxiliary losses from the classification head.
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        classification_head: nn.Module,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classification_head = classification_head
        self.label_smoothing = label_smoothing

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass returning logits and optional auxiliary outputs.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Dictionary with 'logits' key and optional auxiliary outputs
        """
        x = self.feature_extractor(x)
        if isinstance(x, dict):
            x = x["extracted_features"]
        self.extracted_features = x.detach().cpu()
        x = self.classification_head(x)
        return x

    def compute_loss(self, logits: torch.Tensor, y_true: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compute total loss including classification and auxiliary losses.

        Args:
            logits: Model output logits
            y_true: Ground truth labels (0 = real, 1 = synthetic)

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains
            individual loss components for logging
        """
        # Apply label smoothing
        y_smoothed = y_true * (1 - self.label_smoothing) + (1 - y_true) * self.label_smoothing

        classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y_smoothed
        )

        # Optional extra loss from the classification head
        extra_loss = self.classification_head.compute_loss()

        # Create a loss dictionary for detailed logging
        loss_dict = {"cross_entropy": classification_loss}
        loss_dict.update(extra_loss)

        total_loss = sum(loss_dict.values())
        return total_loss, loss_dict
