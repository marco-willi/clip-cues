"""CLIP-Cues: Synthetic Image Detection with CLIP."""

__version__ = "1.0.0"

from .classification_head import (
    HEAD_CLASSES,
    ActivationOrthogonalityHead,
    ClassificationHead,
    LinearHead,
)
from .feature_extractor import EXTRACTOR_CLASSES, CLIPLargePatch14
from .model import (
    AVAILABLE_MODELS,
    SyntheticImageClassifier,
    SyntheticImageClassifierInference,
    list_available_models,
    load_checkpoint,
    load_clip_classifier,
    load_concept_model,
)
from .visualization import plot_collage

__all__ = [
    # Model classes
    "SyntheticImageClassifier",
    "SyntheticImageClassifierInference",
    # Loading functions
    "load_checkpoint",
    "load_clip_classifier",
    "load_concept_model",
    "list_available_models",
    "AVAILABLE_MODELS",
    # Feature extractors
    "CLIPLargePatch14",
    "EXTRACTOR_CLASSES",
    # Classification heads
    "LinearHead",
    "ClassificationHead",
    "ActivationOrthogonalityHead",
    "HEAD_CLASSES",
    # Visualization
    "plot_collage",
]
