"""Image Transforms for Synthetic Image Detection.

This module provides image transformation pipelines for training and inference.
"""

from typing import Callable

import torch
import torchvision.transforms.v2.functional as TF
from PIL import Image
from torchvision.transforms.v2 import (
    RGB,
    Compose,
    RandomHorizontalFlip,
    RandomResizedCrop,
)


class RandomJPEGCompression:
    """Apply random JPEG compression to simulate real-world image degradation."""

    def __init__(self, quality_range: tuple[int, int] = (65, 100)):
        """
        Args:
            quality_range: A tuple specifying the range of JPEG quality factors (1-100).
        """
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply JPEG compression with randomly chosen quality."""
        random_quality = torch.randint(
            self.quality_range[0], self.quality_range[1] + 1, (1,)
        ).item()
        return TF.jpeg(img, random_quality)


class Transforms:
    """Defines transformations for training and inference.

    Training transforms include data augmentation (random crop, flip, JPEG compression).
    Inference transforms only apply the model's required preprocessing.
    """

    def __init__(self, post_processor: Compose, random_crop_size: int = 512):
        """Initialize transforms.

        Args:
            post_processor: The model-specific preprocessing transforms (resize, normalize)
            random_crop_size: Size for random crop during training
        """
        self._train_transforms = Compose(
            [
                RGB(),
                RandomResizedCrop(size=random_crop_size, scale=(0.5, 1.0)),
                RandomHorizontalFlip(),
                RandomJPEGCompression(quality_range=(65, 100)),
            ]
            + post_processor.transforms
        )

        self._test_transforms = Compose(post_processor.transforms)

    def get_train_transforms(self) -> Callable:
        """Get training transforms with data augmentation."""

        def train_transforms(example_batch):
            example_batch["pixel_values"] = [
                self._train_transforms(pil_img.convert("RGB"))
                for pil_img in example_batch["image"]
            ]
            example_batch["label"] = [float(x) for x in example_batch["label"]]
            example_batch.pop("image", None)
            return example_batch

        return train_transforms

    def get_test_transforms(self) -> Callable:
        """Get test/validation transforms (no augmentation)."""

        def test_transforms(example_batch):
            example_batch["pixel_values"] = [
                self._test_transforms(pil_img.convert("RGB"))
                for pil_img in example_batch["image"]
            ]
            example_batch["label"] = [float(x) for x in example_batch["label"]]
            example_batch.pop("image", None)
            return example_batch

        return test_transforms

    def get_inference_transforms(self) -> Callable:
        """Get inference transforms for single images."""

        def inference_transforms(example_batch):
            example_batch["pixel_values"] = [
                self._test_transforms(pil_img.convert("RGB"))
                for pil_img in example_batch["image"]
            ]
            example_batch.pop("image", None)
            return example_batch

        return inference_transforms
