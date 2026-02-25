"""Classification Heads for Synthetic Image Detection.

This module provides various classification head architectures that can be
trained on top of frozen CLIP features.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class BaseClassificationHead(nn.Module):
    """Base class for classification heads."""

    def forward(self, x):
        raise NotImplementedError

    def compute_loss(self) -> dict[str, torch.Tensor]:
        """Compute auxiliary losses (e.g., orthogonality, sparsity)."""
        return {}


class LinearHead(BaseClassificationHead):
    """Simple linear classification head.

    A single linear layer that projects features to logits.
    This is the simplest baseline with minimal parameters.
    """

    def __init__(self, input_dim: int, num_classes: int = 1):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        logits = self.fc(x)
        return {"logits": logits}


class ClassificationHead(BaseClassificationHead):
    """Multi-layer classification head with optional orthogonality regularization.

    A configurable MLP that can include:
    - Multiple hidden layers
    - ReLU activations
    - Orthogonal weight initialization
    - Weight orthogonality loss
    """

    def __init__(
        self,
        input_dim: int,
        layer_dims: list[int] = [192, 32],
        logits_dim: int = 1,
        orthogonal_init: bool = True,
        non_linear: bool = True,
        normalize_last_layer: bool = False,
        loss_weight_ortho: float = 0.0,
    ):
        """Initialize the classification head.

        Args:
            input_dim: Input dimension (from feature extractor)
            layer_dims: List of hidden layer dimensions
            logits_dim: Output dimension (1 for binary classification)
            orthogonal_init: Whether to use orthogonal weight initialization
            non_linear: Whether to use ReLU activations
            normalize_last_layer: Whether to L2-normalize the last layer output
            loss_weight_ortho: Weight for orthogonality regularization loss
        """
        super().__init__()
        self.input_dim = input_dim
        self.layer_dims = layer_dims
        self.logits_dim = logits_dim
        self.orthogonal_init = orthogonal_init
        self.non_linear = non_linear
        self.normalize_last_layer = normalize_last_layer
        self.loss_weight_ortho = loss_weight_ortho

        layers = []
        in_dim = input_dim
        for dim in layer_dims:
            layers.append(nn.Linear(in_dim, dim))
            if non_linear:
                layers.append(nn.ReLU())
            in_dim = dim

        self.layers = nn.ModuleList(layers)
        self.distilled_representations = self.layers[-2] if non_linear else self.layers[-1]

        if orthogonal_init:
            torch.nn.init.orthogonal_(self.distilled_representations.weight)

        self.to_logits = nn.Linear(layer_dims[-1], logits_dim)

    def forward(self, x, output_distilled_representations: bool = False):
        for layer in self.layers:
            x = layer(x)

        if self.normalize_last_layer:
            x = F.normalize(x, p=2.0, dim=1, eps=1e-12)

        logits = self.to_logits(x)
        output = {"logits": logits}

        if output_distilled_representations:
            output["distilled_representations"] = x

        return output

    def compute_loss(self) -> dict[str, torch.Tensor]:
        loss = {}
        if self.loss_weight_ortho > 0.0:
            dw = self.distilled_representations.weight
            eye = torch.eye(dw.shape[0], device=dw.device)
            loss_orthogonality = torch.norm(eye - dw @ dw.T)
            loss["orthogonality"] = loss_orthogonality * self.loss_weight_ortho
        return loss


class ActivationOrthogonalityHead(BaseClassificationHead):
    """Classification head with activation orthogonality loss.

    This head enforces orthogonality on the activations (not weights) using
    a Gram matrix-based loss. This promotes diverse, independent features
    in the learned representation.
    """

    def __init__(
        self,
        input_dim: int,
        layer_dims: list[int] = [8],
        logits_dim: int = 1,
        orthogonal_init: bool = True,
        non_linear: bool = False,
        loss_weight_ortho: float = 0.33,
    ):
        """Initialize the activation orthogonality head.

        Args:
            input_dim: Input dimension (from feature extractor)
            layer_dims: List of hidden layer dimensions
            logits_dim: Output dimension (1 for binary classification)
            orthogonal_init: Whether to use orthogonal weight initialization
            non_linear: Whether to use ReLU activations
            loss_weight_ortho: Weight for activation orthogonality loss
        """
        super().__init__()
        self.input_dim = input_dim
        self.layer_dims = layer_dims
        self.logits_dim = logits_dim
        self.orthogonal_init = orthogonal_init
        self.non_linear = non_linear
        self.loss_weight_ortho = loss_weight_ortho

        layers = []
        in_dim = input_dim
        for dim in layer_dims:
            layers.append(nn.Linear(in_dim, dim))
            if non_linear:
                layers.append(nn.ReLU())
            in_dim = dim

        self.layers = nn.ModuleList(layers)
        self.to_logits = nn.Linear(layer_dims[-1], logits_dim)

        _last_layer = self.layers[-2] if non_linear else self.layers[-1]
        self.distilled_representations = torch.empty_like(_last_layer.weight)

        if orthogonal_init:
            torch.nn.init.orthogonal_(_last_layer.weight)

    def forward(self, x, output_distilled_representations: bool = False):
        for layer in self.layers:
            x = layer(x)

        logits = self.to_logits(x)
        output = {"logits": logits}

        self.distilled_representations = x
        self.last_activations = x

        if output_distilled_representations:
            output["distilled_representations"] = x

        return output

    def compute_loss(self) -> dict[str, torch.Tensor]:
        loss = {}
        if self.loss_weight_ortho > 0.0:
            # Normalize activations per feature dimension
            activations_normalized = F.normalize(self.last_activations, dim=0)
            # Compute Gram matrix
            gram_matrix = activations_normalized.T @ activations_normalized
            eye = torch.eye(gram_matrix.shape[0], device=gram_matrix.device)
            loss_orthogonality = torch.norm(eye - gram_matrix)
            loss["orthogonality"] = loss_orthogonality * self.loss_weight_ortho
        return loss


# Dictionary to map head names to their respective classes
HEAD_CLASSES = {
    "LinearHead": LinearHead,
    "ClassificationHead": ClassificationHead,
    "ActivationOrthogonalityHead": ActivationOrthogonalityHead,
}
