"""Concept Bottleneck Model for Interpretable Synthetic Image Detection.

This module provides a Concept Bottleneck Model that uses CLIP text embeddings
to create an interpretable classifier for synthetic image detection.
"""

import torch
from torch import nn


def bin_concrete_sample(a: torch.Tensor, temperature: float, eps: float = 1e-8) -> torch.Tensor:
    """Sample from the binary concrete distribution.

    Args:
        a: Logits tensor
        temperature: Temperature for concrete distribution (lower = more discrete)
        eps: Small constant for numerical stability

    Returns:
        Samples in range [0, 1]
    """
    U = torch.rand_like(a).clamp(eps, 1.0 - eps)
    L = torch.log(U) - torch.log(1.0 - U)
    X = torch.sigmoid((L + a) / temperature)
    return X


def bernoulli_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Calculate the KL divergence between two Bernoulli distributions.

    Args:
        p: First distribution probabilities
        q: Second distribution probabilities
        eps: Small constant for numerical stability

    Returns:
        KL divergence D_KL(p || q)
    """
    return (p * ((p + eps).log() - (q + eps).log())) + (1.0 - p) * (
        (1.0 - p + eps).log() - (1.0 - q + eps).log()
    )


class ConceptBottleneckModel(nn.Module):
    """Concept Bottleneck Model for interpretable synthetic image detection.

    This model uses CLIP text embeddings to define a vocabulary of concepts,
    then learns to select relevant concepts for distinguishing real from
    synthetic images.

    The model consists of:
    1. Concept selection layer: Learns which concepts are relevant for each image
    2. Classifier layer: Uses selected concepts to predict real/synthetic

    Args:
        text_embeddings: [num_concepts, embedding_dim] CLIP text embeddings for concepts
        tau: Temperature for binary concrete distribution (lower = more sparse selection)
    """

    def __init__(
        self,
        text_embeddings: torch.Tensor,
        tau: float = 0.1,
    ):
        super().__init__()
        text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=1)
        self.num_concepts = text_embeddings.shape[0]
        self.embedding_dim = text_embeddings.shape[1]
        self.register_buffer("text_embeddings", text_embeddings)
        self.tau = tau

        # Concept selection layer
        self.W_concepts = nn.Linear(self.embedding_dim, self.num_concepts)

        # Classifier layer
        self.W_classifier = nn.Linear(self.num_concepts, 1)

        # Initialize weights
        torch.nn.init.zeros_(self.W_concepts.bias)
        torch.nn.init.constant_(self.W_classifier.bias, 0.1)
        torch.nn.init.xavier_normal_(self.W_concepts.weight)
        torch.nn.init.xavier_normal_(self.W_classifier.weight)

    def forward(self, image_embeddings: torch.Tensor) -> dict:
        """Forward pass through the concept bottleneck model.

        Args:
            image_embeddings: [batch_size, embedding_dim] CLIP image embeddings

        Returns:
            Dictionary containing:
            - class_logits: [batch_size, 1] classification logits
            - per_image_concept_samples: [batch_size, num_concepts] selected concepts
            - per_image_concept_logits: [batch_size, num_concepts] selection logits
            - per_concept_logit_contribution: [batch_size, num_concepts] contribution per concept
        """
        # Normalize image features
        image_embeddings = torch.nn.functional.normalize(image_embeddings, dim=1)

        # Calculate image-to-concept similarity
        sim = image_embeddings @ self.text_embeddings.T

        # Concept selection logits
        per_image_concept_logits = self.W_concepts(image_embeddings)

        # Sample concepts (stochastic during training, deterministic during inference)
        if self.training:
            per_image_concept_samples = bin_concrete_sample(
                per_image_concept_logits, temperature=self.tau
            ).clamp(1e-6, 1.0 - 1e-6)
        else:
            per_image_concept_samples = torch.sigmoid(per_image_concept_logits)

        # Mask similarity with selected concepts
        masked_similarity = sim * per_image_concept_samples

        # Classification
        class_logits = self.W_classifier(masked_similarity)

        # Per-concept logit contributions for interpretability
        per_concept_logit_contribution = masked_similarity * self.W_classifier.weight.view(1, -1)

        return {
            "class_logits": class_logits,
            "per_image_concept_samples": per_image_concept_samples,
            "per_image_concept_logits": per_image_concept_logits,
            "per_concept_logit_contribution": per_concept_logit_contribution,
            "image_to_concept_sims_raw": sim,
        }

    def compute_loss(
        self,
        outputs: dict,
        labels: torch.Tensor,
        beta: float = 1e-4,
        alpha: float = 1e-4,
        label_smoothing: float = 0.0,
    ) -> tuple[torch.Tensor, dict]:
        """Compute total loss for training.

        Args:
            outputs: Output dictionary from forward pass
            labels: Ground truth labels (0 = real, 1 = synthetic)
            beta: Weight for KL divergence sparsity loss
            alpha: Target sparsity level for concept selection
            label_smoothing: Label smoothing factor

        Returns:
            Tuple of (total_loss, loss_dict) for logging
        """
        # Apply label smoothing
        smooth_labels = label_smoothing * (1 - labels) + (1 - label_smoothing) * labels

        # Classification loss
        binary_ce = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["class_logits"].view(-1), smooth_labels.view(-1)
        )

        # Sparsity loss (KL divergence)
        per_image_concept_probs = torch.sigmoid(outputs["per_image_concept_logits"])
        kl = (
            bernoulli_kl(
                per_image_concept_probs,
                torch.tensor(alpha, device=per_image_concept_probs.device),
            )
            .sum(1)
            .mean()
        )

        total_loss = binary_ce + beta * kl

        loss_dict = {
            "cross_entropy": binary_ce,
            "kl_divergence": kl,
            "total": total_loss,
        }

        return total_loss, loss_dict
