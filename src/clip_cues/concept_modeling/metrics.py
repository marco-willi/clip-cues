"""Metrics for concept modeling."""

import torch
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import AUROC, AveragePrecision, BinaryAccuracy


class SimpleMetrics(nn.Module):
    """Simple metrics collection for binary classification.

    Args:
        prefix: Prefix for metric names (e.g., "train" or "val")
    """

    def __init__(self, prefix: str = "train"):
        super().__init__()
        self.prefix = f"{prefix}/"

        self.metrics = MetricCollection(
            {
                "auroc": AUROC(task="binary"),
                "ap": AveragePrecision(task="binary"),
                "accuracy": BinaryAccuracy(),
            }
        )

    @torch.no_grad()
    def update(self, probs: torch.Tensor, targets: torch.Tensor):
        """Update metrics with predictions and targets.

        Args:
            probs: Predicted probabilities [N]
            targets: Ground truth labels [N]
        """
        probs = probs.view(-1).float()
        targets = targets.view(-1).long()
        self.metrics.update(probs, targets)

    @torch.no_grad()
    def compute(self) -> dict[str, torch.Tensor]:
        """Compute all metrics.

        Returns:
            Dictionary of metric names to values
        """
        return self.metrics.compute()

    @torch.no_grad()
    def reset(self):
        """Reset all metrics."""
        self.metrics.reset()

    @torch.no_grad()
    def compute_and_reset(self) -> dict[str, torch.Tensor]:
        """Compute metrics and reset.

        Returns:
            Dictionary of metric names to values
        """
        vals = self.compute()
        self.reset()
        return vals
