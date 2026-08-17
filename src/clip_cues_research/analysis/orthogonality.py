"""E5: activation-orthogonality vs explicit weight-orthogonality.

Reviewer 3 asks why we penalize batch-dependent *activation* alignment rather than the
*weights* directly via ``||I - W_L1^T W_L1||_F^2``, and what the trade-off is. The core answer
is mathematical (goes in the paper), but a small empirical check strengthens it: train the
orthogonal head with an explicit weight-orthogonality penalty and compare mAP and measured
orthogonality against the current activation-alignment loss. Embeddings-only, trivial compute.

TODO(E5):
    - weight_orthogonality_loss(W) = ||I - W^T W||_F^2
    - train head variant {activation-ortho, weight-ortho, none} on cached SynthCLIC embeddings
    - report mAP + an orthogonality measure (e.g. off-diagonal Gram mass) for each
"""

from __future__ import annotations

import torch


def weight_orthogonality_loss(weight: torch.Tensor) -> torch.Tensor:
    """Explicit weight-orthogonality penalty ``||I - W^T W||_F^2``.

    Args:
        weight: layer weight matrix of shape (out_features, in_features).
    """
    gram = weight @ weight.t()
    identity = torch.eye(gram.shape[0], device=weight.device, dtype=weight.dtype)
    return torch.linalg.norm(identity - gram, ord="fro") ** 2


def orthogonality_score(weight: torch.Tensor) -> float:
    """Off-diagonal Gram mass as a scalar (lower = more orthogonal)."""
    gram = weight @ weight.t()
    off_diag = gram - torch.diag(torch.diag(gram))
    return float(torch.linalg.norm(off_diag, ord="fro"))
