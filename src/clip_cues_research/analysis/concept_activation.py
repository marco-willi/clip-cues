"""E2: measure how many concepts the bottleneck model actually uses.

Reviewer 3 asks, for the concept-sparsity sweep, to report not just detection performance
(mAP) but also how many concepts remain *active* as the sparsity weight beta varies. A
``ConceptBottleneckModel`` exposes a per-image gate for every concept; at inference these are
``sigmoid(per_image_concept_logits)`` (the ``per_image_concept_samples`` key of the forward
output). We call a concept "active" for an image when its gate exceeds a threshold (default
0.5) and report the mean count per image. We also report the threshold-free mean gate mass
(sum of gates) as a robustness check.

Pairs with ``clip_cues.concept_modeling.train.train_concept_model`` and is driven by
``scripts/run/run_beta_sweep.py``. Research-only (not back-ported to clip_cues).
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def mean_active_concepts(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cuda",
    threshold: float = 0.5,
) -> dict[str, float]:
    """Mean number of activated concepts per image over a dataset.

    Args:
        model: a trained ``ConceptBottleneckModel``.
        loader: DataLoader yielding ``(image_embeddings, labels, image_ids)`` batches.
        device: device to run inference on.
        threshold: gate value above which a concept counts as active.

    Returns:
        Dict with:
            ``mean_active_concepts``: mean #gates > threshold per image.
            ``mean_gate_mass``: mean sum of gate values per image (threshold-free).
            ``num_concepts``: total concepts in the vocabulary.
            ``threshold``: the threshold used.
    """
    model.eval()
    active_total = 0.0
    mass_total = 0.0
    n_images = 0
    num_concepts = 0

    for batch in loader:
        image_embeddings = batch[0].to(device)
        outputs = model(image_embeddings)
        gates = outputs["per_image_concept_samples"]  # eval -> sigmoid(logits), in [0, 1]
        num_concepts = gates.shape[1]

        active_total += (gates > threshold).sum().item()
        mass_total += gates.sum().item()
        n_images += gates.shape[0]

    if n_images == 0:
        raise ValueError("Loader produced no samples; cannot compute active concepts.")

    return {
        "mean_active_concepts": active_total / n_images,
        "mean_gate_mass": mass_total / n_images,
        "num_concepts": float(num_concepts),
        "threshold": threshold,
    }
