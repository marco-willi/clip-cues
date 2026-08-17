"""E4/E5/E7/E8: analysis utilities for the revision.

- ``cross_family``: per-generator breakdown of the 0.37 mAP cross-family failure (E4).
- ``orthogonality``: activation- vs weight-orthogonality empirical check (E5).
- ``interpretation_stability``: stability-of-interpretability metrics for both heads (E8) — direction
  matching/subspace distance, top-K selection overlap, rank correlation, sign agreement, transfer.

These are analysis-only and reuse cached predictions/embeddings — no image-level GPU work.
"""
