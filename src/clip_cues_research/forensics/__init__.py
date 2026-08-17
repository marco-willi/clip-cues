"""E1: low-level forensic baselines on SynthCLIC.

Reviewer 3 calls this contrast "essential": benchmark at least one explicit low-level
forensic detector against the semantic, CLIP-based detector. This package provides:

- ``spectral``: a frequency/DCT-statistics classifier (cheapest; no deep net to train).
- ``patch_cnn``: a patch-based CNN baseline (CNNDetection / Wang-et-al. style).

Both train and evaluate on the SynthCLIC train/val/test split used by the CLIP models, so the
numbers are directly comparable. See PLAN.md §3 (E1) for the framing of either outcome.
"""
