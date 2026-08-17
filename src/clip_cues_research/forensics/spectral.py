"""Frequency / DCT-statistics forensic baseline (E1).

A deliberately simple, defensible low-level detector: extract spectral features (radially
averaged FFT power spectrum and/or block-DCT coefficient statistics) and fit a linear / shallow
classifier. No deep network to train, so it is the cheapest "fingerprint-driven" contrast to
CLIP's semantic cues that Reviewer 3 requests.

TODO(E1):
    - implement ``spectral_features(image) -> np.ndarray`` (radial FFT profile + DCT stats)
    - fit LogisticRegression / small MLP on SynthCLIC train split
    - evaluate (mAP, AUROC) on the SynthCLIC test split, same protocol as the CLIP models
    - log to W&B project ``clip-cues``
"""

from __future__ import annotations

import numpy as np


def spectral_features(image: np.ndarray) -> np.ndarray:
    """Compute low-level spectral features for a single RGB image.

    Returns a fixed-length feature vector (radially averaged power spectrum + DCT statistics).
    """
    raise NotImplementedError("E1: implement spectral feature extraction")


def fit(features: np.ndarray, labels: np.ndarray):
    """Fit the spectral classifier (LogisticRegression or shallow MLP)."""
    raise NotImplementedError("E1: implement classifier fit")


def evaluate(model, features: np.ndarray, labels: np.ndarray) -> dict:
    """Return {'mAP': ..., 'auroc': ...} on the held-out split."""
    raise NotImplementedError("E1: implement evaluation")
