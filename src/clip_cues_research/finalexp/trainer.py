"""The matched training recipe — one definition, used by F1-F4 and F6.

This is the consolidation's core: F1 (1024-d pooler), F3 (768-d projected) and F4 (168-d cue
scores) train the *same* head with the *same* optimizer, loss, schedule and model selection. The
only difference between them is the input space, so any performance difference is attributable to
the representation and not to a protocol difference. F2 swaps the head for the k=8 factorized one
and changes nothing else.

The recipe is the published canonical probe's, verified against
``docs/revision_state/config-audit.md`` §A and byte-compatible with
``scripts/run/run_linear_probe.py`` (whose seed-123 SynthCLIC run is F1's regression anchor):

    Adam(lr=1e-3, weight_decay=0.01)   # coupled L2, NOT AdamW
    BCEWithLogits, label_smoothing 0.1
    batch 64, shuffled; <= 200 epochs
    early stop on val cross-entropy, patience 5, restore the best checkpoint
    frozen cached features, no augmentation, no standardization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from clip_cues.classification_head import ActivationOrthogonalityHead, LinearHead


@dataclass(frozen=True)
class Recipe:
    """The matched recipe. Defaults are the canonical probe's; do not vary them across experiments."""

    lr: float = 1e-3
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    batch_size: int = 64
    epochs: int = 200
    early_stopping_patience: int = 5

    def as_dict(self) -> dict:
        return {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "label_smoothing": self.label_smoothing,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "optimizer": "Adam (coupled L2)",
            "loss": "BCEWithLogits + label smoothing",
            "model_selection": "min val cross-entropy, restore best",
            "features": "frozen cached, no augmentation, no standardization",
        }


RECIPE = Recipe()


@dataclass
class TrainedHead:
    """A trained head plus everything the F-experiments need from it."""

    state_dict: dict
    head_type: str
    input_dim: int
    seed: int
    best_val_ce: float
    best_epoch: int
    epochs_run: int
    history: list[dict] = field(default_factory=list)

    # ── decision geometry ────────────────────────────────────────────────────────────────────
    @property
    def weight(self) -> np.ndarray:
        """The **effective** decision direction in input space, ``w`` with ``z = w.x + b``.

        For ``LinearHead`` this is the single weight row. For the k=8
        ``ActivationOrthogonalityHead`` with ``non_linear=False`` the network is exactly linear, so
        the effective direction is ``w_eff = (w2 @ W1)`` — the object E12 calls ``W0^T w_logit``.
        Distinguishing this from the *individual* factorized axes is the whole point of F2.
        """
        sd = self.state_dict
        if self.head_type == "linear":
            return np.asarray(sd["fc.weight"], dtype=np.float64).ravel()
        w1 = np.asarray(sd["layers.0.weight"], dtype=np.float64)  # (k, d)
        w2 = np.asarray(sd["to_logits.weight"], dtype=np.float64)  # (1, k)
        return (w2 @ w1).ravel()

    @property
    def bias(self) -> float:
        sd = self.state_dict
        if self.head_type == "linear":
            return float(np.asarray(sd["fc.bias"]).ravel()[0])
        w1b = np.asarray(sd["layers.0.bias"], dtype=np.float64)  # (k,)
        w2 = np.asarray(sd["to_logits.weight"], dtype=np.float64)  # (1, k)
        return float((w2 @ w1b).ravel()[0] + np.asarray(sd["to_logits.bias"]).ravel()[0])

    @property
    def axes(self) -> np.ndarray | None:
        """The k individual factorized axes ``(k, d)``, or None for a k=1 linear head."""
        if self.head_type == "linear":
            return None
        return np.asarray(self.state_dict["layers.0.weight"], dtype=np.float64)

    def logits(self, x: np.ndarray) -> np.ndarray:
        """Decision scores for rows of ``x`` — exact, via the effective direction."""
        return np.asarray(x, dtype=np.float64) @ self.weight + self.bias


def make_linear_head(input_dim: int) -> torch.nn.Module:
    return LinearHead(input_dim=input_dim, num_classes=1)


def make_ortho_head(input_dim: int, k: int = 8, lam: float = 0.33) -> torch.nn.Module:
    """The paper's k=8 head: factorized linear + activation-orthogonality penalty, no nonlinearity."""
    return ActivationOrthogonalityHead(
        input_dim=input_dim,
        layer_dims=[k],
        logits_dim=1,
        orthogonal_init=True,
        non_linear=False,
        loss_weight_ortho=lam,
    )


def _val_cross_entropy(logits: torch.Tensor, y: torch.Tensor, label_smoothing: float) -> float:
    """Label-smoothed BCE — the paper's early-stopping metric."""
    p = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    ys = y * (1 - label_smoothing) + (1 - y) * label_smoothing
    return float(-(ys * p.log() + (1 - ys) * (1 - p).log()).mean())


def train_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    head_factory: Callable[[int], torch.nn.Module] | None = None,
    head_type: str = "linear",
    recipe: Recipe = RECIPE,
    device: str = "cpu",
) -> TrainedHead:
    """Train one head under the matched recipe. Deterministic given ``seed``.

    Args:
        x_train / y_train: training features and 0/1 labels.
        x_val / y_val: validation features and labels (early stopping only).
        seed: seeds both initialization and shuffle order.
        head_factory: builds the head from ``input_dim``; defaults to a linear head.
        head_type: ``"linear"`` or ``"ortho_k8"`` — selects how the effective direction is read.
        recipe: the matched recipe (do not vary).
        device: ``"cpu"`` is expected; these heads are tiny.

    Returns:
        A :class:`TrainedHead` restored to its minimum-val-CE checkpoint.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    input_dim = int(x_train.shape[1])

    head = (head_factory or make_linear_head)(input_dim).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=recipe.lr, weight_decay=recipe.weight_decay)
    ls = recipe.label_smoothing

    xt = torch.as_tensor(np.asarray(x_train, dtype=np.float32))
    yt = torch.as_tensor(np.asarray(y_train, dtype=np.float32)).view(-1)
    xv = torch.as_tensor(np.asarray(x_val, dtype=np.float32)).to(dev)
    yv = torch.as_tensor(np.asarray(y_val, dtype=np.float32)).view(-1).to(dev)

    # A seeded generator makes the shuffle order part of the seed, so reruns are bit-identical.
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(xt, yt), batch_size=recipe.batch_size, shuffle=True, generator=gen
    )

    best_ce, best_state, best_epoch, since = float("inf"), None, -1, 0
    history: list[dict] = []
    epochs_run = 0

    for epoch in range(recipe.epochs):
        epochs_run = epoch + 1
        head.train()
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            logits = head(xb)["logits"].view(-1)
            ys = yb * (1 - ls) + (1 - yb) * ls
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, ys)
            # Auxiliary losses (the k=8 head's activation-orthogonality penalty) are part of the
            # head's own contract; a linear head returns {}.
            for extra in head.compute_loss().values():
                loss = loss + extra
            loss.backward()
            opt.step()

        head.eval()
        with torch.no_grad():
            val_ce = _val_cross_entropy(head(xv)["logits"].view(-1), yv, ls)
        history.append({"epoch": epoch, "val_ce": val_ce})

        if val_ce < best_ce:
            best_ce, best_epoch, since = val_ce, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            since += 1
            if since >= recipe.early_stopping_patience:
                break

    if best_state is not None:
        head.load_state_dict(best_state)

    return TrainedHead(
        state_dict={k: v.cpu().numpy() for k, v in head.state_dict().items()},
        head_type=head_type,
        input_dim=input_dim,
        seed=seed,
        best_val_ce=best_ce,
        best_epoch=best_epoch,
        epochs_run=epochs_run,
        history=history,
    )
