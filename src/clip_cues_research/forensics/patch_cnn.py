"""CNNSpot forensic baseline (E1).

Wang et al., "CNN-Generated Images Are Surprisingly Easy to Spot... For Now" (CVPR 2020).
ResNet-50 with a single binary logit, used in two modes:

  Mode A — zero-shot: load the pre-trained ProGAN checkpoint and evaluate on SynthCLIC as-is.
  Mode B — retrained: initialise from ImageNet weights, fine-tune on SynthCLIC train split
            using the CNNSpot augmentation protocol (JPEG + Gaussian blur, each p=0.5).

See PLAN_E1_FORENSIC_BASELINE.md for the full plan.
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
import torchvision.transforms.v2 as Tv2
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── Preprocessing ─────────────────────────────────────────────────────────────

# Must match CNNSpot exactly for Mode A zero-shot results to be valid.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

eval_transform = T.Compose(
    [
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

train_transform = T.Compose(
    [
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


# ── Model ─────────────────────────────────────────────────────────────────────


def build_model(checkpoint_path: str | Path | None = None) -> nn.Module:
    """Return a ResNet-50 with a single binary logit head.

    Args:
        checkpoint_path: Path to a forensic checkpoint. Accepts both the CNNSpot ``.pth`` format
            (flat ``ckpt["model"]`` state dict) and our retrained Lightning ``.ckpt`` (weights under
            ``ckpt["state_dict"]`` with a ``model.`` prefix). If None, weights are ImageNet-pretrained
            (use for Mode B retraining).
    """
    model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(2048, 1)

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model" in ckpt:  # CNNSpot .pth
            state = ckpt["model"]
        elif "state_dict" in ckpt:  # Lightning .ckpt — strip the LightningModule's "model." prefix
            state = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
        else:
            state = ckpt
        model.load_state_dict(state)

    return model


def predict_probs(model: nn.Module, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Run forward pass; return per-image probability of being synthetic (shape [N])."""
    model.eval()
    with torch.inference_mode():
        logits = model(images.to(device)).squeeze(1)  # [N]
    return torch.sigmoid(logits).cpu()


# ── Confidence-image tracker (TP / FP / TN / FN) ─────────────────────────────


class ConfidenceImageTracker:
    """Track highest-confidence TP/FP/TN/FN examples per validation epoch.

    Buckets:
      TP — synthetic, highest model confidence (correctly detected)
      FN — synthetic, lowest model confidence (missed, model thought real)
      TN — real, lowest model confidence (correctly passed as real)
      FP — real, highest model confidence (false alarm, model thought synthetic)
    """

    def __init__(self, top_n: int = 9):
        self.top_n = top_n
        self.reset()

    def update(self, images: torch.Tensor, labels: torch.Tensor, probs: torch.Tensor) -> None:
        for img, label, prob in zip(images.cpu(), labels.cpu(), probs.cpu()):
            label_int = int(label.item())
            prob_val = float(prob.item())
            predicted = prob_val >= 0.5
            if label_int == 1 and predicted:
                self._push(self._tp, img, prob_val, high_is_best=True)
            elif label_int == 1 and not predicted:
                self._push(self._fn, img, prob_val, high_is_best=False)
            elif label_int == 0 and not predicted:
                self._push(self._tn, img, prob_val, high_is_best=False)
            else:
                self._push(self._fp, img, prob_val, high_is_best=True)

    def _push(self, bucket: list, img: torch.Tensor, score: float, *, high_is_best: bool) -> None:
        bucket.append((score, img))
        bucket.sort(key=lambda x: x[0], reverse=high_is_best)
        if len(bucket) > self.top_n:
            bucket.pop()

    def log_to_wandb(self, wandb_run) -> None:
        import torchvision.utils as vutils
        import wandb

        panels = {}
        for key, bucket, caption in [
            ("TP", self._tp, "TP — synthetic, highest-confidence correct"),
            ("FN", self._fn, "FN — synthetic, lowest-confidence (mistaken for real)"),
            ("TN", self._tn, "TN — real, lowest-confidence correct"),
            ("FP", self._fp, "FP — real, highest-confidence (mistaken for synthetic)"),
        ]:
            if bucket:
                grid = vutils.make_grid([img for _, img in bucket], nrow=3, normalize=True)
                panels[f"val/examples/{key}"] = wandb.Image(
                    grid.permute(1, 2, 0).numpy(), caption=caption
                )
        if panels:
            wandb_run.log(panels)

    def reset(self) -> None:
        self._tp: list[tuple[float, torch.Tensor]] = []
        self._fn: list[tuple[float, torch.Tensor]] = []
        self._tn: list[tuple[float, torch.Tensor]] = []
        self._fp: list[tuple[float, torch.Tensor]] = []


# ── Augmentation (CNNSpot protocol, Mode B only) ──────────────────────────────
# JPEG + Gaussian blur, each applied independently with p=0.5.
# Matches the 'Blur+JPEG (0.5)' configuration from Wang et al. Table 1.
augmented_train_transform = T.Compose(
    [
        Tv2.RandomApply([Tv2.JPEG(quality=[30, 100])], p=0.5),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 3.0))], p=0.5),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


# ── Dataset wrapper ───────────────────────────────────────────────────────────


class SynthCLICDataset(Dataset):
    """Thin PyTorch Dataset wrapper around a HuggingFace SynthCLIC split."""

    def __init__(self, hf_split, transform: T.Compose):
        self.split = hf_split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        example = self.split[idx]
        img = example["image"].convert("RGB")
        return self.transform(img), int(example["label"])


# ── Lightning module ───────────────────────────────────────────────────────────


class CNNSpotModule:
    """Lazy import wrapper — Lightning is only needed for Mode B."""

    @staticmethod
    def build(lr: float = 1e-4, lr_patience: int = 3) -> "CNNSpotModule":
        CNNSpotLightningModule = _get_lightning_module_class()
        return CNNSpotLightningModule(lr=lr, lr_patience=lr_patience)


def _get_lightning_module_class():
    """Return the LightningModule class (deferred import to keep Mode A lightning-free)."""
    import lightning as L
    import torchmetrics

    class CNNSpotLightningModule(L.LightningModule):
        def __init__(self, lr: float = 1e-4, lr_patience: int = 3):
            super().__init__()
            self.save_hyperparameters()
            self.model = build_model()  # ImageNet init — Mode B never uses the ProGAN ckpt
            self.criterion = nn.BCEWithLogitsLoss()
            self.val_ap = torchmetrics.AveragePrecision(task="binary")
            self._img_tracker = ConfidenceImageTracker(top_n=9)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x).squeeze(1)

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = self.criterion(logits, y.float())
            self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = self.criterion(logits, y.float())
            probs = torch.sigmoid(logits).detach()
            self.val_ap.update(probs, y)
            self._img_tracker.update(x, y, probs)
            self.log("val/loss", loss, on_epoch=True, prog_bar=True)

        def on_validation_epoch_end(self):
            ap = self.val_ap.compute()
            self.val_ap.reset()
            self.log("val/ap", ap, prog_bar=True)
            try:
                import wandb

                if wandb.run is not None:
                    self._img_tracker.log_to_wandb(wandb.run)
            except ImportError:
                pass
            self._img_tracker.reset()

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=self.hparams.lr_patience
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val/ap"},
            }

    return CNNSpotLightningModule


# ── Evaluation (Step 3 — Mode A zero-shot) ────────────────────────────────────


def evaluate(
    model: nn.Module,
    dataset_split,
    device: torch.device,
    batch_size: int = 64,
    max_samples: int | None = None,
) -> dict:
    """Evaluate on a HuggingFace dataset split; return overall and per-source metrics.

    Per-source AP follows the SynthCLIC protocol from validate_model.py:
    each synthetic source is scored against ALL real samples combined.

    Returns:
        dict with keys: mAP, auroc, accuracy, per_source_ap, predictions, labels, sources
    """
    model = model.to(device)
    model.eval()

    if max_samples is not None:
        dataset_split = dataset_split.shuffle(seed=42).select(
            range(min(max_samples, len(dataset_split)))
        )

    all_probs: list[float] = []
    all_labels: list[int] = []
    all_sources: list[str] = []

    for i in tqdm(range(0, len(dataset_split), batch_size), desc="Evaluating"):
        batch = dataset_split[i : i + batch_size]

        images = batch["image"] if isinstance(batch["image"], list) else [batch["image"]]
        tensors = torch.stack([eval_transform(img.convert("RGB")) for img in images])
        probs = predict_probs(model, tensors, device)
        all_probs.extend(probs.tolist())

        labels = batch["label"] if isinstance(batch["label"], list) else [batch["label"]]
        all_labels.extend(labels)

        sources = batch["source"] if isinstance(batch["source"], list) else [batch["source"]]
        all_sources.extend(sources)

    probs_arr = np.array(all_probs)
    labels_arr = np.array(all_labels)

    overall_ap = float(average_precision_score(labels_arr, probs_arr))
    auroc = float(roc_auc_score(labels_arr, probs_arr))
    accuracy = float(accuracy_score(labels_arr, (probs_arr >= 0.5).astype(int)))

    # Per-source AP: each synthetic source vs ALL real (SynthCLIC protocol)
    source_data: dict[str, dict] = defaultdict(lambda: {"probs": [], "labels": []})
    for prob, label, src in zip(probs_arr, labels_arr, all_sources):
        source_data[src]["probs"].append(float(prob))
        source_data[src]["labels"].append(int(label))

    real_sources = [s for s, d in source_data.items() if all(label == 0 for label in d["labels"])]
    synth_sources = [s for s, d in source_data.items() if all(label == 1 for label in d["labels"])]

    # Per-source AP needs at least one pure-real and one pure-synthetic source. Some datasets (or
    # subsamples) may not satisfy this — fall back to the overall AP and skip the per-source table.
    per_source_ap: dict[str, float] = {}
    if real_sources and synth_sources:
        real_probs = np.concatenate([source_data[s]["probs"] for s in real_sources])
        real_labels_arr = np.zeros(len(real_probs), dtype=int)
        for src in synth_sources:
            combined_probs = np.concatenate([real_probs, source_data[src]["probs"]])
            combined_labels = np.concatenate(
                [real_labels_arr, np.ones(len(source_data[src]["probs"]), dtype=int)]
            )
            per_source_ap[src] = float(average_precision_score(combined_labels, combined_probs))

    mAP = float(np.mean(list(per_source_ap.values()))) if per_source_ap else overall_ap

    return {
        "mAP": mAP,
        "overall_ap": overall_ap,
        "auroc": auroc,
        "accuracy": accuracy,
        "per_source_ap": per_source_ap,
        "predictions": probs_arr,
        "labels": labels_arr,
        "sources": all_sources,
    }


def print_eval_results(results: dict, title: str = "CNNSpot evaluation") -> None:
    """Print a formatted results table matching validate_model.py output style."""
    sep = "=" * 70
    print(f"\n{sep}")
    print(title)
    print(sep)
    print(f"  mAP      : {results['mAP']:.4f}")
    print(f"  AUROC    : {results['auroc']:.4f}")
    print(f"  Accuracy : {results['accuracy']:.4f}")
    print(f"  Overall AP (full split): {results['overall_ap']:.4f}")
    print(f"\n  Per-source AP ({len(results['per_source_ap'])} synthetic sources):")
    print(f"  {'Source':<35} {'AP':>8}")
    print(f"  {'-' * 44}")
    for src, ap in sorted(results["per_source_ap"].items(), key=lambda x: -x[1]):
        print(f"  {src:<35} {ap:8.4f}")
    print(f"  {'-' * 44}")
    print(f"  {'mAP (mean across sources)':<35} {results['mAP']:8.4f}")
    print(sep)


# ── Training (Step 4 — Mode B retrain) ────────────────────────────────────────


def train(
    hf_dataset,
    device: torch.device,
    batch_size: int = 64,
    max_epochs: int = 30,
    lr: float = 1e-4,
    lr_patience: int = 3,
    early_stopping_patience: int = 5,
    checkpoint_dir: Path = Path("data/checkpoints/cnnspot"),
    ckpt_filename: str = "cnnspot_synthclic_retrained",
    max_samples: int | None = None,
    wandb_logger=None,
) -> tuple[nn.Module, Path]:
    """Fine-tune ResNet-50 on SynthCLIC train split (Mode B).

    Uses CNNSpot augmentation protocol (JPEG + Gaussian blur, each p=0.5).
    Early-stops on val AP; saves best checkpoint to checkpoint_dir.

    Args:
        hf_dataset: HuggingFace DatasetDict with 'train' and 'validation' splits.
        device: Target device.
        wandb_logger: Optional Lightning WandbLogger (pass from the script).

    Returns:
        (model, best_checkpoint_path) — model loaded from the best val-AP checkpoint.
    """
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    CNNSpotLightningModule = _get_lightning_module_class()

    # ── Data ──────────────────────────────────────────────────────────────────
    train_split = hf_dataset["train"]
    val_split = hf_dataset["validation"]

    if max_samples is not None:
        train_split = train_split.shuffle(seed=42).select(range(min(max_samples, len(train_split))))
        val_samples = max(16, max_samples // 8)
        val_split = val_split.shuffle(seed=42).select(range(min(val_samples, len(val_split))))

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        SynthCLICDataset(train_split, augmented_train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        SynthCLICDataset(val_split, eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    # ── Module + callbacks ────────────────────────────────────────────────────
    module = CNNSpotLightningModule(lr=lr, lr_patience=lr_patience)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=ckpt_filename,
        monitor="val/ap",
        mode="max",
        save_top_k=1,
        verbose=True,
    )
    callbacks = [
        EarlyStopping(
            monitor="val/ap",
            mode="max",
            patience=early_stopping_patience,
            verbose=True,
        ),
        ckpt_callback,
    ]

    # ── Trainer ───────────────────────────────────────────────────────────────
    accelerator = "gpu" if device.type == "cuda" else "cpu"
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks,
        logger=wandb_logger if wandb_logger is not None else True,
        log_every_n_steps=max(1, len(train_loader) // 5),
        enable_progress_bar=True,
    )

    trainer.fit(module, train_loader, val_loader)

    # ── Load best checkpoint ──────────────────────────────────────────────────
    best_path = Path(ckpt_callback.best_model_path)
    best_module = CNNSpotLightningModule.load_from_checkpoint(best_path)
    return best_module.model, best_path
