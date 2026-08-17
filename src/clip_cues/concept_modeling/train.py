"""Training script for Concept Bottleneck Model.

This script trains a concept bottleneck model for interpretable synthetic image detection.
It requires pre-computed CLIP embeddings and concept vocabulary.
"""

import argparse
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from clip_cues.concept_modeling.dataset import CLIPFeatureDataset
from clip_cues.concept_modeling.metrics import SimpleMetrics
from clip_cues.concept_modeling.networks import ConceptBottleneckModel


def prepare_datasets(
    embeddings_path: Path, ds_names: list[str], train_splits: list[str], val_splits: list[str]
):
    """Prepare training and validation datasets.

    Args:
        embeddings_path: Path to image embeddings pickle file
        ds_names: List of dataset names to include
        train_splits: List of splits to use for training
        val_splits: List of splits to use for validation

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    with open(embeddings_path, "rb") as f:
        image_embeddings = pickle.load(f)

    df = image_embeddings["df"].copy()

    # Filter for training data
    idx_train = df["ds_name"].isin(ds_names) & df["split"].isin(train_splits)
    train_clip_features = image_embeddings["embeddings"][idx_train, :]
    train_labels = df.loc[idx_train]["label"].values
    train_image_ids = df.loc[idx_train]["image_id"].values

    # Filter for validation data
    idx_val = df["ds_name"].isin(ds_names) & df["split"].isin(val_splits)
    val_clip_features = image_embeddings["embeddings"][idx_val, :]
    val_labels = df.loc[idx_val]["label"].values
    val_image_ids = df.loc[idx_val]["image_id"].values

    # Create datasets
    train_dataset = CLIPFeatureDataset(
        torch.from_numpy(train_clip_features),
        torch.from_numpy(train_labels).to(torch.float32),
        train_image_ids,
    )

    val_dataset = CLIPFeatureDataset(
        torch.from_numpy(val_clip_features),
        torch.from_numpy(val_labels).to(torch.float32),
        val_image_ids,
    )

    return train_dataset, val_dataset


def train_epoch(model, train_loader, optimizer, device, beta=1e-4, alpha=1e-4, label_smoothing=0.0):
    """Train for one epoch.

    Args:
        model: ConceptBottleneckModel
        train_loader: DataLoader for training data
        optimizer: Optimizer
        device: Device to train on
        beta: Weight for KL divergence loss
        alpha: Target sparsity level
        label_smoothing: BCE label-smoothing factor (paper: 0.1)

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    train_metrics = SimpleMetrics(prefix="train").to(device)

    for batch in tqdm(train_loader, desc="Training"):
        image_embeddings, labels, _ = batch
        image_embeddings = image_embeddings.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(image_embeddings)

        # Compute loss
        loss, loss_dict = model.compute_loss(
            outputs, labels, beta=beta, alpha=alpha, label_smoothing=label_smoothing
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update metrics
        probs = torch.sigmoid(outputs["class_logits"]).view(-1)
        targets = labels.view(-1).long()
        train_metrics.update(probs, targets)

    avg_loss = total_loss / len(train_loader)
    metrics = train_metrics.compute_and_reset()

    return avg_loss, metrics


@torch.no_grad()
def validate(model, val_loader, device, beta=1e-4, alpha=1e-4, sparsity_eps: float = 0.01):
    """Validate the model.

    Args:
        model: ConceptBottleneckModel
        val_loader: DataLoader for validation data
        device: Device to validate on
        beta: Weight for KL divergence loss
        alpha: Target sparsity level
        sparsity_eps: gate threshold for ``concept_sparsity_rel`` (original early-stop eps=0.01).

    Returns:
        Average validation loss and a metrics dict. The metrics include ``concept_sparsity_rel``
        (mean per-sample fraction of active concepts, gate > ``sparsity_eps``) so the composite
        early-stopping objective ``(1 - sparsity_rel) + (1 - auroc)`` can be evaluated.
    """
    model.eval()
    total_loss = 0.0
    val_metrics = SimpleMetrics(prefix="val").to(device)
    active_frac_sum = 0.0
    n_samples = 0

    for batch in tqdm(val_loader, desc="Validation"):
        image_embeddings, labels, _ = batch
        image_embeddings = image_embeddings.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(image_embeddings)

        # Compute loss
        loss, loss_dict = model.compute_loss(outputs, labels, beta=beta, alpha=alpha)

        total_loss += loss.item()

        # Update metrics
        probs = torch.sigmoid(outputs["class_logits"]).view(-1)
        targets = labels.view(-1).long()
        val_metrics.update(probs, targets)

        # concept_sparsity_rel: per-sample proportion of active gates (eval gates = sigmoid(logits)).
        gates = outputs["per_image_concept_samples"]
        active_frac_sum += (gates > sparsity_eps).float().mean(dim=1).sum().item()
        n_samples += gates.shape[0]

    avg_loss = total_loss / len(val_loader)
    metrics = val_metrics.compute_and_reset()
    metrics["concept_sparsity_rel"] = active_frac_sum / max(n_samples, 1)

    return avg_loss, metrics


def train_concept_model(
    image_embeddings_path: Path,
    text_embeddings_path: Path,
    *,
    ds_names: list[str] | None = None,
    train_splits: list[str] | None = None,
    val_splits: list[str] | None = None,
    tau: float = 0.1,
    beta: float = 1e-4,
    alpha: float = 1e-4,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.0,
    batch_size: int = 256,
    epochs: int = 4000,
    device: str = "cuda",
    seed: int = 123,
    output_dir: Path | None = None,
    epoch_callback=None,
    early_stopping_patience: int = 10,
    check_val_every_n_epoch: int = 40,
    sparsity_eps: float = 0.01,
    selection: str = "val_loss",
    verbose: bool = True,
):
    """Train a Concept Bottleneck Model and return the trained model + history.

    This is the reusable entry point behind the CLI ``main()``; it is also called by the
    E2 beta-sensitivity sweep (``scripts/run/run_beta_sweep.py``). Model selection follows the published
    concept-model configs (``cm_chatgptv4_antonyms_{synthclic_v2,synthbuster_v1,cnnspot_v1,
    combined_v1}``): validation runs every ``check_val_every_n_epoch`` epochs (40) and the checkpoint
    minimising **``val/loss`` = BCE + beta*KL** (``selection="val_loss"``) is restored, with early
    stopping after ``early_stopping_patience`` (10) checks without improvement. Alternatives:
    ``selection="composite"`` = ``(1-sparsity_rel)+(1-auroc)`` (the code's old default, NOT used by the
    published configs); ``selection="auroc"`` = best val AUROC.

    Args:
        image_embeddings_path: Path to cached image embeddings (.pkl).
        text_embeddings_path: Path to concept text embeddings (.pt).
        ds_names / train_splits / val_splits: dataset + split selection.
        tau, beta, alpha, lr, batch_size, epochs, seed: hyperparameters.
        device: "cuda" or "cpu" (falls back to CPU if CUDA unavailable).
        output_dir: if given, the best (by val AUROC) checkpoint is written here.
        epoch_callback: optional ``fn(epoch, train_loss, train_metrics, val_loss, val_metrics)``
            called after every epoch (e.g. to log to W&B).
        verbose: print per-epoch progress.

    Returns:
        Dict with keys: ``model``, ``train_loader``, ``val_loader``, ``device``,
        ``concept_names``, ``best_val_auroc``, ``best_val_metrics``, ``history``.
    """
    ds_names = list(ds_names) if ds_names is not None else ["synthclic"]
    train_splits = list(train_splits) if train_splits is not None else ["train"]
    val_splits = list(val_splits) if val_splits is not None else ["validation"]

    torch.manual_seed(seed)

    image_embeddings_path = Path(image_embeddings_path)
    text_embeddings_path = Path(text_embeddings_path)
    if not text_embeddings_path.exists():
        raise FileNotFoundError(f"Text embeddings not found: {text_embeddings_path}")
    if not image_embeddings_path.exists():
        raise FileNotFoundError(f"Image embeddings not found: {image_embeddings_path}")

    # Load text embeddings (concept vocabulary)
    embeddings_dict = torch.load(text_embeddings_path)
    text_features = embeddings_dict["embeddings"]
    concept_names = embeddings_dict["vocabulary"]
    if verbose:
        print(f"Loaded {len(concept_names)} concepts")

    # Prepare datasets / loaders
    train_dataset, val_dataset = prepare_datasets(
        image_embeddings_path, ds_names, train_splits, val_splits
    )
    if verbose:
        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Val dataset: {len(val_dataset)} samples")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model + optimizer
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {torch_device}")
    model = ConceptBottleneckModel(text_features, tau=tau).to(torch_device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.W_concepts.parameters(), "lr": lr * 10, "weight_decay": 0.0},
            {"params": model.W_classifier.parameters(), "lr": lr, "weight_decay": weight_decay},
        ],
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    def _early_stop_score(val_loss: float, m: dict) -> float:
        """Early-stopping objective, lower = better. Default = ``val/loss`` (BCE + beta*KL), which is
        what the published concept-model configs monitor. ``selection="composite"`` =
        ``(1-sparsity_rel)+(1-auroc)``; ``selection="auroc"`` = ``1-auroc``."""
        if selection == "val_loss":
            return float(val_loss)
        auroc = float(m["auroc"])
        if selection == "auroc":
            return 1.0 - auroc
        sparsity_rel = float(m.get("concept_sparsity_rel", 0.0))
        return (1.0 - sparsity_rel) + (1.0 - auroc)

    best_score = float("inf")
    best_val_auroc = 0.0
    best_val_metrics: dict = {}
    best_state: dict | None = None
    history: list[dict] = []
    checks_since_improve = 0

    for epoch in range(epochs):
        if verbose:
            print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss, train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            torch_device,
            beta=beta,
            alpha=alpha,
            label_smoothing=label_smoothing,
        )

        # Validate / check early-stopping only every ``check_val_every_n_epoch`` epochs (paper: 40),
        # plus on the final epoch.
        is_check = ((epoch + 1) % check_val_every_n_epoch == 0) or (epoch + 1 == epochs)
        if not is_check:
            history.append({"epoch": epoch, "train_loss": train_loss})
            continue

        val_loss, val_metrics = validate(
            model, val_loader, torch_device, beta=beta, alpha=alpha, sparsity_eps=sparsity_eps
        )
        score = _early_stop_score(val_loss, val_metrics)

        if verbose:
            print(
                f"Train Loss: {train_loss:.4f}, AUROC: {train_metrics['auroc']:.4f}, "
                f"AP: {train_metrics['ap']:.4f}"
            )
            print(
                f"Val Loss: {val_loss:.4f}, AUROC: {val_metrics['auroc']:.4f}, "
                f"AP: {val_metrics['ap']:.4f}, sparsity_rel: "
                f"{float(val_metrics['concept_sparsity_rel']):.4f}, early_stop: {score:.4f}"
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_auroc": float(train_metrics["auroc"]),
                "val_auroc": float(val_metrics["auroc"]),
                "val_ap": float(val_metrics["ap"]),
                "val_concept_sparsity_rel": float(val_metrics["concept_sparsity_rel"]),
                "val_early_stop": score,
            }
        )
        if epoch_callback is not None:
            epoch_callback(epoch, train_loss, train_metrics, val_loss, val_metrics)

        if score < best_score:
            best_score = score
            best_val_auroc = float(val_metrics["auroc"])
            best_val_metrics = {k: float(v) for k, v in val_metrics.items()}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            checks_since_improve = 0
            if output_dir is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_auroc": best_val_auroc,
                        "val_early_stop": best_score,
                        "concept_names": concept_names,
                    },
                    output_dir / "best_model.pt",
                )
                if verbose:
                    print(f"Saved best model (early_stop={best_score:.4f})")
        else:
            checks_since_improve += 1
            if early_stopping_patience > 0 and checks_since_improve >= early_stopping_patience:
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch + 1} (patience {early_stopping_patience})"
                    )
                break

    # Restore the best (min composite) checkpoint into the returned model.
    if best_state is not None:
        model.load_state_dict({k: v.to(torch_device) for k, v in best_state.items()})

    if verbose:
        print("\nTraining complete!")
        print(f"Best val AUROC: {best_val_auroc:.4f} (early_stop={best_score:.4f})")

    return {
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "device": torch_device,
        "concept_names": concept_names,
        "best_val_auroc": best_val_auroc,
        "best_val_metrics": best_val_metrics,
        "best_early_stop": best_score,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Concept Bottleneck Model")

    # Paths
    parser.add_argument(
        "--text-embeddings-path",
        type=Path,
        required=True,
        help="Path to text embeddings file (.pt) containing concept vocabulary",
    )
    parser.add_argument(
        "--image-embeddings-path",
        type=Path,
        required=True,
        help="Path to image embeddings pickle file (.pkl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/concept_model"),
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--ds-names",
        nargs="+",
        default=["synthclic"],
        help="Dataset names to use",
    )
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=["train"],
        help="Splits to use for training",
    )
    parser.add_argument(
        "--val-splits",
        nargs="+",
        default=["validation"],
        help="Splits to use for validation",
    )

    # Model hyperparameters
    parser.add_argument(
        "--tau",
        type=float,
        default=0.1,
        help="Temperature for concrete distribution",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1e-4,
        help="Weight for KL divergence loss",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-4,
        help="Target sparsity level",
    )

    # Training
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="W_classifier weight decay"
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.0, help="BCE label smoothing (concept cfg: 0.0)"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--epochs", type=int, default=4000, help="Max epochs (early stopping caps it)"
    )
    parser.add_argument(
        "--early-stopping-patience", type=int, default=10, help="Checks without improvement to stop"
    )
    parser.add_argument(
        "--check-val-every-n-epoch",
        type=int,
        default=40,
        help="Validate/check this often (paper:40)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=123, help="Random seed (paper: 123)")

    args = parser.parse_args()

    print(f"Loading text embeddings from: {args.text_embeddings_path}")
    print(f"Loading image embeddings from: {args.image_embeddings_path}")
    print("\nStarting training...")

    train_concept_model(
        image_embeddings_path=args.image_embeddings_path,
        text_embeddings_path=args.text_embeddings_path,
        ds_names=args.ds_names,
        train_splits=args.train_splits,
        val_splits=args.val_splits,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
