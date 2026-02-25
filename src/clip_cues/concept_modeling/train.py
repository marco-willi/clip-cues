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


def train_epoch(model, train_loader, optimizer, device, beta=1e-4, alpha=1e-4):
    """Train for one epoch.

    Args:
        model: ConceptBottleneckModel
        train_loader: DataLoader for training data
        optimizer: Optimizer
        device: Device to train on
        beta: Weight for KL divergence loss
        alpha: Target sparsity level

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
        loss, loss_dict = model.compute_loss(outputs, labels, beta=beta, alpha=alpha)

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
def validate(model, val_loader, device, beta=1e-4, alpha=1e-4):
    """Validate the model.

    Args:
        model: ConceptBottleneckModel
        val_loader: DataLoader for validation data
        device: Device to validate on
        beta: Weight for KL divergence loss
        alpha: Target sparsity level

    Returns:
        Average validation loss and metrics
    """
    model.eval()
    total_loss = 0.0
    val_metrics = SimpleMetrics(prefix="val").to(device)

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

    avg_loss = total_loss / len(val_loader)
    metrics = val_metrics.compute_and_reset()

    return avg_loss, metrics


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
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load text embeddings (concept vocabulary)
    print(f"Loading text embeddings from: {args.text_embeddings_path}")
    if not args.text_embeddings_path.exists():
        raise FileNotFoundError(f"Text embeddings not found: {args.text_embeddings_path}")

    embeddings_dict = torch.load(args.text_embeddings_path)
    text_features = embeddings_dict["embeddings"]
    concepts_list = embeddings_dict["vocabulary"]
    print(f"Loaded {len(concepts_list)} concepts")

    # Prepare datasets
    print(f"Loading image embeddings from: {args.image_embeddings_path}")
    if not args.image_embeddings_path.exists():
        raise FileNotFoundError(f"Image embeddings not found: {args.image_embeddings_path}")

    train_dataset, val_dataset = prepare_datasets(
        args.image_embeddings_path,
        args.ds_names,
        args.train_splits,
        args.val_splits,
    )
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ConceptBottleneckModel(text_features, tau=args.tau).to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        [
            {"params": model.W_concepts.parameters(), "lr": args.lr * 10, "weight_decay": 0.0},
            {"params": model.W_classifier.parameters(), "lr": args.lr, "weight_decay": 1e-5},
        ],
    )

    # Training loop
    best_val_auroc = 0.0
    print("\nStarting training...")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, device, beta=args.beta, alpha=args.alpha
        )

        # Validate
        val_loss, val_metrics = validate(
            model, val_loader, device, beta=args.beta, alpha=args.alpha
        )

        # Print metrics
        print(
            f"Train Loss: {train_loss:.4f}, AUROC: {train_metrics['auroc']:.4f}, AP: {train_metrics['ap']:.4f}"
        )
        print(
            f"Val Loss: {val_loss:.4f}, AUROC: {val_metrics['auroc']:.4f}, AP: {val_metrics['ap']:.4f}"
        )

        # Save best model
        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            checkpoint_path = args.output_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auroc": best_val_auroc,
                    "concept_names": concepts_list,
                },
                checkpoint_path,
            )
            print(f"Saved best model to: {checkpoint_path}")

    print("\nTraining complete!")
    print(f"Best validation AUROC: {best_val_auroc:.4f}")


if __name__ == "__main__":
    main()
