"""
CIFAR-10 Training Pipeline for Apple Silicon (MPS).

This script provides a complete training pipeline with:
- Data loading from HuggingFace datasets
- CNN and ViT model support
- MPS/CPU device detection
- Training with early stopping
- Metrics and confusion matrix visualization

Usage:
    python train.py --model cnn --epochs 10
    python train.py --model vit --img-size 128
    python train.py --debug --epochs 3  # Quick test with 200 samples
"""

import argparse
import random
import shutil
import tempfile
from pathlib import Path
from typing import Tuple, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from datasets import load_dataset
from models import SimpleCNN, create_vit_model


# CIFAR-10 class names for visualization
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Same seed = same random numbers = same results every run.
    Sets seeds for: Python random, NumPy, PyTorch (CPU and MPS).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # MPS uses the same seed as CPU via torch.manual_seed


def get_device(use_mps: bool = True) -> torch.device:
    """
    Detect best available device: CUDA (Nvidia) > MPS (Apple) > CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA (Nvidia GPU - Colab Mode)")
    elif use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")
    return device

def create_dataloaders(
    train_ds, val_ds, test_ds,
    img_size: int = 64,
    batch_size: int = 64,
    num_workers: Optional[int] = None, # שינוי ל-Optional
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    if num_workers is None:
        num_workers = 2 if torch.cuda.is_available() else 0

# ============================================================================
# Data Loading
# ============================================================================

def get_transforms(img_size: int = 64, is_train: bool = True) -> transforms.Compose:
    """Create image preprocessing pipeline with optional augmentation."""
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 channel means
        std=[0.2470, 0.2435, 0.2616]    # CIFAR-10 channel stds
    )

    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])


class CIFAR10Dataset(Dataset):
    """Bridges HuggingFace dataset → PyTorch DataLoader format."""

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.dataset[idx]
        image = item["img"]
        label = item["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


def load_cifar10(
    cache_dir: Optional[Path] = None,
    num_samples: Optional[int] = None,
    val_split: float = 0.1,
) -> Tuple:
    """
    Load CIFAR-10 from HuggingFace Hub with proper train/val/test split.

    Args:
        cache_dir: Where to cache data
        num_samples: Limit samples for debugging
        val_split: Fraction of training data for validation (default 10%)

    Returns:
        (train_ds, val_ds, test_ds, cache_dir)
    """
    if cache_dir is None:
        cache_dir = Path(tempfile.mkdtemp(prefix="cifar10_cache_"))
        print(f"Cache directory: {cache_dir}")

    print("Loading CIFAR-10 from HuggingFace...")
    dataset = load_dataset(
        "cifar10",
        cache_dir=str(cache_dir),
    )

    full_train_ds = dataset["train"]  # 50,000 samples
    test_ds = dataset["test"]          # 10,000 samples (untouched!)

    # Debug mode: limit samples
    if num_samples is not None:
        print(f"Debug mode: limiting to {num_samples} samples")
        full_train_ds = full_train_ds.select(range(min(num_samples, len(full_train_ds))))
        test_ds = test_ds.select(range(min(num_samples // 5, len(test_ds))))

    # Split training data into train and validation
    total_train = len(full_train_ds)
    val_size = int(total_train * val_split)
    train_size = total_train - val_size

    # Create index ranges for split
    train_ds = full_train_ds.select(range(train_size))
    val_ds = full_train_ds.select(range(train_size, total_train))

    print(f"Data split:")
    print(f"  Train: {len(train_ds)} samples (for learning)")
    print(f"  Val:   {len(val_ds)} samples (for early stopping & model selection)")
    print(f"  Test:  {len(test_ds)} samples (for final evaluation only)")

    return train_ds, val_ds, test_ds, cache_dir


def create_dataloaders(
    train_ds,
    val_ds,
    test_ds,
    img_size: int = 64,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wrap datasets in DataLoaders for batched iteration.

    Returns (train_loader, val_loader, test_loader).
    """
    train_transform = get_transforms(img_size, is_train=True)
    eval_transform = get_transforms(img_size, is_train=False)  # No augmentation

    train_dataset = CIFAR10Dataset(train_ds, transform=train_transform)
    val_dataset = CIFAR10Dataset(val_ds, transform=eval_transform)
    test_dataset = CIFAR10Dataset(test_ds, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    print(f"DataLoaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader)} batches")
    print(f"  Test:  {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


# ============================================================================
# Checkpointing
# ============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    accuracy: float,
    path: Path,
) -> None:
    """Save model checkpoint to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "accuracy": accuracy,
    }, path)
    print(f"Checkpoint saved: {path} (acc: {accuracy:.2f}%)")


# ============================================================================
# Training Loop
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """
    Train for one epoch. Returns (average_loss, accuracy).
    """
    model.train()  # Training mode: dropout active, batch norm uses batch stats

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for images, labels in pbar:
        # Move to device (MPS/CPU)
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()          # Clear old gradients
        outputs = model(images)        # (B, 10) predictions
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()                # Compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()               # Update weights

        # Track metrics
        running_loss += loss.item()
        _, predicted = outputs.max(dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.1f}%"})

    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    """
    Evaluate model on test set.
    Returns (average_loss, accuracy, all_predictions, all_labels).
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(test_loader, desc="Evaluating")

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = outputs.max(dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Collect for confusion matrix
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy, all_preds, all_labels


# ============================================================================
# Metrics & Visualization
# ============================================================================

def plot_confusion_matrix(
    all_preds: List[int],
    all_labels: List[int],
    save_path: Path,
) -> None:
    """Create and save confusion matrix visualization."""
    # Compute confusion matrix using sklearn
    cm = confusion_matrix(all_labels, all_preds)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw heatmap
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Set labels
    ax.set(
        xticks=np.arange(len(CIFAR10_CLASSES)),
        yticks=np.arange(len(CIFAR10_CLASSES)),
        xticklabels=CIFAR10_CLASSES,
        yticklabels=CIFAR10_CLASSES,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )

    # Rotate x labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Add text annotations (numbers in each cell)
    thresh = cm.max() / 2.0
    for i in range(len(CIFAR10_CLASSES)):
        for j in range(len(CIFAR10_CLASSES)):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def print_classification_report(all_preds: List[int], all_labels: List[int]) -> None:
    """Print per-class precision, recall, F1-score."""
    report = classification_report(all_labels, all_preds, target_names=CIFAR10_CLASSES)
    print("\nClassification Report:")
    print(report)


# ============================================================================
# Main Function
# ============================================================================

def main():
    """
    Main training function.

    Edit the config dictionary below to change settings.
    Then run this script from PyCharm.
    """

    # =========================================================================
    # CONFIGURATION - Edit these values directly
    # =========================================================================
    config = {
        # Model: "cnn" or "vit"
        "model": "vit",

        # Image size: 64 (fast) or 128 (more accurate)
        "img_size": 64,

        # Training parameters
        "batch_size": 64,
        "epochs": 10,           # Set to any number you want
        "lr": 1e-3,             # Learning rate
        "patience": 3,          # Early stopping patience (0 to disable)

        # Debug mode: use only 200 samples for quick testing
        "debug": False,

        # Device: True = use MPS (Apple GPU), False = use CPU
        "use_mps": True,

        # Cleanup: delete cached data after training
        "cleanup": False,

        # Random seed for reproducibility
        "seed": 42,

        # Output directory for checkpoints and plots
        "output_dir": "./output",
    }
    # =========================================================================

    # Setup
    set_seed(config["seed"])
    device = get_device(use_mps=config["use_mps"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data with proper train/val/test split
    num_samples = 200 if config["debug"] else None
    train_ds, val_ds, test_ds, cache_dir = load_cifar10(num_samples=num_samples)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds,
        img_size=config["img_size"],
        batch_size=config["batch_size"],
    )

    # Create model
    print(f"\nCreating {config['model'].upper()} model...")
    if config["model"] == "cnn":
        model = SimpleCNN(num_classes=10, input_size=config["img_size"])
    else:
        model = create_vit_model(num_classes=10, img_size=config["img_size"])

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["lr"] * 0.01)

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0

    print(f"\nStarting training for {config['epochs']} epochs...")
    print("=" * 60)

    for epoch in range(1, config["epochs"] + 1):
        # Train on training set
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Evaluate on VALIDATION set (not test!)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Print epoch summary
        print(f"\nEpoch {epoch}/{config['epochs']}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, output_dir / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping (based on validation)
        if config["patience"] > 0 and patience_counter >= config["patience"]:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {config['patience']} epochs)")
            break

    # =========================================================================
    # FINAL EVALUATION ON TEST SET (only now do we touch test data!)
    # =========================================================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - Now evaluating on TEST set")
    print("=" * 60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    # Evaluate on TEST set (first and only time!)
    print("\nFinal evaluation on held-out TEST set:")
    test_loss, test_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, device)
    print(f"TEST Accuracy: {test_acc:.2f}%")
    print(f"TEST Loss: {test_loss:.4f}")

    # Save confusion matrix (from test set)
    plot_confusion_matrix(all_preds, all_labels, output_dir / "confusion_matrix.png")
    print_classification_report(all_preds, all_labels)

    # Cleanup
    if config["cleanup"]:
        shutil.rmtree(cache_dir)
        print(f"\nCleaned up cache: {cache_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
