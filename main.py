"""
Main training script for CIFAR-10
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import get_dataloaders
from model import get_model


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return correct / total, loss_sum / total


def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating"):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)

    return correct / total, loss_sum / total


def test(model, loader, criterion, device):
    """Test the model"""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Testing"):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)

    return correct / total, loss_sum / total


def save_best_model(model, val_acc, best_val_acc, checkpoint_path):
    """
    Save model checkpoint when validation accuracy improves

    You can customize this function to save based on different criteria:
    - Highest validation accuracy (default)
    - Lowest validation loss
    - Combination of metrics

    Args:
        model: The model to save
        val_acc: Current validation accuracy
        best_val_acc: Best validation accuracy so far
        checkpoint_path: Path to save checkpoint

    Returns:
        Updated best_val_acc
    """
    # Default: Save model with highest validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✓ Saved best model (val_acc: {best_val_acc:.4f})")

    return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth", help="Checkpoint path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=2
    )

    # Initialize model
    print("Initializing model...")
    model = get_model(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Training loop
    best_val_acc = 0.0

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_acc, train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_loss = validate(model, val_loader, criterion, device)

        print(f"Train - Acc: {train_acc:.4f}, Loss: {train_loss:.4f}")
        print(f"Val   - Acc: {val_acc:.4f}, Loss: {val_loss:.4f}")

        # Save best model
        best_val_acc = save_best_model(model, val_acc, best_val_acc, args.checkpoint)

    # Load best model and evaluate on test set
    print(f"\nLoading best model from {args.checkpoint}...")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    print("\nEvaluating on CIFAR-10 test set...")
    test_acc, test_loss = test(model, test_loader, criterion, device)

    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss:     {test_loss:.4f}")
    print("="*60)

    print(f"\nTraining complete! Best model saved to: {args.checkpoint}")


if __name__ == "__main__":
    main()
