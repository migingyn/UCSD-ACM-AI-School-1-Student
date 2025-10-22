"""
CIFAR-100 Competition - Training Script

This script trains a CNN on CIFAR-100 and generates predictions for Kaggle submission.

Usage:
    python main.py --epochs 20 --lr 0.001 --batch_size 128

    Or with all options:
    python main.py --epochs 50 --lr 0.001 --batch_size 64 --optimizer adam --scheduler

Arguments:
    --epochs: Number of training epochs (default: 10)
    --lr: Learning rate (default: 0.001)
    --batch_size: Batch size for training (default: 128)
    --optimizer: Optimizer choice - adam, sgd, or adamw (default: adam)
    --scheduler: Use learning rate scheduler (flag, default: False)
    --model: Model to use (default: SimpleCNN)

The script will:
1. Load CIFAR-100 dataset
2. Train the model
3. Save the best model as 'best_model.pth'
4. Generate 'submission.csv' for Kaggle (if test.csv and test_images/ exist)
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import pandas as pd
from PIL import Image
import os

# Import the model
from model import SimpleCNN


def get_transforms(augment=False):
    """
    Get data transforms for training and testing

    Args:
        augment: If True, apply data augmentation (for training)
                 If False, only normalize (for testing)

    Returns:
        Composed transforms
    """
    if augment:
        # TODO: Add MORE augmentations here! This is KEY to better performance!
        # The test set has augmentations (noise, blur, color shifts, etc.)
        # Train with similar augmentations to generalize better!
        #
        # Suggestions:
        # - transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
        # - transforms.RandomRotation(15)
        # - transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
        # - transforms.RandomGrayscale(p=0.1)
        # - Add noise? Blur? (use custom transforms)
        #
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            # TODO: Add more augmentations here!

            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        # No augmentation for testing
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def generate_submission(model, test_csv='test.csv', test_images_dir='test_images',
                       output_csv='submission.csv', device='cpu'):
    """
    Generate Kaggle submission file

    Args:
        model: Trained model
        test_csv: Path to test.csv (contains image IDs)
        test_images_dir: Directory containing test images
        output_csv: Path to save submission
        device: Device to run inference on
    """
    print('\n' + '='*60)
    print('GENERATING KAGGLE SUBMISSION')
    print('='*60)

    # Check if test files exist
    if not os.path.exists(test_csv):
        print(f'❌ {test_csv} not found!')
        print('   Download test.csv from Kaggle to generate submission.')
        return

    if not os.path.exists(test_images_dir):
        print(f'❌ {test_images_dir}/ not found!')
        print('   Download and unzip test_images.zip from Kaggle.')
        return

    # Load test image IDs
    test_df = pd.read_csv(test_csv)
    print(f'Found {len(test_df)} test images')

    # Get transforms (no augmentation for testing!)
    test_transform = get_transforms(augment=False)

    # Generate predictions
    model.eval()
    predictions = []

    with torch.no_grad():
        for img_id in tqdm(test_df['id'], desc='Predicting'):
            # Load image
            img_path = os.path.join(test_images_dir, f'{img_id}.png')
            if not os.path.exists(img_path):
                print(f'Warning: {img_path} not found, skipping...')
                continue

            img = Image.open(img_path).convert('RGB')
            img_tensor = test_transform(img).unsqueeze(0).to(device)

            # Predict
            output = model(img_tensor)
            pred_class = output.argmax(1).item()

            predictions.append({
                'id': img_id,
                'label': pred_class
            })

    # Save submission
    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv(output_csv, index=False)

    print(f'\n✅ Submission saved to {output_csv}')
    print(f'   Total predictions: {len(submission_df)}')
    print('\nPreview:')
    print(submission_df.head(10))
    print('\n' + '='*60)
    print(f'📤 Upload {output_csv} to Kaggle!')
    print('='*60 + '\n')


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CIFAR-100 Competition Training')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training (default: 128)')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd', 'adamw'],
                       help='Optimizer: adam, sgd, or adamw (default: adam)')
    parser.add_argument('--scheduler', action='store_true',
                       help='Use learning rate scheduler (default: False)')
    parser.add_argument('--model', type=str, default='SimpleCNN',
                       help='Model architecture (default: SimpleCNN)')
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')

    # Print configuration
    print('='*60)
    print('TRAINING CONFIGURATION')
    print('='*60)
    print(f'Device: {device}')
    print(f'Epochs: {args.epochs}')
    print(f'Learning Rate: {args.lr}')
    print(f'Batch Size: {args.batch_size}')
    print(f'Optimizer: {args.optimizer}')
    print(f'LR Scheduler: {args.scheduler}')
    print(f'Model: {args.model}')
    print('='*60 + '\n')

    # Load CIFAR-100 dataset
    print('Loading CIFAR-100 dataset...')
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True,
                                     transform=get_transforms(augment=True))
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True,
                                    transform=get_transforms(augment=False))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    print(f'Training images: {len(train_dataset)}')
    print(f'Test images: {len(test_dataset)}\n')

    # Create model
    # TODO: Support different models from model.py
    model = SimpleCNN(num_classes=100).to(device)
    print(f'Model: {args.model}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}\n')

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer selection
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    print(f'Optimizer: {optimizer.__class__.__name__}')

    # Learning rate scheduler (optional)
    scheduler = None
    if args.scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        print(f'LR Scheduler: StepLR (step_size=10, gamma=0.1)')
    print()

    # Training loop
    print('='*60)
    print('TRAINING START')
    print('='*60 + '\n')

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f'Epoch {epoch}/{args.epochs}')

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        test_loss, test_acc = validate(model, test_loader, criterion, device)

        print(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'✓ Saved best model (acc: {best_acc:.2f}%)')

        # Update learning rate scheduler
        if scheduler:
            scheduler.step()
            print(f'Learning rate: {scheduler.get_last_lr()[0]:.6f}')

        print()

    print('='*60)
    print('TRAINING COMPLETE')
    print('='*60)
    print(f'Best test accuracy: {best_acc:.2f}%')
    print('Model saved as best_model.pth\n')

    # Generate Kaggle submission
    print('Attempting to generate Kaggle submission...')
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    generate_submission(model, device=device)


if __name__ == '__main__':
    main()
