"""
Generate Kaggle Submission from Trained Model

This script loads a trained model and generates submission.csv for Kaggle.

Usage:
    python kaggle_submission.py

    Or specify custom paths:
    python kaggle_submission.py --model my_model.pth --output my_submission.csv

Requirements:
    - best_model.pth (or specify with --model)
    - test.csv
    - test_images/ folder
"""

import argparse
import os

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Import model builders
try:
    from model_solution import get_model  # Advanced architectures
    SOLUTION_MODELS_AVAILABLE = True
except ImportError:
    SOLUTION_MODELS_AVAILABLE = False

from model import SimpleCNN


def get_test_transforms(arch: str):
    """
    Get transforms for test images (no augmentation!)
    """
    if arch == 'simple':
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def build_model(arch: str, device: torch.device):
    """Instantiate the requested architecture."""
    if arch == 'simple':
        model = SimpleCNN(num_classes=10).to(device)
        description = 'SimpleCNN (model.py)'
    else:
        if not SOLUTION_MODELS_AVAILABLE:
            raise RuntimeError('model_solution.py is unavailable, cannot load advanced architectures.')
        model = get_model(arch, num_classes=10).to(device)
        description = f'{arch} (model_solution.py)'
    return model, description


def generate_submission(model, test_csv='test.csv', test_images_dir='test_images',
                       output_csv='submission.csv', device='cpu', arch='simple'):
    """
    Generate Kaggle submission file

    Args:
        model: Trained model
        test_csv: Path to test.csv (contains image IDs)
        test_images_dir: Directory containing test images
        output_csv: Path to save submission
        device: Device to run inference on
        arch: Architecture name for normalization settings
    """
    print('\n' + '='*60)
    print('GENERATING KAGGLE SUBMISSION')
    print('='*60)

    # Check if test files exist
    if not os.path.exists(test_csv):
        print(f'[ERROR] {test_csv} not found!')
        print('   Download test.csv from Kaggle to generate submission.')
        return False

    if not os.path.exists(test_images_dir):
        print(f'[ERROR] {test_images_dir}/ not found!')
        print('   Download and unzip test_images.zip from Kaggle.')
        return False

    # Load test image IDs
    test_df = pd.read_csv(test_csv)
    print(f'Found {len(test_df)} test images')

    # Get transforms (no augmentation for testing!)
    test_transform = get_test_transforms(arch)

    # Generate predictions
    model.eval()
    predictions = []

    with torch.no_grad():
        for img_id in tqdm(test_df['id'], desc='Predicting'):
            # Load image
            img_filename = f'{str(img_id).zfill(5)}.png'
            img_path = os.path.join(test_images_dir, img_filename)
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

    print(f'\n[OK] Submission saved to {output_csv}')
    print(f'   Total predictions: {len(submission_df)}')
    print('\nPreview:')
    print(submission_df.head(10))
    print('\n' + '='*60)
    print(f'>>> Upload {output_csv} to Kaggle!')
    print('='*60 + '\n')

    return True


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate Kaggle Submission')
    parser.add_argument('--model', type=str, default='best_model.pth',
                       help='Path to trained model (default: best_model.pth)')
    parser.add_argument('--test_csv', type=str, default='test.csv',
                       help='Path to test.csv (default: test.csv)')
    parser.add_argument('--test_images', type=str, default='test_images',
                       help='Path to test images directory (default: test_images)')
    parser.add_argument('--output', type=str, default='submission.csv',
                       help='Output submission filename (default: submission.csv)')
    parser.add_argument('--arch', type=str, default='simple',
                       choices=['simple', 'advanced', 'wide'],
                       help='Model architecture to evaluate (default: simple)')
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')

    print('='*60)
    print('KAGGLE SUBMISSION GENERATOR')
    print('='*60)
    print(f'Device: {device}')
    print(f'Model: {args.model}')
    print(f'Architecture: {args.arch}')
    print(f'Test CSV: {args.test_csv}')
    print(f'Test Images: {args.test_images}')
    print(f'Output: {args.output}')
    print('='*60)

    # Check if model file exists
    if not os.path.exists(args.model):
        print(f'\n[ERROR] Model file not found: {args.model}')
        print('   Please train a model first by running: python main.py')
        return

    # Load model
    print(f'\nLoading model from {args.model}...')

    try:
        model, model_name = build_model(args.arch, device)
        print(f'Using {model_name}')
    except RuntimeError as err:
        print(f'[ERROR] {err}')
        return

    try:
        model.load_state_dict(torch.load(args.model, map_location=device))
        print('[OK] Model loaded successfully!')
    except Exception as e:
        print(f'[ERROR] Error loading model: {e}')
        print('\nTip: Make sure the model architecture matches the saved weights.')
        return

    # Generate submission
    success = generate_submission(
        model,
        test_csv=args.test_csv,
        test_images_dir=args.test_images,
        output_csv=args.output,
        device=device,
        arch=args.arch
    )

    if success:
        print('\n*** Done! Your submission is ready for Kaggle.')
    else:
        print('\n[ERROR] Failed to generate submission. Please check the errors above.')


if __name__ == '__main__':
    main()
