"""
Generate Kaggle competition submission from trained model

This script:
1. Loads your best trained model (best_model.pth)
2. Reads test.csv to get image IDs
3. Loads images from test_images/ folder
4. Generates predictions
5. Saves submission.csv for Kaggle upload
"""

import argparse
import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from model import get_model
from dataset import get_transforms


CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def generate_submission(model_path, test_csv, test_images_dir, output_csv, device):
    """
    Generate Kaggle submission file

    Args:
        model_path: Path to saved model checkpoint (e.g., best_model.pth)
        test_csv: Path to test.csv file (contains image IDs)
        test_images_dir: Directory containing test images
        output_csv: Path to save submission CSV
        device: Device to run inference on
    """
    # Load model
    print(f"Loading model from: {model_path}")
    model = get_model(num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully!")

    # Load test transforms (no augmentation)
    test_tfms = get_transforms(augment=False)

    # Read test.csv to get image IDs
    print(f"\nReading test data from: {test_csv}")
    if not os.path.isfile(test_csv):
        raise FileNotFoundError(f"test.csv not found at: {test_csv}")

    test_df = pd.read_csv(test_csv)

    if 'id' not in test_df.columns:
        raise ValueError("test.csv must have an 'id' column")

    print(f"Found {len(test_df)} images to predict")

    # Check test images directory
    if not os.path.isdir(test_images_dir):
        raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")

    # Generate predictions
    print(f"\nGenerating predictions from {test_images_dir}/")
    predictions = []

    with torch.no_grad():
        for img_id in tqdm(test_df['id'], desc="Predicting"):
            # Load image
            img_path = os.path.join(test_images_dir, f"{img_id}.png")

            if not os.path.exists(img_path):
                print(f"\nWarning: Image {img_path} not found, skipping...")
                continue

            img = Image.open(img_path).convert("RGB")

            # Apply transforms
            x = test_tfms(img).unsqueeze(0).to(device)

            # Predict
            logits = model(x)
            pred_idx = logits.argmax(1).item()
            pred_label = CLASSES[pred_idx]

            predictions.append({
                "id": img_id,
                "label": pred_label
            })

    # Create submission DataFrame
    submission_df = pd.DataFrame(predictions)

    # Save submission
    submission_df.to_csv(output_csv, index=False)

    print(f"\n{'='*60}")
    print("SUBMISSION GENERATED")
    print(f"{'='*60}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"Submission saved to: {output_csv}")
    print(f"\nSubmission preview:")
    print(submission_df.head(10))
    print(f"\nClass distribution:")
    print(submission_df['label'].value_counts().sort_index())
    print(f"\n{'='*60}")
    print(f"\nYou can now upload '{output_csv}' to Kaggle!")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle submission from trained model")
    parser.add_argument("--model", type=str, default="best_model.pth",
                        help="Path to saved model checkpoint")
    parser.add_argument("--test_csv", type=str, default="test.csv",
                        help="Path to test.csv file")
    parser.add_argument("--test_images_dir", type=str, default="test_images",
                        help="Directory containing test images")
    parser.add_argument("--output", type=str, default="submission.csv",
                        help="Path to save submission CSV")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else
                          "cpu")
    print(f"Using device: {device}")

    # Generate submission
    generate_submission(
        model_path=args.model,
        test_csv=args.test_csv,
        test_images_dir=args.test_images_dir,
        output_csv=args.output,
        device=device
    )


if __name__ == "__main__":
    main()
