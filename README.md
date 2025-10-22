# ACM AI Workshop - Computer Vision

Welcome to the ACM AI Workshop on Computer Vision and Image Classification!

## 📚 Workshop Structure

### Part 1: Demo - CIFAR-10 (Instructor-led)
**File:** `cifar10_demo_student.ipynb`

In this interactive demo, you'll learn:
- What image classification is
- Feature engineering (the traditional way)
- Data augmentation (making models robust)
- Convolutional Neural Networks (the modern way)

**👉 Follow along as the instructor walks through the notebook!**

---

### Part 2: Competition - CIFAR-100 (Hands-on)
**File:** `cifar100_comp.ipynb`

Build a CNN to classify CIFAR-100 images (100 classes) and compete on Kaggle!

**Goal:** Achieve the highest accuracy on the test set.

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install torch torchvision pandas pillow tqdm matplotlib scikit-learn scikit-image
```

### 2. Demo (Follow Along)

Open the demo notebook:
```bash
jupyter notebook cifar10_demo_student.ipynb
```

The instructor will guide you through completing the code!

### 3. Competition (Hands-on)

Open the competition notebook:
```bash
jupyter notebook cifar100_comp.ipynb
```

This notebook includes:
- Data exploration
- Starter CNN code
- Training loop
- Function to generate `submission.csv`

---

## 📊 Competition Workflow

### Step 1: Train Your Model

Run `cifar100_comp.ipynb` to:
- Explore the CIFAR-100 dataset
- Build and train a CNN
- Save the best model as `best_model.pth`

### Step 2: Download Test Data from Kaggle

Download these files from the Kaggle competition page:
- `test.csv` - List of test image IDs
- `test_images.zip` - Test images folder

Unzip `test_images.zip` in the same directory as your notebook.

### Step 3: Generate Predictions

Run the last cell in `cifar100_comp.ipynb` to:
- Load your best model
- Generate predictions for test images
- Save `submission.csv`

### Step 4: Submit to Kaggle

Upload `submission.csv` to Kaggle and check your score!

---

## 💡 Tips for Success

### 1. Data Augmentation is KEY! 🔑

The test set has augmentations (noise, blur, color shifts, etc.).

**Add augmentations to your training data** in the `get_transforms()` function:
```python
transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
transforms.RandomRotation(15),
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
```

### 2. Improve the Model Architecture

The baseline CNN is simple. Try:
- Adding more convolutional layers
- Using BatchNorm after Conv layers
- Experimenting with different filter sizes
- Adding residual connections

### 3. Train Longer

The default is 10 epochs. Try training for 20-50 epochs!

### 4. Experiment with Hyperparameters

- Learning rate
- Batch size
- Optimizer (Adam vs SGD vs AdamW)
- Dropout rate

---

## 🐛 Troubleshooting

### "test.csv not found"
Download `test.csv` from Kaggle and place it in the same directory as the notebook.

### "test_images/ not found"
Download and unzip `test_images.zip` from Kaggle.

### "CUDA out of memory"
Reduce batch size in the notebook (try 64 or 32 instead of 128).

### Low Kaggle score despite good training accuracy?
You're overfitting to clean images! Add more data augmentation.

---

## 📁 File Structure

```
workshop/
├── cifar10_demo_student.ipynb    # Demo notebook (follow along)
├── cifar100_comp.ipynb            # Competition notebook (hands-on)
├── best_model.pth                 # Your trained model (generated)
├── submission.csv                 # Your predictions (generated)
├── test.csv                       # Test image IDs (download from Kaggle)
└── test_images/                   # Test images (download from Kaggle)
    ├── 00000.png
    ├── 00001.png
    └── ...
```

---

## 🏆 Submission Format

Your `submission.csv` must have:
```csv
id,label
00000,42
00001,17
00002,89
...
```

- **id:** Image ID (from test.csv)
- **label:** Predicted class (0-99)

---

## 📚 Resources

- **PyTorch Tutorials:** https://pytorch.org/tutorials/
- **CIFAR-100 Dataset:** https://www.cs.toronto.edu/~kriz/cifar.html
- **Data Augmentation:** https://pytorch.org/vision/stable/transforms.html

---

## 🤝 Getting Help

- Ask questions during the workshop!
- Check the Kaggle competition discussion forum
- Review the demo notebook for examples

---

Good luck and have fun! 🚀

**Remember:** The key to winning is data augmentation + good architecture!
