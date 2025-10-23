# Frequently Asked Questions (FAQ)

## 📋 Table of Contents
- [Setup & Installation](#setup--installation)
- [Competition Basics](#competition-basics)
- [Model Training](#model-training)
- [Data & Augmentation](#data--augmentation)
- [Kaggle Submission](#kaggle-submission)
- [Debugging & Errors](#debugging--errors)
- [Performance & Optimization](#performance--optimization)

---

## Setup & Installation

### Q: What are the system requirements?
**A:** You need:
- Python 3.8 or higher
- 4GB+ RAM (8GB recommended)
- 5GB free disk space
- GPU is optional but recommended (can use Google Colab for free GPU)

### Q: How do I install all dependencies?
**A:** Run this command in your terminal:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch torchvision pandas pillow tqdm matplotlib scikit-learn scikit-image
```

### Q: Can I use Google Colab instead of local setup?
**A:** Yes! Upload `cifar10_comp_colab.ipynb` to Google Colab and use their free GPU:
1. Go to https://colab.research.google.com
2. Upload the notebook
3. Runtime → Change runtime type → GPU
4. Run all cells

### Q: I don't have a GPU. Can I still participate?
**A:** Absolutely! Options:
- Train on CPU (slower but works)
- Use Google Colab (free GPU)
- Reduce batch size and train fewer epochs

---

## Competition Basics

### Q: What is CIFAR-10?
**A:** CIFAR-10 is an image classification dataset with:
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training images (32×32 pixels)
- 10,000 validation images

### Q: What's different about the competition test set?
**A:** The test set has **augmentations** applied:
- Gaussian noise
- Motion blur
- Color shifts
- Brightness/contrast changes
- Rotations and translations

This makes it more challenging than the standard CIFAR-10!

### Q: What files do I need from Kaggle?
**A:** Download these two files:
1. `test.csv` - Contains image IDs
2. `test_images.zip` - Contains test images

Place both in the `cifar10_comp/` folder and unzip the images.

### Q: Can I work in teams?
**A:** Yes, it is highly recommended that you work in groups of up to 4 people. 

---

## Model Training

### Q: How long does training take?
**A:** Depends on your hardware.

### Q: How many epochs should I train?
**A:** Start with:
- 10 epochs for initial experiments
- 20-30 epochs for better results
- Monitor validation accuracy - stop if it plateaus

### Q: Should I use Adam or SGD optimizer?
**A:**
- **Adam:** Easier to use, good default choice
- **SGD with momentum:** Can achieve better results but requires tuning
- **AdamW:** Good middle ground with weight decay

Start with Adam, then experiment with others.

### Q: What learning rate should I use?
**A:** Common values:
- **Adam:** 0.001 (default), try 0.0001 or 0.003
- **SGD:** 0.01 or 0.1, use learning rate scheduler
- If loss doesn't decrease, try lower LR
- If training is too slow, try higher LR

### Q: My training accuracy is 99% but validation is 60%. What's wrong?
**A:** You're **overfitting**! Solutions:
1. Add more data augmentation
2. Increase dropout (try 0.5 or 0.7)
3. Use weight decay (switch to AdamW)
4. Train for fewer epochs
5. Simplify your model

### Q: My model isn't improving after epoch 5. What should I do?
**A:** Try:
1. Lower learning rate (divide by 10)
2. Use learning rate scheduler
3. Add BatchNorm layers
4. Check if you're underfitting (train acc also low?)
5. Increase model capacity (more layers/filters)

---

## Data & Augmentation

### Q: What data augmentation should I use?
**A:** Essential augmentations (add to `get_transforms()`):
```python
transforms.RandomHorizontalFlip(),
transforms.RandomCrop(32, padding=4),
transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
transforms.RandomRotation(15),
```

Advanced (optional):
```python
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
transforms.RandomGrayscale(p=0.1),
transforms.GaussianBlur(kernel_size=3),
```

### Q: Why is data augmentation so important?
**A:** The competition test set has augmentations! Training with augmentations:
- Makes your model robust to distortions
- Prevents overfitting to clean images
- Can improve Kaggle score by 10-15%

### Q: Should I augment the validation/test data?
**A:** **NO!** Only augment training data. Validation and test should use clean transforms:
```python
transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
```

### Q: How much augmentation is too much?
**A:** Signs of too much augmentation:
- Training accuracy is very low (<40%)
- Validation accuracy > Training accuracy
- Model can't learn patterns

Start conservative, gradually add more.

---

## Kaggle Submission

### Q: What should my submission.csv look like?
**A:** Format:
```csv
id,label
0,3
1,8
2,5
...
```
- **id:** Test image ID (from test.csv)
- **label:** Predicted class (0-9)

The scripts generate this automatically!

### Q: How do I generate submission.csv?
**A:** Two ways:

**Using Notebook:**
Run the last cell in `cifar10_comp.ipynb`

**Using Script:**
```bash
python kaggle_submission.py
```

### Q: I got "test.csv not found" error
**A:**
1. Download `test.csv` from Kaggle
2. Place it in the `cifar10_comp/` folder (same directory as main.py)
3. Make sure you're running the script from the correct directory

### Q: My submission has wrong number of predictions
**A:** Check:
1. Did you unzip `test_images.zip`?
2. Are all images in `cifar10_comp/test_images/`?
3. Does test.csv have 10,000 rows (plus header)?

### Q: Can I submit multiple times?
**A:** Yes! Kaggle typically allows:
- Multiple submissions per day (check competition rules)
- Your best score is used for the leaderboard

---

## Debugging & Errors

### Q: ImportError: No module named 'torch'
**A:** Install PyTorch:
```bash
pip install torch torchvision
```

### Q: CUDA out of memory error
**A:** Solutions:
1. Reduce batch size: `--batch_size 64` (or 32)
2. Use smaller model
3. Use CPU instead: Model training will be slower but work
4. Use Google Colab with free GPU

### Q: RuntimeError: Expected all tensors to be on same device
**A:** Make sure images and model are on same device:
```python
images = images.to(device)
labels = labels.to(device)
model = model.to(device)
```

### Q: My validation accuracy is 10% (random guessing)
**A:** Check:
1. Is your model architecture correct?
2. Are you using the right loss function? (CrossEntropyLoss)
3. Is learning rate too high? Try 0.001
4. Did you normalize your data?

### Q: Loss is NaN or Infinity
**A:** Causes:
1. Learning rate too high → Lower it
2. No gradient clipping → Add clipping
3. Numerical instability → Check normalization
4. Exploding gradients → Use BatchNorm

### Q: FileNotFoundError: test_images/00000.png
**A:**
1. Unzip `test_images.zip` in `cifar10_comp/` folder
2. Check images are in `cifar10_comp/test_images/`
3. Image files should be named `00000.png`, `00001.png`, etc.

---

## Performance & Optimization

### Q: What accuracy should I expect?
**A:** Benchmarks:
- **Baseline (SimpleCNN, no aug):** 50-55%
- **With augmentation:** 60-65%
- **Improved model + augmentation:** 68-73%
- **Advanced techniques:** 73-78%

### Q: How can I improve my Kaggle score?
**A:** Priority order:

**1. Data Augmentation (🔥 Highest impact)**
- Add ColorJitter, RandomRotation, RandomCrop
- Impact: +10-15%

**2. Model Architecture**
- Add BatchNorm layers
- Add more conv layers
- Impact: +5-10%

**3. Train Longer**
- 20-30 epochs instead of 10
- Impact: +2-5%

**4. Hyperparameter Tuning**
- Try different learning rates
- Experiment with optimizers
- Impact: +1-3%

### Q: My validation accuracy is 75% but Kaggle score is only 65%
**A:** This is **normal**! The test set has augmentations, causing domain shift:
- Expect Kaggle score to be 2-10% lower
- Solution: Add more augmentation to training

### Q: Should I use pretrained models?
**A:** No.

### Q: How do I prevent overfitting?
**A:** Techniques:
1. **Data augmentation** (most effective)
2. **Dropout** (0.3-0.5 after each layer)
3. **Weight decay** (use AdamW optimizer)
4. **Early stopping** (stop when val acc plateaus)
5. **Reduce model complexity**

### Q: Can I ensemble multiple models?
**A:** Yes! If you have time:
1. Train 3-5 models with different:
   - Random seeds
   - Architectures
   - Hyperparameters
2. Average their predictions
3. Can improve score by 1-2%

---

## General Tips

### Q: I'm stuck and don't know what to do next
**A:** Follow this checklist:
- [ ] Did you add data augmentation?
- [ ] Did you add BatchNorm to your model?
- [ ] Are you training for 20+ epochs?
- [ ] Is your learning rate appropriate?
- [ ] Are you monitoring train vs validation accuracy?
- [ ] Did you try different model architectures?

### Q: What's the most common mistake?
**A:** Not adding data augmentation! This is the #1 factor for good Kaggle scores.

### Q: Where can I find more help?
**A:** Resources:
1. Ask questions during workshop
2. Kaggle competition discussion forum
3. PyTorch documentation: https://pytorch.org/docs
4. Check README.md for setup instructions

---

## Advanced Questions

### Q: What is BatchNorm and why do I need it?
**A:** BatchNorm normalizes layer activations:
- Stabilizes training
- Allows higher learning rates
- Reduces overfitting
- Add after Conv layers: `nn.BatchNorm2d(channels)`

### Q: What's the difference between validation and test sets?
**A:**
- **Validation:** From CIFAR-10 (clean images), used during training
- **Test (Kaggle):** From competition (augmented images), used for scoring
- Your goal: Generalize from clean to augmented

### Q: Should I use learning rate scheduling?
**A:** Yes, it helps! Options:
```python
# Step decay
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Cosine annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

### Q: What is test-time augmentation (TTA)?
**A:** Apply augmentations during inference:
1. Augment test image multiple times
2. Get prediction for each version
3. Average predictions
4. Can improve score by 0.5-1%

### Q: How do I know if my model is underfitting or overfitting?
**A:**

**Underfitting (model too simple):**
- Low train accuracy (<70%)
- Low validation accuracy
- Solution: Bigger model, train longer

**Overfitting (model too complex):**
- High train accuracy (>90%)
- Low validation accuracy (<70%)
- Large gap between train and val
- Solution: More augmentation, dropout, weight decay

**Good fit:**
- Train: 75-85%
- Val: 70-80%
- Small gap (3-5%)

---

## Still Have Questions?

If your question isn't answered here:
1. Check the main **README.md**
2. Ask during the workshop
3. Post on Discord discussion forum
4. Review the demo notebook for examples

**Good luck! 🚀**
