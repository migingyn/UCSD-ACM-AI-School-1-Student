"""
CNN Models for CIFAR-10 Competition

This file contains model architectures for the competition.
SimpleCNN is provided as a baseline - try improving it!

TODO: Experiment with different architectures to improve accuracy!
"""

import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple CNN baseline for CIFAR-10

    This is a basic architecture to get you started.
    Current architecture achieves ~50-60% accuracy.

    TODO: Improve this architecture! Some ideas:
    - Add more convolutional layers
    - Add BatchNorm layers after Conv layers
    - Try different filter sizes
    - Experiment with different pooling strategies
    - Add residual connections (ResNet-style)
    - Try different activation functions
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32x32x3 -> 32x32x32
            nn.ReLU(inplace=True),
            # TODO: Add BatchNorm here? nn.BatchNorm2d(32)
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # 32x32x32 -> 16x16x32

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 16x16x32 -> 16x16x64
            nn.ReLU(inplace=True),
            # TODO: Add BatchNorm here?
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),  # 16x16x64 -> 8x8x64

            nn.Conv2d(64, 128, kernel_size=5, padding=2), # 8x8x64 -> 8x8x128
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(2),  # 8x8x128 -> 4x4x128

            # Block 3
            nn.Conv2d(128, 256, kernel_size=5, padding=2),  # 4x4x128 -> 4x4x256
            nn.ReLU(inplace=True),
            # TODO: Add BatchNorm here?
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((1, 1))  # 8x8x128 -> 1x1x128

            # TODO: Add more blocks? More layers usually = better performance!
            
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),  # TODO: Experiment with different dropout rates?
            nn.Linear(256, num_classes)
            # TODO: Add more fully connected layers?
            
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


# TODO: Try creating your own model architecture!
# Example:
# class ImprovedCNN(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ImprovedCNN, self).__init__()
#         # Your improved architecture here!
#         pass
#
#     def forward(self, x):
#         # Your forward pass here!
#         pass

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x