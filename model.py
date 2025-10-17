"""
Model definition for CIFAR-10 classification

TODO: Implement your own model architecture
"""

import torch.nn as nn


class SimpleCNN(nn.Module):
    """Simple CNN baseline for reference"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_model(num_classes=10):
    """
    Factory function to create and return model

    Args:
        num_classes: Number of output classes

    Returns:
        nn.Module: Model instance
    """
    return SimpleCNN(num_classes=num_classes)
