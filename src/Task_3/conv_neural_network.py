import torch.nn as nn
import torch


class VanillaCNN(nn.Module):
    """
    Simple convolutional neural network

    """

    def __init__(self):
        super(VanillaCNN, self).__init__()

        # Neural network architecture
        self.network = nn.Sequential(
            ## FEATURE LEARNING ##
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14x64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7 x 7 x 64
            ## CLASSIFICATION ##
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.network(x)


class ImprovedCNN(nn.Module):
    """
    Improved convolutional neural network

    """

    def __init__(self):
        super(ImprovedCNN, self).__init__()

        # Neural network architecture
        self.network = nn.Sequential(
            ## FEATURE LEARNING ##
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14x64
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7 x 7 x 64
            ## CLASSIFICATION ##
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.Dropout(0.5),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.network(x)
