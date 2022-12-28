import torch.nn as nn
import torch

torch.set_default_dtype(torch.float64)


class VanillaDNN(nn.Module):
    """
    Simple fully connected neural network for classification

    """

    def __init__(self):
        super(VanillaDNN, self).__init__()
        # First unit of neural network
        self.network = nn.Sequential(
            nn.Linear(22, 30),
            nn.Dropout(0.5),
            nn.Sigmoid(),
            nn.Linear(30, 2),
            nn.Dropout(0.5),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.network(x)


class ImprovedDNN(nn.Module):
    """
    Improved fully connected neural network for classification

    """

    def __init__(self):
        super(ImprovedDNN, self).__init__()
        # First unit of neural network
        self.network = nn.Sequential(
            nn.Linear(22, 200),
            nn.Dropout(0.8),
            nn.ReLU(),
            nn.Linear(200, 300),
            nn.Dropout(0.8),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.Dropout(0.8),
            nn.ReLU(),
            nn.Linear(200, 2),
            nn.Dropout(0.8),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.network(x)
