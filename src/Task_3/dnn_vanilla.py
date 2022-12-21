import torch.nn as nn

class VanillaDNN(nn.Module):
    """
    Simple fully connected neural network for classification
    
    """
    
    def __init__(self, num_classes=10):
        super(VanillaDNN, self).__init__()
        #First unit of neural network
        self.network = nn.Sequential(
            
            nn.Linear(22, 128),
            nn.Dropout(0.7),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(0.7),
            nn.Softmax(dim=1))

    def forward(self, x):
        return self.network(x)

