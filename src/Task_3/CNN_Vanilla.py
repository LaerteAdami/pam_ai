import torch.nn as nn

class VanillaNN(nn.Module):
    
    def __init__(self, num_classes=10):
        super(VanillaNN, self).__init__()
        #First unit of convolution
        self.network = nn.Sequential(
            
            ## FEATURE LEARNING ##
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14x14x64
      
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 7 x 7 x 64

            ## CLASSIFICATION ##
            nn.Flatten(),

            nn.Linear(7*7*64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Softmax(dim=1))

    def forward(self, x):
        return self.network(x)

