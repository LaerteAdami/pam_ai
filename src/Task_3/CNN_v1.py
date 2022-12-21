import torch.nn as nn

class CNN_v1(nn.Module):
    
    def __init__(self, num_classes=10):
        super(CNN_v1, self).__init__()
        #First unit of convolution
        self.network = nn.Sequential(
            
            ## FEATURE LEARNING ##
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14x14x64
      
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=2, stride=2), # 7 x 7 x 64

            ## CLASSIFICATION ##
            nn.Flatten(),

            nn.Linear(7*7*64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
            nn.Softmax(dim=1))

    def forward(self, x):
        return self.network(x)

