import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet2(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet2, self).__init__()
        self.features = nn.Sequential(
            # 1st layer
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.LocalResponseNorm(k=2, size=5, alpha=0.0001, beta=0.75), # paper section 3.3
            # 2nd layer
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.LocalResponseNorm(k=2, size=5, alpha=0.0001, beta=0.75),
            # 3rd layer
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 4th layer
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 5th layer
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            # 6th layer
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(),
            # 7th layer
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            # last layer
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 7 * 7)
        x = self.classifier(x)
        return x
