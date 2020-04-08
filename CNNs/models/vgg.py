import torch.nn as nn
import torch.nn.functional as F

class VGG11(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG11, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        y = F.relu(self.conv1_1(x))
        y = self.maxpool(y)
        y = F.relu(self.conv2_1(y))
        y = self.maxpool(y)
        y = F.relu(self.conv3_1(y))
        y = F.relu(self.conv3_2(y))
        y = self.maxpool(y)
        y = F.relu(self.conv4_1(y))
        y = F.relu(self.conv4_2_1(y))
        y = self.maxpool(y)
        y = F.relu(self.conv4_2_2(y))
        y = F.relu(self.conv4_2_3(y))
        y = self.maxpool(y)
        y = self.avgpool(y)

        y = y.view(-1, 512)
        y = self.fc(y)
        return y

class VGG19(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG19, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2_5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2_6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2_7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        y = F.relu(self.conv1_1(x))
        y = F.relu(self.conv1_2(y))
        y = self.maxpool(y)
        y = F.relu(self.conv2_1(y))
        y = F.relu(self.conv2_2(y))
        y = self.maxpool(y)
        y = F.relu(self.conv3_1(y))
        y = F.relu(self.conv3_2_1(y))
        y = F.relu(self.conv3_2_2(y))
        y = self.maxpool(y)
        y = F.relu(self.conv4_1(y))
        y = F.relu(self.conv4_2_1(y))
        y = F.relu(self.conv4_2_2(y))
        y = F.relu(self.conv4_2_3(y))
        y = self.maxpool(y)
        y = F.relu(self.conv4_2_4(y))
        y = F.relu(self.conv4_2_5(y))
        y = F.relu(self.conv4_2_6(y))
        y = F.relu(self.conv4_2_7(y))
        y = self.maxpool(y)
        y = self.avgpool(y)

        y = y.view(-1, 512)
        y = self.fc(y)
        return y

