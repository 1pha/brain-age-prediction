import torch
import torch.nn as nn


class Vanilla3d(nn.Module):
    def __init__(self, cfg=None):
        super(Vanilla3d, self).__init__()
        self.cfg = cfg

        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 8, 3, 2),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv3d(8, 16, 3, 2),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv3d(16, 32, 3, 2),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)
        x = self.fc2(x)

        return x
