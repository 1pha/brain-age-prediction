import torch
import torch.nn as nn
import torch.nn.functional as F


class Levakov(nn.Module):

    def __init__(self, task_type):
        super(Levakov, self).__init__()
        self.task_type = task_type

        self.BN = nn.BatchNorm3d(1)
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 8, 3), nn.ReLU(),
            nn.Conv3d(8, 8, 3), nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(8)
        )

        self.layer2 = nn.Sequential(
            nn.Conv3d(8,  16, 3), nn.ReLU(),
            nn.Conv3d(16, 16, 3), nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(16),
            nn.Dropout(.5)
        )

        self.layer3 = nn.Sequential(
            nn.Conv3d(16, 32, 3), nn.ReLU(),
            nn.Conv3d(32, 32, 3), nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(32),
            nn.Dropout(.5)
        )

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(.3)


    def forward(self, x):
        
        x = self.BN(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.reshape(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        
        x = self.dropout(x)

        return x
