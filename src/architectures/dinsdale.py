import torch
import torch.nn as nn
import torch.nn.functional as F

class Dinsdale(nn.Module):

    def __init__(self, in_channels, num_classes, filters=1):

        super(Dinsdale, self).__init__()

        filter_lst = {
            1: [ 8, 16, 16, 16, 32],
            2: [16, 32, 32, 32, 64],
            3: [32, 64, 64, 64, 96],
        }
        if isinstance(filters, int):
            self.filters = filter_lst[filters]

        elif isinstance(filters, list):
            self.filters = filters

        f1, f2, f3, f4, f5 = self.filters

        self.block1 = nn.Sequential(
            BasicConv3d(in_channels, f1),
            BasicConv3d(f1, f1, max_pool=True)
        )

        self.block2 = nn.Sequential(
            BasicConv3d(f1, f2),
            BasicConv3d(f2, f2, max_pool=True)
        )

        self.block3 = nn.Sequential(
            BasicConv3d(f2, f3),
            BasicConv3d(f3, f3, max_pool=True)
        )

        self.block4 = nn.Sequential(
            BasicConv3d(f3, f4),
            BasicConv3d(f4, f4),
            BasicConv3d(f4, f4, max_pool=True)
        )

        self.block5 = nn.Sequential(
            BasicConv3d(f4, f5),
            BasicConv3d(f5, f5, ada_pool=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


class BasicConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, max_pool=False, ada_pool=False):

        super(BasicConv3d, self).__init__()

        self.Conv3d_1  = nn.Conv3d(in_channels,  out_channels, 3, padding=1)
        # self.Conv3d_2  = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.batchnorm = nn.BatchNorm3d(out_channels)

        self.max_pool = max_pool
        if self.max_pool:
            self.maxpool = nn.MaxPool3d(2, 2)

        self.ada_pool = ada_pool
        if self.ada_pool:
            self.adapool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):

        x = self.Conv3d_1(x)
        # x = self.Conv3d_2(x)
        x = self.batchnorm(x)

        if self.max_pool:
            x = self.maxpool(x)

        if self.ada_pool:
            x = self.adapool(x)

        return F.relu(x, inplace=True)