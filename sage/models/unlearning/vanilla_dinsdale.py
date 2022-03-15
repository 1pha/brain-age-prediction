from functools import reduce

import torch
import torch.nn as nn

from torchsummary import summary


"""
This module should contain models
    1. Takes 4D Brain Tensor as input
    2. that outputs 1-dim EMBEDDED vectors
        - this 1d embed vector will be fed to models from predictors
"""


class PoolBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.pool(x)

        return x


class StrideBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm3d(num_features=out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


def get_inplanes(start_channels=16):
    if start_channels == 8:
        return [8, 16, 32, 64, 128]

    elif start_channels == 16:
        return [16, 32, 64, 128, 256]

    elif start_channels == 32:
        return [32, 64, 128, 256, 512]

    elif start_channels == 64:
        return [64, 128, 256, 512, 1024]


class VanillaConv(nn.Module):
    def __init__(self, cfg=None, start_channels=16):
        super().__init__()

        self.cfg = cfg

        layers = get_inplanes(start_channels)
        self.layers = layers
        self.feature_extractor = nn.Sequential(
            StrideBlock(1, layers[0]),
            StrideBlock(layers[0], layers[1]),
            StrideBlock(layers[1], layers[2]),
            StrideBlock(layers[2], layers[3]),
            StrideBlock(layers[3], layers[4]),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):

        x = self.feature_extractor(x)
        if reduce(lambda x, y: x * y, x.shape[1:]) // self.layers[-1] > 2**3:
            x = self.avgpool(x)
        x = x.reshape(x.size(0), -1).contiguous()

        return x

    @property
    def conv_layers(self):

        conv_layers = []
        for block in self.feature_extractor:
            conv_layers.append(block.conv1)
            conv_layers.append(block.conv2)

        return conv_layers


if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = VanillaConv().to(device)
    print(model)
    print(summary(model, input_size=(1, 96, 96, 96)))

    # sample = torch.zeros((2, 1, 96, 96, 96)).to(device)
    # print(model(sample).shape)
