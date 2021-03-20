from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, activation=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True) if activation is None else activation
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = partial(self._downsample_basic_block, planes=planes, stride=stride)
        self.stride = stride

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Residual(nn.Module):

    def __init__(self, cfg=None):
        super().__init__()

        layers = cfg.layers if cfg is not None else [4, 8, 16, 32]
        self.feature_extractor = nn.Sequential(
            BasicBlock(1, layers[0]),
            BasicBlock(layers[0], layers[0], stride=2),

            BasicBlock(layers[0], layers[1]),
            BasicBlock(layers[1], layers[1], stride=2),

            BasicBlock(layers[1], layers[2]),
            BasicBlock(layers[2], layers[2], stride=2),

            BasicBlock(layers[2], layers[3]),
            BasicBlock(layers[3], layers[3], stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool3d((2, 2, 2))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(256, 1))
        ]))

    def forward(self, x):

        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__=="__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = Residual().to(device)
    print(summary(model, input_size=(1, 96, 96, 96)))



        