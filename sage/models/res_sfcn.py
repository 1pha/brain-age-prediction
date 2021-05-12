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

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     padding=1,
                     bias=False)

class Conv1Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation=None, batchnorm=True):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True) if activation is None else activation
        self.stride = stride
        self.batchnorm = batchnorm

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out

class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, activation=None, batchnorm=True):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True) if activation is None else activation
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = partial(self._downsample_basic_block, planes=planes, stride=stride) if downsample is not None else None
        self.stride = stride
        self.batchnorm = batchnorm

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), abs(planes - out.size(1)), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.batchnorm: out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batchnorm: out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            out += residual

        out = self.relu(out)

        return out

class ResSFCN(nn.Module):

    def __init__(self, cfg=None):
        super().__init__()

        layers = cfg.layers if cfg is not None else [4, 8, 16, 32, 32, 16]
        batchnorm = cfg.batchnorm

        feature_extractor = [
            ResidualBlock(1, layers[0], batchnorm=batchnorm),
            ResidualBlock(layers[0], layers[0], stride=2, batchnorm=batchnorm)
        ]
 
        for _prev, _next in zip(layers[:-1], layers[1:-1]):

            if cfg.double:
                feature_extractor.append(ResidualBlock(_prev, _next, batchnorm=batchnorm))
                feature_extractor.append(ResidualBlock(_next, _next, stride=2, batchnorm=batchnorm))

            else:
                feature_extractor.append(ResidualBlock(_prev, _next, stride=2, batchnorm=batchnorm))

        feature_extractor.append(Conv1Block(layers[-2], layers[-1], stride=1))
        self.feature_extractor = nn.Sequential(*feature_extractor)

        self.avgpool = nn.AdaptiveAvgPool3d((2, 2, 2))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(layers[-1] * 8, 1))
        ]))

    def forward(self, x):

        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__=="__main__":

    class CFG:
        layers = [32, 64, 128, 256, 256, 64]
        batchnorm = True
        double = False
    cfg = CFG()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResSFCN(cfg=cfg).to(device)
    print(model)
    print(summary(model, input_size=(1, 96, 96, 96)))

    # sample = torch.zeros((2, 1, 96, 96, 96)).to(device)
    # print(model(sample).squeeze(1))
