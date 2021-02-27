import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Any, Optional, Tuple, List     


class Inception3(nn.Module):

    def __init__(self, num_classes=1, aux_logits=True,
        transform_input=False, inception_blocks=None, init_weights=None):

        super(Inception3, self).__init__()
        if inception_blocks is None:
            inception_blocks = [BasicConv3d, InceptionA, InceptionB]

        if init_weights is None:
            init_weights = True

        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]

        self.Conv3d_1a_3x3x3 = conv_block(1, 8, kernel_size=3, stride=2)
        self.Conv3d_2a_3x3x3 = conv_block(8, 16, kernel_size=3)
        self.Conv3d_2b_3x3x3 = conv_block(16, 16, kernel_size=3, padding=1)

        self.maxpool1 = nn.MaxPool3d(kernel_size=3, stride=2)
        self.Conv3d_3b_1x1x1 = conv_block(16, 32, kernel_size=1)
        self.Conv3d_4a_3x3x3 = conv_block(32, 32, kernel_size=3)

        self.maxpool2 = nn.MaxPool3d(kernel_size=3, stride=2)
        self.Mixed_5b = inception_a(32, pool_features=32)
        self.Mixed_5c = inception_a(128, pool_features=64)
        self.Mixed_5d = inception_a(160, pool_features=64)

        self.Mixed_6a = inception_b(160)

        self.maxpool3 = nn.MaxPool3d(kernel_size=3, stride=2)

        self.dropout = nn.Dropout()
        self.fc = nn.Linear(224, num_classes)
        
    def forward(self, x):
        x = self.Conv3d_1a_3x3x3(x)
        x = self.Conv3d_2a_3x3x3(x)
        x = self.Conv3d_2b_3x3x3(x)

        x = self.maxpool1(x)

        x = self.Conv3d_3b_1x1x1(x)
        x = self.Conv3d_4a_3x3x3(x)
        x = self.maxpool2(x)

        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)

        x = self.maxpool3(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class InceptionA(nn.Module):

    def __init__(self, in_channels: int, pool_features: int):
        super(InceptionA, self).__init__()

        conv_block = BasicConv3d

        self.branch1x1x1 = conv_block(in_channels, 32, kernel_size=1)

        self.branch5x5x5_1 = conv_block(in_channels, 16, kernel_size=1)
        self.branch5x5x5_2 = conv_block(16, 32, kernel_size=5, padding=2)

        self.branch3x3x3db1_1 = conv_block(in_channels, 16, kernel_size=1)
        self.branch3x3x3db2_1 = conv_block(16, 32, kernel_size=3, padding=1)
        self.branch3x3x3db2_2 = conv_block(32, 32, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x: Tensor) -> List[Tensor]:

        branch1x1x1 = self.branch1x1x1(x)

        branch3x3x3 = self.branch3x3x3db1_1(x)
        branch3x3x3 = self.branch3x3x3db2_1(branch3x3x3)
        branch3x3x3 = self.branch3x3x3db2_2(branch3x3x3)

        branch5x5x5 = self.branch5x5x5_1(x)
        branch5x5x5 = self.branch5x5x5_2(branch5x5x5)

        branch_pool = F.avg_pool3d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1x1, branch5x5x5, branch3x3x3, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionB(nn.Module):

    def __init__(self, in_channels: int):
        super(InceptionB, self).__init__()

        conv_block = BasicConv3d

        self.branch3x3x3 = conv_block(in_channels, 32, kernel_size=3, stride=2)

        self.branch3x3x3db1_1 = conv_block(in_channels, 16, kernel_size=1)
        self.branch3x3x3db2_1 = conv_block(16, 32, kernel_size=3, padding=1)
        self.branch3x3x3db2_2 = conv_block(32, 32, kernel_size=3, stride=2)


    def _forward(self, x: Tensor) -> List[Tensor]:

        branch3x3x3 = self.branch3x3x3(x)

        branch3x3x3db1 = self.branch3x3x3db1_1(x)
        branch3x3x3db1 = self.branch3x3x3db2_1(branch3x3x3db1)
        branch3x3x3db1 = self.branch3x3x3db2_2(branch3x3x3db1)

        branch_pool = F.max_pool3d(x, kernel_size=3, stride=2)

        outputs = [branch3x3x3, branch3x3x3db1, branch_pool]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class BasicConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):

        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, eps=.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)