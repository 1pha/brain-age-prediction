import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Any, Optional, Tuple, List



class Inception(nn.Module):

    def __init__(self, in_channels: int, pool_features: int):
        super(Inception, self).__init__()

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


class BasicConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):

        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, eps=.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


