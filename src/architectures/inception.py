import torch
import torch.nn as nn


class Inception(nn.Module):

    def __init__(self, task_type):
        super(Inception, self).__init__()
        self.task_type = task_type

        self.branch1x1x1 = nn.Sequential(
            nn.Conv3d(1, 16, 1, 1)
        )

        self.branch3x3x3 = nn.Sequential(
            nn.Conv3d(1, 16, 3),
        )