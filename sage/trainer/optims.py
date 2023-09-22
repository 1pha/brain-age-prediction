import torch
import torch.nn as nn


class confusion_loss(nn.Module):
    def __init__(self, task=0):
        super(confusion_loss, self).__init__()
        self.task = task

    def forward(self, x, target):
        # We only care about x
        log = torch.log(x)
        log_sum = torch.sum(log, dim=1)
        normalised_log_sum = torch.div(log_sum, x.size()[1])
        loss = torch.mul(torch.sum(normalised_log_sum, dim=0), -1)
        return loss


class KLDivLossSFCN(nn.Module):
    """Returns K-L Divergence loss (SFCN, https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain)
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.loss_fn = nn.KLDivLoss(reduction='sum')

    def forward(self, x, y):
        y += self.eps
        n = y.shape[0]
        loss = self.loss_fn(x, y) / n
        return loss
