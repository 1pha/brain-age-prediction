import torch.nn as nn


class NKRegressor(nn.Module):
    def __init__(self, cfg=None):
        super(NKRegressor, self).__init__()

        if cfg is None:
            init_node = 96
        else:
            self.cfg = cfg
            init_node = self.cfg.init_node

        self.regressor = nn.Sequential(
            nn.Linear(init_node, init_node // 2),
            nn.ReLU(),
            nn.Linear(init_node // 2, init_node // 4),
            nn.ReLU(),
            nn.Linear(init_node // 4, 1),
        )

    def forward(self, x):
        out = self.regressor(x)
        return out


class NKDomainPredictor(nn.Module):
    def __init__(self, cfg=None):
        super(NKDomainPredictor, self).__init__()

        if cfg is None:
            num_dbs = 2
        else:
            self.cfg = cfg
            init_node = self.cfg.init_node
            num_dbs = self.cfg.num_dbs

        self.domain = nn.Sequential(
            nn.Linear(init_node, init_node // 2),
            nn.ReLU(),
            nn.Dropout3d(p=0.2),
            nn.Linear(init_node // 2, init_node // 4),
            nn.ReLU(),
            nn.Linear(init_node // 4, num_dbs),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        out = self.domain(x)
        return out


load_predictors = {
    "nkregressor": NKRegressor,
    "nkdomainpredictor": NKDomainPredictor,
}
