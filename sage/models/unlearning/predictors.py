import torch.nn as nn

class NKRegressor(nn.Module):

    def __init__(self, cfg=None):
        super(NKRegressor, self).__init__()

        if cfg is None:
            init_node = 96
        else:
            self.cfg = cfg
            init_node = self.cfg.init_node

        self.regressor = nn.Sequential()
        self.regressor.add_module('r_fc2', nn.Linear(init_node, 32))
        self.regressor.add_module('r_relu2', nn.ReLU(True))
        self.regressor.add_module('r_pred', nn.Linear(32, 1))

    def forward(self, x):
        regression = self.regressor(x)
        return regression


class NKDomainPredictor(nn.Module):
    
    def __init__(self, cfg=None):
        super(NKDomainPredictor, self).__init__()

        if cfg is None:
            num_dbs = 96
        else:
            self.cfg = cfg
            num_dbs = self.cfg.num_dbs
        self.num_dbs = num_dbs
        self.domain = nn.Sequential()
        self.domain.add_module('d_fc2', nn.Linear(96, 32))
        self.domain.add_module('d_relu2', nn.ReLU(True))
        self.domain.add_module('r_dropout', nn.Dropout3d(p=0.2))
        self.domain.add_module('d_fc3', nn.Linear(32, num_dbs))
        self.domain.add_module('d_pred', nn.Softmax(dim=1))

    def forward(self, x):
        domain_pred = self.domain(x)
        return domain_pred


load_predictors = {
    'nkregressor': NKRegressor,
    'nkdomainpredictor': NKDomainPredictor,
}