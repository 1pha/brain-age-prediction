import os
import collections
from ..config import edict2dict
import torch
from torchsummary import summary

from .naive_models.dinsdale import *
from .naive_models.levakov_96 import *
from .naive_models.resnet import *
from .naive_models.sequential import *
from .naive_models.sfcn import *
from .naive_models.vanilla import *
from .naive_models.residual_vanilla import *
from .naive_models.res_sfcn import *

from .unlearning.vanilla_dinsdale import VanillaConv
from .unlearning.resnet import load_resnet
from .unlearning.predictors import *


def load_model(cfg=None, gpu=True, verbose=True):
    
    model_name = cfg.model_name
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Model {model_name.capitalize()} is selected.')

    if model_name == 'resnet' or model_name == 'resnet_no_maxpool':
        
        opt = Option()
        model = generate_model(model_depth=opt.model_depth,
                                    n_classes=opt.n_classes,
                                    n_input_channels=opt.n_input_channels,
                                    shortcut_type=opt.shortcut_type,
                                    conv1_t_size=opt.conv1_t_size,
                                    conv1_t_stride=opt.conv1_t_stride,
                                    no_max_pool=opt.no_max_pool,
                                    widen_factor=opt.resnet_widen_factor,
                                    start_channels=cfg.start_channels)

        if model_name == 'resnet_no_maxpool':
            model.no_max_pool = True

    elif model_name == 'levakov':
        model = Levakov(task_type='age')

    elif model_name == 'dinsdale':
        model = Dinsdale(1, 1, 2)

    elif model_name == 'sfcn':
        model = SFCN(cfg)

    elif model_name == 'vanilla':
        model = Vanilla3d(cfg)

    elif model_name == 'vanilla_residual':
        model = Residual(cfg)

    elif model_name == 'vanilla_residual_past':
        model = ResidualPast(cfg)

    elif model_name == 'res_sfcn':
        model = ResSFCN(cfg)

    else: return None

    if gpu:
        model.to(device)
        
    if verbose:
        print(summary(model, input_size=(1, 96, 96, 96)))
    
    return model, device


ENCODERS = {
    'vanillaconv': VanillaConv,
    'resnet': load_resnet,
}

PREDICTORS = {
    'nkregressor': NKRegressor,
    'nkdomainpredictor': NKDomainPredictor,
}

def num_params(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_unlearn_models(cfg):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Use {device} as a device.')

    encoder = ENCODERS[cfg.encoder.name](cfg.encoder).to(device)
    vector_size = encoder(torch.zeros((2, 1, 96, 96, 96)).to(device)).shape
    assert len(vector_size) == 2 # It should be 1-dim vector with batch (=2dim)
    print(f"Output from encoder is {vector_size[1]}.")
    cfg.regressor.init_node = vector_size[1]
    cfg.domainer.init_node  = vector_size[1]
    regressor = PREDICTORS[cfg.regressor.name](cfg.regressor).to(device)
    domainer  = PREDICTORS[cfg.domainer.name](cfg.domainer).to(device)

    cfg.encoder.num_params = num_params(encoder)
    cfg.regressor.num_params = num_params(regressor)
    cfg.domainer.num_params = num_params(domainer)
    cfg.num_params = sum(cfg.encoder.num_params + cfg.regressor.num_params + cfg.domainer.num_params)

    return (encoder, regressor, domainer), device


def save_checkpoint(states, model_name, model_dir='./models/'):

    print('Saving ...')

    # MAKE DIRECTORY
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  

    # STATES - SINGLE
    if not isinstance(states, dict):

        # MODEL IS ALONE and NOT STATE_DICT
        if isinstance(states, nn.Module):
            states = states.state_dict()

        # MODEL IS ALONE and STATE_DICT
        elif isinstance(states, collections.OrderedDict):
            pass # THEN ITS STATE_DICT

        torch.save(states, os.path.join(model_dir, model_name))

    # STATES - PLURAL
    elif isinstance(states, dict):

        for name, s in states.items():

            if isinstance(s, nn.Module):
                s = s.state_dict()
            
            _model_dir = os.path.join(model_dir, name)
            torch.save(s, os.path.join(_model_dir, model_name))


if __name__ == "__main__":

    model = 'resnet'
    model, device = load_model(model)