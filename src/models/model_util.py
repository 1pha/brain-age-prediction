import os
import torch
from torchsummary import summary
from .dinsdale import *
from .levakov_64 import *
from .levakov_96 import *
from .resnet import *
from .sequential import *
from .sfcn import *
from .vanilla import *

def load_model(model, cfg=None, gpu=True, verbose=True):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Model {model.capitalize()} is selected.')

    if model == 'resnet':
        

        opt = Option()
        model = generate_model(model_depth=opt.model_depth,
                                    n_classes=opt.n_classes,
                                    n_input_channels=opt.n_input_channels,
                                    shortcut_type=opt.shortcut_type,
                                    conv1_t_size=opt.conv1_t_size,
                                    conv1_t_stride=opt.conv1_t_stride,
                                    no_max_pool=opt.no_max_pool,
                                    widen_factor=opt.resnet_widen_factor)

    elif model == 'levakov':
        model = Levakov(task_type='age')

    elif model == 'inception':
        model = Inception3()

    elif model == 'dinsdale':
        model = Dinsdale(1, 1, 2)

    elif model == 'sfcn':
        model = SFCN()

    elif model == 'vanilla':
        model = Vanilla3d(cfg)

    else: return None

    if gpu:
        model.to(device)
        
    if verbose:
        print(summary(model, input_size=(1, 96, 96, 96)))
    
    return model, device

def save_checkpoint(state, model_filename, model_dir='./models/', is_best=False):
    print('Saving ...')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    
    # torch.save(state, os.path.join(model_dir, model_filename))    
    if is_best:
        torch.save(state, os.path.join(model_dir, 'best_' + model_filename))

if __name__ == "__main__":

    model = 'resnet'
    model, device = load_model(model)