import matplotlib.pyplot as plt
from skimage.transform import resize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class CAM:

    __date__ = 'Aug 18, 2021'

    def __init__(self, cfg, model):
        self.gradient = []

        self.cfg = cfg
        self.model = model
        self.model.eval()

        self.resized_cams = None
        self.layer_index = None
        self.make_conv_layers()

    def __call__(self, x, y, layer_index=None):
        
        '''
        x: 5-dim Brain (1, 1, 96, 96, 96): torch.tensor
        y: single float of age: torch.tensor
        '''

        self.register_hooks()
        x, y = x.to(self.cfg.device), y.to(self.cfg.device)
        output = self.model(x)
        print(f'[true] {int(y.data.cpu())}', end=' ')
        print(f'[pred] {float(output.data.cpu()):.3f}')
        output.backward()

        self.cam_over_layers()
        vismap = self.resizing()
        self.remove_hook()

        if layer_index is not None:
            if isinstance(layer_index, int):
                return [vismap[layer_index]]

            if isinstance(layer_index, list):
                return [vismap[idx] for idx in layer_index]
                
        else:
            return vismap

        return vismap

    def make_conv_layers(self, model_name=None):

        if model_name is None:
            model_name = self.cfg.model_name

        self.hooks = dict()
        if hasattr(self.model, 'conv_layers'):
            self.conv_layers = self.model.conv_layers

        else:
            if model_name == 'resnet' or model_name == 'resnet_no_maxpool':
                self.conv_layers = [
                    self.model.conv1,
                    self.model.layer1[0].conv1,
                    self.model.layer1[0].conv2,
                    self.model.layer2[0].conv1,
                    self.model.layer2[0].conv2,
                    self.model.layer3[0].conv1,
                    self.model.layer3[0].conv2,
                    self.model.layer4[0].conv1,
                    self.model.layer4[0].conv2,
                ]

            elif model_name == 'CNN':
                self.conv_layers = [
                    self.model.layer[0],
                    self.model.layer[3],
                    self.model.layer[7],
                    self.model.layer[11],
                ]

            elif model_name == 'vanilla_residual_past':
            
                self.conv_layers = [
                    self.model.feature_extractor[0].conv1,
                    self.model.feature_extractor[0].conv2,
                    self.model.feature_extractor[1].conv1,
                    self.model.feature_extractor[1].conv2,
                    self.model.feature_extractor[2].conv1,
                    self.model.feature_extractor[2].conv2,
                    self.model.feature_extractor[3].conv1,
                    self.model.feature_extractor[3].conv2,
                    self.model.feature_extractor[4].conv1,
                    self.model.feature_extractor[4].conv2,
                    self.model.feature_extractor[5].conv1,
                    self.model.feature_extractor[5].conv2,
                    self.model.feature_extractor[6].conv1,
                    self.model.feature_extractor[6].conv2,
                    self.model.feature_extractor[7].conv1,
                    self.model.feature_extractor[7].conv2,
                ]

            elif model_name == "vanillaconv": # unlearning/vanilla_conv.py
                self.conv_layers = 1 # TODO

            else:
                raise NotImplementedError

    def register_hooks(self, model_name=None):
        for i, layer in enumerate(self.conv_layers):
            self.hooks[i] = layer.register_backward_hook(self.save_gradient)
        
    def save_gradient(self, *args):
        grad_input = args[1]
        grad_output= args[2]
        self.gradient.append(grad_output[0])
      
    def get_gradient(self, idx):
        return self.gradient[idx]
    
    def remove_hook(self):
        self.gradient = []
        for layer in self.hooks.values():
            layer.remove()
            
    def normalize_cam(self, x):
        x_norm = 2 * (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-8) - 1
        x_norm[x_norm < torch.max(x_norm)] = -1
        return x_norm
    
    def visualize(self, slc=48):

        '''
        DEPRECATED
        '''

        self.resized_cams = self.resizing() if self.resized_cams is None else self.resized_cams
        if self.cfg.model_name == 'resnet' or self.cfg.model_name == 'resnet_no_maxpool':

            _, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))
            for idx, (cam_, layer) in enumerate(zip(self.resized_cams, self.conv_layers.keys())):

                row, col = idx // 3, idx % 3
                ax[row, col].imshow(cam_[:, slc, :])
                ax[row, col].set_title(layer)

        else:
            _, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
            for idx, cam_ in enumerate(self.resized_cams):

                row, col = idx // 2, idx % 2
                ax[row, col].imshow(cam_)
                       
    def get_cam(self, idx):

        '''
        Get CAM = ReLU(sum(alpha*grad_cam))
        '''

        grad = self.gradient[idx]
        if self.cfg.model_name == 'resnet':
            alpha = torch.sum(grad,  dim=4, keepdim=True)
            alpha = torch.sum(alpha, dim=3, keepdim=True)
            alpha = torch.sum(alpha, dim=2, keepdim=True)
        
        else:
            alpha = torch.sum(grad,  dim=3, keepdim=True)
            alpha = torch.sum(alpha, dim=2, keepdim=True)
        
        cam = alpha * grad
        cam = torch.sum(cam, dim=0)
        cam = torch.sum(cam, dim=0)
        
        return F.relu(cam)

    def cam_over_layers(self):
        self.cams = [self.get_cam(idx) for idx, _ in enumerate(self.gradient)]
        return self.cams

    def resizing(self):
        self.resized_cams = [resize(c.cpu().numpy(), output_shape=self.cfg.resize)
                            for c in self.cams]
        return self.resized_cams


def run_gradcam(model, data, cfg):
    
    model.eval()
    cam = CAM(cfg, model)
    cam.register_hooks()
    
    x, y = data
    device = next(model.parameters()).device
    output = model.forward(x.to(device))
    output.backward()
    
    cam.cam_over_layers()
    cam.remove_hook()
    return cam.resizing()


class GuidedBackpropRelu(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0] # forwarding 할 때의 입력값
        grad_input = grad_output.clone() # backward 할 때의 입력된 미분 값
        grad_input[grad_input < 0] = 0 # 미분 양인 것만
        grad_input[input < 0] = 0 # 입력도 양인 것만
        return grad_input


class GuidedReluModel(nn.Module):

    def __init__(self, model):
        super(GuidedReluModel, self).__init__()
        self.model = model
        self.output = []
        
    def reset_output(self):
        self.output = []
    
    def hook(self, grad):
        # out = grad[:, 0, :, :].cpu().data#.numpy()
        out = grad.cpu().data
        self.output.append(out)
        
    def get_visual(self):
        grad = self.output[0].squeeze()
        return grad
        
    def forward(self,x):
        x.register_hook(self.hook)
        x = self.model(x)
        return x

def make_gbpmodel(cfg):

    if cfg.model_name == 'resnet' or cfg.model_name == 'resnet_no_maxpool':

        opt = Option()
        opt.activation = GuidedBackpropRelu.apply
        model = generate_model(model_depth=opt.model_depth,
                            n_classes=opt.n_classes,
                            n_input_channels=opt.n_input_channels,
                            shortcut_type=opt.shortcut_type,
                            conv1_t_size=opt.conv1_t_size,
                            conv1_t_stride=opt.conv1_t_stride,
                            no_max_pool=opt.no_max_pool,
                            widen_factor=opt.resnet_widen_factor,
                            activation=opt.activation)


def run_gbp(model, data, cfg):
    
    guide.eval()
    model = make_gbpmodel(cfg)
    
    x, y = data
    device = next(guide.parameters()).device
    x.requires_grad = True
    output = guide.forward(x.to(device))
    output.backward()
    return guide.get_visual()


if __name__=="__main__":

    pass