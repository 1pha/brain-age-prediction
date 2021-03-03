import matplotlib.pyplot as plt
from skimage.transform import resize

import torch
import torch.nn.functional as F
from torch.autograd import Function

class CAM:
    def __init__(self, cfg, model):
        self.gradient = []

        self.cfg = cfg
        self.model = model

        self.resized_cams = None

    def register_hooks(self, model_name=None):

        if model_name is None:
            model_name = self.cfg.model_name

        self.hooks = dict()
        if model_name == 'resnet':
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
            self.conv_layers = {
                'conv1': self.model.conv1,
                'layer1_conv1': self.model.layer1[0].conv1,
                'layer1_conv2': self.model.layer1[0].conv2,
                'layer2_conv1': self.model.layer2[0].conv1,
                'layer2_conv2': self.model.layer2[0].conv2,
                'layer3_conv1': self.model.layer3[0].conv1,
                'layer3_conv2': self.model.layer3[0].conv2,
                'layer4_conv1': self.model.layer4[0].conv1,
                'layer4_conv2': self.model.layer4[0].conv2,
            }


        elif model_name == 'CNN':
            self.conv_layers = [
                self.model.layer[0],
                self.model.layer[3],
                self.model.layer[7],
                self.model.layer[11],
            ]

        else:
            raise NotImplementedError

        for i, layer in enumerate(self.conv_layers.values()):
            self.hooks[i] = layer.register_backward_hook(self.save_gradient)
        
    def save_gradient(self, *args):
        grad_input = args[1]
        grad_output= args[2]
        self.gradient.append(grad_output[0])
      
    def get_gradient(self, idx):
        return self.gradient[idx]
    
    def remove_hook(self):
        for layer in self.hooks.values():
            layer.remove()
            
    def normalize_cam(self, x):
        x = 2*(x-torch.min(x))/(torch.max(x)-torch.min(x)+1e-8)-1
        x[x<torch.max(x)]=-1
        return x
    
    def visualize(self, slc=48):

        self.resized_cams = self.resizing() if self.resized_cams is None else self.resized_cams
        if self.cfg.model_name == 'resnet':

            fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))
            for idx, (cam_, layer) in enumerate(zip(self.resized_cams, self.conv_layers.keys())):

                row, col = idx // 3, idx % 3
                ax[row, col].imshow(cam_[:, slc, :])
                ax[row, col].set_title(layer)

        elif self.cfg.model_name == 'CNN':

            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
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
        
        self.remove_hook()
        return F.relu(cam)

    def cam_over_layers(self):
        self.cams = [self.get_cam(idx) for idx, _ in enumerate(self.gradient)]
        return self.cams

    def resizing(self):
        self.resized_cams = [resize(c.cpu().numpy(), output_shape=self.cfg.preprocess['resize'])
                            for c in self.cams]
        return self.resized_cams
        

if __name__=="__main__":

    pass