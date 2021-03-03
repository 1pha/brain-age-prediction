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
            conv_layers = [
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

            for i, layer in enumerate(conv_layers):
                self.hooks[i] = layer.register_backward_hook(self.save_gradient)
        
        else:
            raise NotImplementedError
            
        
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

        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))

        self.resized_cams = self.resizing() if self.resized_cams is None else self.resized_cams
        for idx, cam_ in enumerate(self.resized_cams):

            row, col = idx // 3, idx % 3
            ax[row, col].imshow(cam_[:, slc, :], cmap='gray')
                       
        
    
    def get_cam(self, idx):
        '''
        Get CAM = ReLU(sum(alpha*grad_cam))
        '''
        grad = self.gradient[idx]
        alpha = torch.sum(grad,  dim=4, keepdim=True)
        alpha = torch.sum(alpha, dim=3, keepdim=True)
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