import torch
from torch.autograd import Function

class CAM():
    def __init__(self,model):
        self.gradient = []
        # self.h = model.module.layer[-1].register_backward_hook(self.save_gradient)
        # self.h = model.layer4[0].conv1.register_backward_hook(self.save_gradient)
        self.h = model.conv1.register_backward_hook(self.save_gradient)

        
    def save_gradient(self,*args):
        grad_input = args[1]
        grad_output= args[2]
        self.gradient.append(grad_output[0])
      
    def get_gradient(self,idx):
        return self.gradient[idx]
    
    def remove_hook(self):
        self.h.remove()
            
    def normalize_cam(self,x):
        x = 2*(x-torch.min(x))/(torch.max(x)-torch.min(x)+1e-8)-1
        x[x<torch.max(x)]=-1
        return x
    
    def visualize(self,cam_img,img_var):
        cam_img = resize(cam_img.cpu().data.numpy(),output_shape=(28,28))
        x = img_var[0,:,:].cpu().data.numpy()

        plt.subplot(1,3,1)
        plt.imshow(cam_img)

        plt.subplot(1,3,2)
        plt.imshow(x,cmap="gray")

        plt.subplot(1,3,3)
        plt.imshow(x+cam_img)
        plt.show()
    
    def get_cam(self,idx):
        grad = self.get_gradient(idx)
        alpha = torch.sum(grad,dim=3,keepdim=True)
        alpha = torch.sum(alpha,dim=2,keepdim=True)
        
        cam = alpha[idx]*grad[idx]
        cam = torch.sum(cam,dim=0)
        cam = self.normalize_cam(cam)
        
        self.remove_hook()
        return cam

class GuidedBackpropRelu(Function):
    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx,grad_output):
        input = ctx.saved_tensors[0]
        grad_input = grad_output.clone()
        grad_input[grad_input<0] = 0
        grad_input[input<0]=0
        return grad_input
     

class GuidedReluModel:
    def __init__(self,model,to_be_replaced,replace_to):
        self.model = model
        self.to_be_replaced = to_be_replaced
        self.replace_to = replace_to
        self.layers=[]
        self.output=[]
        
        for m in self.model.modules():
            if isinstance(m,self.to_be_replaced):
                self.layers.append(self.replace_to )
                #self.layers.append(m)
            elif isinstance(m,nn.Conv2d):
                self.layers.append(m)
            elif isinstance(m,nn.BatchNorm2d):
                self.layers.append(m)
            elif isinstance(m,nn.Linear):
                self.layers.append(m)
            elif isinstance(m,nn.AvgPool2d):
                self.layers.append(m)
                
        for i in self.layers:
            print(i)
        
    def reset_output(self):
        self.output = []
    
    def hook(self,grad):
        out = grad[:,0,:,:].cpu().data#.numpy()
        print("out_size:",out.size())
        self.output.append(out)
        
    def visualize(self,idx,origina_img):
        grad = self.output[0][idx]
        x = origina_img[idx].cpu().data.numpy()[0]
        
        plt.subplot(1,2,1)
        plt.imshow(grad,cmap="gray")
        plt.subplot(1,2,2)
        plt.imshow(x,cmap="gray")
        plt.show()
        
    def forward(self,x):
        out = x 
        out.register_hook(self.hook)
        for i in self.layers[:-3]:
            out = i(out)
        out = out.view(out.size()[0],-1)
        for j in self.layers[-3:]:
            out = j(out)
        return out


if __name__=="__main__":

    # How to use
    # 1. Make Inference (e.g. outputs = mode.forward(x))
    # 2. 
    pass