import numpy as np

from ..src.config import *
from ..src.models.model_utils import *

class SmoothGrad:
    
    def __init__(self, pretrained_model, cfg, stdev=.1, n_samples=25):
        
        self.pretrained_model = pretrained_model
        self.stdev = stdev
        self.n_samples = n_samples
        self.cfg = cfg
        
    def __call__(self, x, y, verbose=False):
        
        x = x.data.cpu().numpy()
        stdev = self.stdev * (np.max(x) - np.min(x))
        total_gradients = np.zeros_like(x)
        print(f'Y: {y}')
        for i in range(self.n_samples):
            
            noise = np.random.normal(0, stdev, x.shape).astype(np.float32)
            x_plus_noise = torch.Tensor(x + noise)
            x_plus_noise = x_plus_noise.to(self.cfg.device)
            x_plus_noise.requires_grad = True
            output = self.pretrained_model(x_plus_noise).squeeze()
            if verbose:
                print(output)
            output.backward()
            
            grad = x_plus_noise.grad.data.cpu().numpy()
            total_gradients += grad
            
        avg_gradients = total_gradients[0, ...] / self.n_samples
        
        return avg_gradients


if __name__=="__main__":

    # Load Config
    cfg = load_config()

    # Load Model
    cfg.model_name = 'vanilla_residual'
    model = load_model(cfg.model_name, verbose=False, cfg=cfg)

    # Dataloader
    train_dataset = DatasetPlus(cfg, augment=False)
    sample_dl = DataLoader(train_dataset, batch_size=1)

    data = next(iter(sample_dl))

    # Make SmoothGrad Instance
    sgrad = SmoothGrad(model, cfg, stdev=.01)

    # Forward a single brain with age to the SmoothGrad instance
    smooth_grad = sgrad(data[0][None, ...], data[1])