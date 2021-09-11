import numpy as np
import torch


class SmoothGrad:
    def __init__(self, cfg, model, stdev=0.1, n_samples=25):

        """
        Parameters
        ----------
        cfg: easydict
            Configuration file

        model: torch.nn.Module
            Use 'load_weight' method in order to use pre-trained model

        stdev: float
            Amplitude of noise given to samples

        n_samples: int
            Trials to make noised input
        """

        self.model = model
        self.stdev = stdev
        self.n_samples = n_samples
        self.cfg = cfg

    def __call__(self, x, y, normalize=True, verbose=False):

        """
        Parameters
        ----------
        x: torch.FloatTensor with 5-dim (1, 1, H, W, D)
        y: torch.IntTensor witih 1-dim (age,)
        normalize: bool, default=True
            normalize saliency map through min/max
        verbose: bool | int
            if True or > 1, prints a output from the model
            if 2, prints which augmentation was used.

        """

        x.requires_grad = True
        output = self.model(x).squeeze()
        if verbose:
            print(f"[true] {int(y.data.cpu())}", end=" ")
            print(f"[pred] {float(output.data.cpu()):.3f}")
        output.backward()
        total_gradients = x.grad.data.cpu().numpy()
        x.requires_grad = False

        x = x.data.cpu().numpy()
        stdev = self.stdev * (np.max(x) - np.min(x))
        for i in range(self.n_samples):

            noise = np.random.normal(0, stdev, x.shape).astype(np.float32)
            x_plus_noise = torch.Tensor(x + noise)
            x_plus_noise = x_plus_noise.to(self.cfg.device)
            x_plus_noise.requires_grad = True
            output = self.model(x_plus_noise).squeeze()
            if verbose:
                print(f"{i}th: {float(output.data.cpu()):.3f}")
            output.backward()

            grad = x_plus_noise.grad.data.cpu().numpy()
            total_gradients += grad

        avg_gradients = total_gradients[0, ...] / self.n_samples

        return self.normalize(avg_gradients) if normalize else avg_gradients

    def normalize(self, vismap, eps=1e-4):

        numer = vismap - np.min(vismap)
        denom = (vismap.max() - vismap.min()) + eps
        vismap = numer / denom

        return vismap if len(vismap.shape) < 4 else vismap[0]
