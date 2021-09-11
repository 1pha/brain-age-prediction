import numpy as np
import torchio as tio


class AugGrad:
    def __init__(self, cfg, model, n_samples=25):

        """
        Parameters
        ----------
        cfg: easydict
            Configuration file

        model: torch.nn.Module
            Use 'load_weight' method in order to use pre-trained model

        n_samples: int
            Trials to make augmented input
        """

        self.model = model
        self.n_samples = n_samples
        self.cfg = cfg
        self.augmentation = cfg.augmentation

        scales, degrees = cfg.aug_intensity["affine"]
        num_control_points, max_displacement = cfg.aug_intensity["elastic_deform"]
        self.transform = {
            "affine": tio.RandomAffine(scales=scales, degrees=degrees),
            "flip": tio.RandomFlip(axes=["left-right"]),
            "elastic_deform": tio.RandomElasticDeformation(
                num_control_points=num_control_points, max_displacement=max_displacement
            ),
        }

        p = list(self.augmentation.values())
        norm = sum(p)
        self.p = list(map(lambda x: x / norm, p))

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
        for sample in range(self.n_samples):

            aug_choice = np.random.choice(list(self.transform.keys()), p=self.p)
            x_aug = self.transform[aug_choice](x[0].cpu()).to(self.cfg.device)[
                None, ...
            ]
            x_aug.requires_grad = True
            output = self.model(x_aug).squeeze()
            if verbose == 2:
                print(f"{sample}th [{aug_choice}]: {float(output.data.cpu()):.3f}")
            (output - y.to(self.cfg.device)).backward()

            total_gradients += x_aug.grad.data.cpu().numpy()

        avg_gradients = total_gradients[0, ...] / self.n_samples

        return self.normalize(avg_gradients) if normalize else avg_gradients

    def normalize(self, vismap, eps=1e-4):

        numer = vismap - np.min(vismap)
        denom = (vismap.max() - vismap.min()) + eps
        vismap = numer / denom

        return vismap if len(vismap.shape) < 4 else vismap[0]
