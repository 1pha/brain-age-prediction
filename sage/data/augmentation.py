import monai.transforms as mt
import torch


def no_augment():
    return mt.Compose([
        mt.Resize(spatial_size=(96, 96, 96)),
        mt.ScaleIntensity(),
        mt.Lambda(lambda t: t.unsqueeze(dim=1)),
    ])
    

def augment(spatial_size: tuple = (96, 96, 96),
            mask: torch.Tensor = None):
    if mask is not None:
        assert mask.squeeze().shape == spatial_size
    return mt.Compose([
        mt.Resize(spatial_size=spatial_size),
        mt.ScaleIntensity(),
        mt.RandAdjustContrast(prob=0.1, gamma=(0.5, 3.5)),
        mt.RandCoarseDropout(holes=20, spatial_size=8, prob=0.4, fill_value=0.),
        mt.RandAxisFlip(prob=0.5),
        mt.RandZoom(prob=0.5, min_zoom=0.9, max_zoom=1.3, mode="trilinear"),
        mt.Lambda(lambda t: t.unsqueeze(dim=1)),
    ])
