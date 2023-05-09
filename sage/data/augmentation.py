import torch
import monai.transforms as mt


def mask_transform(spatial_size: tuple = (96, 96, 96),
                   mask: torch.Tensor = None) -> mt.transform:
    """ Checks if mask is valid when given,
    and return monai.transforms masking out masks
    Otherwise, identity transform will be returned. """
    if mask is not None:
        assert mask.squeeze().shape == spatial_size
        return mt.Lambda(lambda t: t * mask.to(t.device))
    else:
        return mt.Lambda(lambda t: t)


def no_augment(spatial_size: tuple = (96, 96, 96),
               mask: torch.Tensor = None):
    apply_mask: mt.transform = mask_transform(spatial_size=spatial_size, mask=mask)
    return mt.Compose([
        mt.Resize(spatial_size=spatial_size),
        mt.ScaleIntensity(),
        apply_mask,
        mt.Lambda(lambda t: t.unsqueeze(dim=1)),
    ])


def augment(spatial_size: tuple = (96, 96, 96),
            mask: torch.Tensor = None):
    apply_mask: mt.transform = mask_transform(spatial_size=spatial_size, mask=mask)
    return mt.Compose([
        mt.Resize(spatial_size=spatial_size),
        mt.ScaleIntensity(),
        apply_mask,
        mt.RandAdjustContrast(prob=0.1, gamma=(0.5, 3.5)),
        mt.RandCoarseDropout(holes=20, spatial_size=8, prob=0.4, fill_value=0.),
        mt.RandAxisFlip(prob=0.5),
        mt.RandZoom(prob=0.5, min_zoom=0.9, max_zoom=1.3, mode="trilinear"),
        mt.Lambda(lambda t: t.unsqueeze(dim=1)),
    ])
