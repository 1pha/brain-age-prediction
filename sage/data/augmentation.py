import torch
import monai.transforms as mt

SPATIAL_SIZE = (128, 128, 128)

def mask_transform(spatial_size: tuple = SPATIAL_SIZE,
                   mask: torch.Tensor = None,
                   prob: float = 1.0) -> mt.transform:
    """ Checks if mask is valid when given,
    and return monai.transforms masking out masks
    Otherwise, identity transform will be returned. """
    if mask is not None:
        assert mask.squeeze().shape == spatial_size
        return mt.RandLambda(lambda t: t * mask.to(t.device), prob=prob)
    else:
        return mt.RandLambda(lambda t: t, prob=prob)


def no_augment(spatial_size: tuple = SPATIAL_SIZE,
               mask: torch.Tensor = None):
    apply_mask: mt.transform = mask_transform(spatial_size=spatial_size, mask=mask)
    return mt.Compose([
        mt.Resize(spatial_size=spatial_size),
        mt.ScaleIntensity(),
        apply_mask,
        mt.Lambda(lambda t: t.unsqueeze(dim=1)),
    ])


def augment(spatial_size: tuple = SPATIAL_SIZE,
            mask: torch.Tensor = None):
    apply_mask: mt.transform = mask_transform(spatial_size=spatial_size, mask=mask)
    return mt.Compose([
        mt.Resize(spatial_size=spatial_size),
        mt.ScaleIntensity(),
        apply_mask,
        mt.RandAdjustContrast(prob=0.1, gamma=(0.5, 2.0)),
        mt.RandCoarseDropout(holes=20, spatial_size=8, prob=0.4, fill_value=0.),
        mt.RandAxisFlip(prob=0.5),
        mt.RandZoom(prob=0.4, min_zoom=0.9, max_zoom=1.4, mode="trilinear"),
        mt.Lambda(lambda t: t.unsqueeze(dim=1)),
    ])


def augment_mild(spatial_size: tuple = SPATIAL_SIZE,
                 mask: torch.Tensor = None):
    apply_mask: mt.transform = mask_transform(spatial_size=spatial_size, mask=mask)
    return mt.Compose([
        mt.Resize(spatial_size=spatial_size),
        mt.ScaleIntensity(),
        apply_mask,
        mt.RandAxisFlip(prob=0.5),
        mt.Lambda(lambda t: t.unsqueeze(dim=1)),
    ])

    
def augment_mask(spatial_size: tuple = SPATIAL_SIZE,
                 mask: torch.Tensor = None):
    """ Using masking as augmentation probabilistically """
    apply_mask: mt.transform = mask_transform(spatial_size=spatial_size, mask=mask, prob=0.7)
    return mt.Compose([
        mt.Resize(spatial_size=spatial_size),
        mt.ScaleIntensity(),
        apply_mask,
        mt.RandAdjustContrast(prob=0.1, gamma=(0.5, 2.0)),
        mt.RandCoarseDropout(holes=20, spatial_size=8, prob=0.4, fill_value=0.),
        mt.RandAxisFlip(prob=0.5),
        mt.RandZoom(prob=0.4, min_zoom=0.9, max_zoom=1.4, mode="trilinear"),
        mt.Lambda(lambda t: t.unsqueeze(dim=1)),
    ])
