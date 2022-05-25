from functools import reduce

import torch.nn as nn
from timm.models.layers import trunc_normal_

from .layers import to_3tuple


class PatchEmbed3D(nn.Module):
    """Image to Patch Embedding, from timm"""

    def __init__(self, img_size=96, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()

        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)

        num_patches = reduce(
            lambda x, y: x * y, [i // p for i, p in zip(img_size, patch_size)]
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.apply(self._init_weights)

    def forward(self, x):
        B, C, H, W, D = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1] and D == self.img_size[2]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
