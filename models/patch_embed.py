from torch import nn as nn
from itertools import repeat
import collections.abc
import torch
import math
from torch.nn.init import trunc_normal_

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
    
class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=27, patch_size=3, stride=2, in_chans=64, embed_dim=128):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = 196
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=patch_size, stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2)),
            nn.GELU(),
            nn.BatchNorm2d(embed_dim // 2),
            nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=1, stride=1),
            nn.GELU(),
            nn.BatchNorm2d(embed_dim // 2),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=1, stride=1),
            nn.GELU(),
            nn.BatchNorm2d(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1),
            nn.GELU(),
            nn.BatchNorm2d(embed_dim)
            )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x

## OverlapPatchEmbed example
# x = torch.randn(32, 64, 27, 27)
# model = OverlapPatchEmbed()
# out, H, W = model(x)
# print(out.shape, H, W) # torch.Size([32, 196, 128]) 14 14

# ## patchify for OverlapPatchEmbed
# def patchify(self, imgs):
#     """
#     imgs: (N, C, H, W) H = W = 27
#     x: (N, L, patch_size**2 *3)
#     """
#     p = self.patch_embed.patch_size[0]
    
#     x = nn.Unfold(imgs, kernel_size=p, stride=self.stride, padding=(p // 2, p // 2))

#     return x
 
# ## unpatchify for OverlapPatchEmbed
# def unpatchify(self, x):
#     """
#     x: (N, L, patch_size**2 *hid_chans)
#     imgs: (N, C, H, W)
#     """
#     p = self.patch_embed.patch_size[0]
#     h = self.h
#     w = self.w
#     imgs = nn.Fold(x, (h * p, w * p), kernel_size=p, stride=self.stride, padding=(p // 2, p // 2))

#     return imgs

# x = torch.randn(32, 30, 27, 27)
# conv = nn.Unfold(kernel_size=3, stride=3 //2 + 1, padding=(3 // 2, 3 // 2))
# out = conv(x)
# print(out.shape) ##torch.Size([32, 27, 196]) target shape torch.Size([32, 196, 27])
# fold = nn.Fold(output_size=(27, 27), kernel_size=3, stride=3 //2 + 1, padding=(3 // 2, 3 // 2))
# out = fold(out)
# print(out.shape) ##torch.Size([32, 3, 27, 27]) target shape torch.Size([32, 27, 27, 3])