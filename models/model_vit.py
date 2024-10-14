import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block
from models.patch_embed import OverlapPatchEmbed
from models import *
from models.model_mae import get_1d_sincos_pos_embed,get_2d_sincos_pos_embed,spe_PatchEmbed

class spa_vit(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=16, num_classes=1000, in_chans=3, hid_chans = 128,
                 embed_dim=1024, depth=24, num_heads=16,drop_rate=0.,attn_drop_rate=0., drop_path_rate=0.,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, global_pool = False):
        super().__init__()
        self.patch_size = patch_size

        self.dimen_redu = nn.Sequential(
            nn.Conv2d(in_chans, hid_chans, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hid_chans),
            nn.ReLU(),
                           
            nn.Conv2d(hid_chans, hid_chans, 1, 1, 0, bias=True),
            nn.BatchNorm2d(hid_chans),
            nn.ReLU(),
            )
        self.hid_chans = hid_chans
        self.dimen_expa = nn.Conv2d(hid_chans, in_chans, kernel_size=1, stride=1, padding=0, bias=True)

        self.patch_embed = OverlapPatchEmbed(img_size, patch_size,2, hid_chans , embed_dim, )
        # self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=hid_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio,attn_drop=attn_drop_rate, drop_path=drop_path_rate, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        self.head = nn.Linear(embed_dim, num_classes, bias=True)
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm
    def initialize_weights(self):

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.dimen_redu(x)
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x 

# class spe_PatchEmbed2d(nn.Module):
#     """ 1D signal to Patch Embedding
#     """
#     def __init__(self, in_chans=1, spectral_size = 103, embed_dim=32, norm_layer=None, flatten=True, patch_size = 12,cov_patch_size = 3):
#         super().__init__()
#         self.flatten = flatten 
#         self.patch_size = patch_size
#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=cov_patch_size,stride=cov_patch_size)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
#         self.num_patches =  np.ceil((patch_size/cov_patch_size)*(patch_size/cov_patch_size)).astype(np.int32)
#     def forward(self, x):
#         target_shape = (x.size(0), self.patch_size, self.patch_size, 1)
#         num_elements = target_shape[1] * target_shape[2]
#         padding_needed = num_elements - x.shape[2]
#         if padding_needed > 0:
#             # 初始化padded_x为正确的形状
#             padded_x = torch.zeros(x.size(0), x.shape[1], num_elements, dtype=x.dtype, device=x.device)
#             # 正确填充x的数据到padded_x
#             padded_x[:, :, :x.shape[2]] = x
#         else:
#             padded_x = x
#         # 重塑padded_x以匹配目标形状
#         padded_x = padded_x.view(target_shape)

#         x = padded_x.permute(0, 3, 1, 2)
#         x = self.proj(x)
#         if self.flatten:
#             x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
#         x = self.norm(x)
#         return x
    
class spe_vit(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,num_classes=1000,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, global_pool = False
                 ):
        super().__init__()

        self.h = img_size[0]// patch_size 
        self.w = img_size[1]// patch_size 
        self.in_chans = in_chans
        self.patch_embed = spe_PatchEmbed(spectral_size=in_chans,patch_size=21, embed_dim=embed_dim, flatten=True)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
       
        self.head = nn.Linear(embed_dim, num_classes, bias=True)
        
        
        self.norm_pix_loss = norm_pix_loss
        self.global_pool = False

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches), cls_token=True)

        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x 