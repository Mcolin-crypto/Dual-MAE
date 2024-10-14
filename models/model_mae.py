import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block
from models.patch_embed import OverlapPatchEmbed
from models.pos_embed import *

class spa_mae(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, hid_chans = 32,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., drop_rate=0.,attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 cls_hidden_mlp=256, nb_classes=1000, global_pool=False,
                 mlp_depth=2):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE dimensionality reduction/expansion specifics
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
        self.h = img_size[0]// patch_size 
        self.w = img_size[1]// patch_size 
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = OverlapPatchEmbed(img_size, patch_size, 2,hid_chans , embed_dim, )
        # self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=hid_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio,attn_drop=attn_drop_rate, drop_path=drop_path_rate, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * hid_chans, bias=True) # decoder to patch
        
        if cls_hidden_mlp == 0:
            self.cls_head = nn.Linear(embed_dim, nb_classes, bias = False)
        else:
            assert mlp_depth in [2], "mlp depth should be 2"
            if mlp_depth == 2:
                self.cls_head = nn.Sequential(
                    nn.Linear(embed_dim, cls_hidden_mlp),
                    nn.BatchNorm1d(cls_hidden_mlp),
                    nn.ReLU(inplace=True),
                    nn.Linear(cls_hidden_mlp, nb_classes),
                ) 
        hidden_dim = 64
        self.projection_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.ph1 = Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)  
               
        
        # --------------------------------------------------------------------------
        self.global_pool = global_pool
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

        
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)

        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

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
            
    # patchify for OverlapPatchEmbed
    def patchify(self, imgs):
        """
        imgs: (N, C, H, W) H = W = 27
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        
        x = nn.Unfold(kernel_size=p, stride=2, padding=(p // 2, p // 2))(imgs)
        x = x.transpose(1,2)

        return x
    
    ## unpatchify for OverlapPatchEmbed
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *hid_chans)
        imgs: (N, C, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = self.h
        w = self.w
        x = x.transpose(1,2)
        imgs = nn.Fold((h * p, w * p), kernel_size=p, stride=2, padding=(p // 2, p // 2))(x)

        return imgs
    
    # def patchify(self, imgs):
    #     """
    #     imgs: (N, 3, H, W)
    #     x: (N, L, patch_size**2 *3)
    #     """
    #     p = self.patch_embed.patch_size[0]
    #     # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    #     h = imgs.shape[2] // p
    #     w = imgs.shape[3] // p

    #     x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
    #     x = torch.einsum('nchpwq->nhwpqc', x)
    #     x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * imgs.shape[1]))
    #     return x
    

    # def unpatchify(self, x):
    #     """
    #     x: (N, L, patch_size**2 *3)
    #     imgs: (N, 3, H, W)
    #     """
    #     p = self.patch_embed.patch_size[0]
    #     h = self.h
    #     w = self.w
    #     assert h * w == x.shape[1]
        
    #     hid_chans = int(x.shape[2]/(p**2))
    #     x = x.reshape(shape=(x.shape[0], h, w, p, p, hid_chans))
    #     x = torch.einsum('nhwpqc->nchpwq', x)
    #     imgs = x.reshape(shape=(x.shape[0], hid_chans, h * p, w * p))
    #     return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.dimen_redu(x)
        x = self.patch_embed(x)
        
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed
        x1 = self.ph1(x)
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        x = self.unpatchify(x)
        x = self.dimen_expa(x)
        x = self.patchify(x)

        return x, x1
    
    def forward_classification(self, x):
        # x = self.ph3(x)
        if self.global_pool:
            feat = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        else:
            feat = x[:, 0, :]  # with cls token
        logits = self.cls_head(feat)
        return logits
    
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        # loss = loss.mean()

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs,y = None, mask_ratio=0.2):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)        
        # Reconstruction branch
        pred, latent_f = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)

        # Classification branch
        logits = self.forward_classification(latent)
        
        return loss, pred, mask, logits
    
    @staticmethod
    def _extract_class_weights(encoder_q):
        if isinstance(encoder_q, nn.Sequential):
            class_weight = encoder_q[-1].weight.clone().detach()
        else:
            class_weight = encoder_q.weight.clone().detach()
        return class_weight 

class spe_PatchEmbed(nn.Module):
    """ 1D signal to Patch Embedding
    """
    def __init__(self, in_chans=1, spectral_size = 103, embed_dim=32, norm_layer=None, flatten=True, patch_size = 21):
        super().__init__()
        self.flatten = flatten   
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=10)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.num_patches =  np.ceil(spectral_size/patch_size).astype(np.int32)
        self.patch_size = patch_size
        
    def forward(self, x):
        B, C, L = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.transpose(1, 2)  # BCN-> BNC
        x = self.norm(x)
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

class spe_mae(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 cls_hidden_mlp=256, nb_classes=1000, global_pool=False,
                 mlp_depth=2):
        super().__init__()

        self.h = img_size[0]// patch_size 
        self.w = img_size[1]// patch_size 
        self.in_chans = in_chans
        self.patch_embed = spe_PatchEmbed(spectral_size=in_chans,patch_size=21,embed_dim=embed_dim, flatten=True)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, np.ceil(in_chans / num_patches).astype(np.int32), bias=True) # decoder to patch
        # --------------------------------------------------------------------------
        if cls_hidden_mlp == 0:
            self.cls_head = nn.Linear(embed_dim, nb_classes, bias = False)
        else:
            assert mlp_depth in [2], "mlp depth should be 2"
            if mlp_depth == 2:
                self.cls_head = nn.Sequential(
                    nn.Linear(embed_dim, cls_hidden_mlp),
                    nn.BatchNorm1d(cls_hidden_mlp),
                    nn.ReLU(inplace=True),
                    nn.Linear(cls_hidden_mlp, nb_classes),
                ) 
        hidden_dim = 128
        self.projection_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim)
        )
        
        self.prediction_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim)
        )

        self.ph1 = Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        
        self.norm_pix_loss = norm_pix_loss
        self.global_pool = False

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches), cls_token=True)

        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

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

    # def patchify(self, imgs):
    #     """
    #     imgs: (N, 3, H, W)
    #     x: (N, L, patch_size**2 *3)
    #     """
    #     p = self.patch_embed.patch_size[0]
    #     assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    #     h = imgs.shape[2] // p
    #     w = imgs.shape[3] // p
        
    #     x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
    #     x = torch.einsum('nchpwq->nhwpqc', x)
    #     x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * imgs.shape[1]))
    #     return x
    
    def patchify(self, imgs):
        """
        imgs: (B, C, 1)
        output: (B, num_patchs, ceil(C/num_patchs))
        """
        imgs = imgs.transpose(1, 2)  # (B, 1, C)
        N, C, _ = imgs.shape
        num_patchs = self.patch_embed.num_patches
        remainder = C % num_patchs
        if remainder != 0:
            # If C is not divisible by num_patchs, pad imgs with zeros
            padding = torch.zeros((N, num_patchs - remainder, 1), dtype=imgs.dtype, device=imgs.device)
            imgs = torch.cat([imgs, padding], dim=1)
            C = imgs.shape[1]  # Update C

        imgs = imgs.reshape(N, num_patchs, np.ceil(C / num_patchs).astype(np.int32))
        return imgs   
        
    def unpatchify(self, imgs,C):
        """
        imgs: (B, num_patchs, ceil(C/num_patchs))
        output: (B, C, 1)
        """
        N, num_patchs, a = imgs.shape
        x = a * num_patchs
        remainder = x % C
        
        if remainder != 0:
            # If C is not divisible by num_patchs, remove the zero padding

            imgs = imgs[:, :(x - remainder), :]
            
        imgs = imgs.reshape(N, C, 1)
        return imgs
    
    # def unpatchify(self, x):
    #     """
    #     x: (N, L, patch_size**2 *3)
    #     imgs: (N, 3, H, W)
    #     """
    #     p = self.patch_embed.patch_size[0]
    #     h = self.h
    #     w = self.w
    #     assert h * w == x.shape[1]
        
    #     hid_chans = int(x.shape[2]/(p**2))
    #     x = x.reshape(shape=(x.shape[0], h, w, p, p, hid_chans))
    #     x = torch.einsum('nhwpqc->nchpwq', x)
    #     imgs = x.reshape(shape=(x.shape[0], hid_chans, h * p, h * p))
    #     return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):   
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed
        x1 = self.ph1(x)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        

        return x,x1

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def forward_classification(self, x):
        # x = self.ph3(x)
        if self.global_pool:
            feat = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        else:
            feat = x[:, 0, :]  # with cls token
        logits = self.cls_head(feat)
        return logits

    def forward(self, imgs,y = None, mask_ratio=0.2):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)        
        # Reconstruction branch
        pred, latent_f = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)

        # Classification branch
        logits = self.forward_classification(latent)

        return loss, pred, mask, logits
    
    @staticmethod
    def _extract_class_weights(encoder_q):
        if isinstance(encoder_q, nn.Sequential):
            class_weight = encoder_q[-1].weight.clone().detach()
        else:
            class_weight = encoder_q.weight.clone().detach()
        return class_weight 