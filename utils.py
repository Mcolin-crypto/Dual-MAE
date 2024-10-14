import torch
import os
import torch.nn as nn
from logger import *
import random
import numpy as np
import torch.nn.functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

class branch_fusion(nn.Module):
    def __init__(self, channel, reduction=16):
        super(branch_fusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Linear(channel, channel // reduction, bias=False)
        self.relu=nn.ReLU(inplace=True)
        self.fc2=nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x): 
        b, c = x.size()        
        y1=self.fc1(x)
        self.fc1_feature=y1
        y2=self.relu(y1)  
        self.relu_feature=y2
        y3=self.fc2(y2)   
        self.fc2_feature=y3
        y4=self.sigmoid(y3).view(b, c)
        self.sigmoid_feature=y4
        y5=x * y4.expand_as(x)
        self.feature=y5
        return y5

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

def save_checkpoint(output_dir=None, ckpt_interval=10, epoch=None, model: torch.nn.Module=None,
                    optimizer: torch.optim.Optimizer=None,
                    lr_scheduler=None, max_acc=None,
                    val_acc=None, lr_drop=None, save_path=None,logger = false_logger):
    _none = lambda : None
    save_state = {'model': getattr(model, 'state_dict', _none)(),
                  'optimizer': getattr(optimizer, 'state_dict', _none)(),
                  'lr_scheduler': getattr(lr_scheduler, 'state_dict', _none)(),
                  'max_accuracy': max_acc,
                  'epoch': epoch,}
    save_state = {k:v for k, v in save_state.items() if v is not None}

    if save_path is None:
        if epoch is not None:
            if (epoch + 1) % ckpt_interval == 0:
                save_path = os.path.join(output_dir, f'ckpt_epoch_{epoch}.pth')
        if lr_drop is not None:
            if (epoch + 1) == lr_drop:
                save_path = os.path.join(output_dir, f'ckpt_epoch_{epoch}_before_drop.pth')
        if val_acc is not None and max_acc is not None:
            if val_acc > max_acc:
                save_path = os.path.join(output_dir, 'best_ckpt.pth')
    if save_path is not None:
        torch.save(save_state, save_path)
        logger.info(f"{save_path} saved.")