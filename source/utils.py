"""Utilities"""
import numpy as np
import torch
import torch.nn.init as init
import random


def set_seed(seed):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def unnorm(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """Unnormalize image"""
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(s)
    return img


def load_checkpoint(ckpt_path, map_location=None):
    """Load weights from checkpoint"""
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(f'Load from checkpoint {ckpt_path} success!', flush=True)
    return ckpt


def save_checkpoint(state, save_path):
    """Save state"""
    torch.save(state, save_path)


def update_req_grad(models, requires_grad=True):
    """Update model require grad"""
    for model in models:
        for param in model.parameters():
            param.requires_grad = requires_grad


def init_weights(model, gain=0.02):
    """Init weights for model"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    model.apply(init_func)
