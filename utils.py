
import pydoc
import numpy as np
import time
import torch
import os
from collections import defaultdict
from typing import Tuple
from functools import partial
import imageio
from imgaug.augmentables.heatmaps import HeatmapsOnImage

def object_from_dict(
    d,
    parent=None,
    return_type: str = 'object',
    **default_kwargs
):
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    for k, v in kwargs.items():
        if isinstance(v, dict) and v.get("type") is not None:
            kwargs[k] = object_from_dict(d=v)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)  # skipcq PTC-W0034

    if return_type == 'object':
        return pydoc.locate(object_type)(**kwargs)
    elif return_type == 'func':
        return partial(pydoc.locate(object_type), **kwargs)
    else:
        raise Exception(f'Wrong return_type({return_type}) for importing.')

def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v

def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


def calc_psnr(sr, hr, scale=None, data_range=1.0):
    diff = (sr - hr) / data_range
    if scale is None:
        valid = diff
    else:
        shave = scale
        valid = diff[..., shave:-shave, shave:-shave]

    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)

def data2dem(data, file_pth, heatmap_flag=False):
    if isinstance(data, torch.Tensor):
        data=data.squeeze()
        data=(data-data.min())/(data.max()-data.min()+1.0e-6)

        if heatmap_flag:
            data = data.detach().cpu().numpy()
            heatmap_img = HeatmapsOnImage(data, shape=data.shape)
            imageio.imsave(file_pth,heatmap_img.draw(size=data.shape)[0])

        else:
            data=(data*255.0).floor()
            data = data.detach().cpu().numpy().astype(np.uint8)
            imageio.imsave(file_pth, data)

def data2heatmap(data, file_pth, max_value=1.0):
    data=data.squeeze()
    data = data.detach().cpu().numpy()
    heatmap_img = HeatmapsOnImage(data, shape=data.shape, max_value=max_value)
    imageio.imsave(file_pth,heatmap_img.draw(size=data.shape)[0])

def data2tif(data, file_pth):
    data=data.squeeze()
    data = data.detach().cpu().numpy()
    imageio.imsave(file_pth, data)