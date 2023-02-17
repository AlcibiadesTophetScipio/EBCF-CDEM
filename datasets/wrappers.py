import random
import math

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import to_pixel_samples, resize_fn

class SDFImplicitDownsampled(Dataset):
    def __init__(self,
                 dataset,
                 inp_size=None,
                 scale_min=1, 
                 scale_max=None,
                 sample_q=None,
                 **kwargs,
                 ) -> None:
        '''
        Distribute the elevation value at (x,y)
        '''
        super().__init__()
        self.dataset = dataset
        self.length = len(dataset)

        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.sample_q = sample_q


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dem_pkgs = self.dataset[idx]
        img = dem_pkgs['dem_data']
        add_args = dem_pkgs['add_args']
        img_coord, img_value = to_pixel_samples(img.contiguous())
        img_coord = img_coord.permute(1,0).view([-1, *(img.shape[1:])])

        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img

            global_coord = img_coord[:, :round(h_lr * s), :round(w_lr * s)]
            # crop_hr = img[:, :round(h_lr * s), :round(w_lr * s)]
            # crop_lr = resize_fn(crop_hr, (h_lr, w_lr))
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)
            global_coord = img_coord[:, x0: x0 + w_hr, y0: y0 + w_hr]
            
        global_coord = global_coord.reshape(2,-1).permute(1,0)
        # hr_value = crop_hr.reshape(img.shape[0],-1).permute(1,0)
        hr_coord, hr_value = to_pixel_samples(crop_hr.contiguous())
        
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_value = hr_value[sample_lst]
            global_coord = global_coord[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_value,
            'add_args': add_args,
            'global_coord': global_coord,
        }, idx
