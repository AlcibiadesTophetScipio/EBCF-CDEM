import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


from .dem2be import *

class DEMFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none', 
                 bias_value=1e4, scale_value=10.0,
                 transM='origin'):
        self.repeat = repeat
        self.cache = cache
        self.bias_value = bias_value
        self.scale_value = scale_value
        self.transM = transM

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]
        
        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'in_memory':
                dem_pkgs = self.dem2tensor(file)
                self.files.append(dem_pkgs)

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            dem_pkgs = self.dem2tensor(x)
            return dem_pkgs

        elif self.cache == 'in_memory':
            return x

    def dem2tensor(self,file):
        dem_data = demfile_io(file)

        # [scale, bias]
        add_args = [1.0, 0.0]
        if self.transM == 'origin':
            dem_data = dem_data
        elif self.transM == 'dem2one':
            dem_data, add_args = dem2one(dem_data=dem_data)
        elif self.transM == 'dem2multi':
            dem_data, add_args = dem2one(dem_data=dem_data)
            add_channels = dem2multi(dem_data=dem_data)

            dem_data = np.stack([dem_data, add_channels], axis=-1)

            
        # elif self.transM=='tif2rgb':
        #     dem_data = tif2rgb(dem_data=dem_data,
        #                         bias_value=self.bias_value,
        #                         scale_value=self.scale_value
        #                         )
        #     add_args=[self.scale_value, self.bias_value]
        # elif self.transM == 'tif2one3':
        #     dem_data, add_args = tif2one3(dem_data=dem_data)
        # elif self.transM == 'tif2e':
        #     dem_data, add_args = tif2e(dem_data=dem_data)
        else:
            raise Exception('Choose trans method.')

        return {
            'dem_data': transforms.ToTensor()(dem_data.astype(np.float32)),
            'add_args': torch.tensor(add_args, dtype=torch.float32),
        }

