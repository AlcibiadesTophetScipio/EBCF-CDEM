

from datasets.dem2be import demfile_io

import yaml
import json
import argparse
import glob
import os
import random
from pathlib import Path
from torch.utils.data import DataLoader
import imageio
import torch
import numpy as np
from tqdm import tqdm

def split_single_dem(
    dem_file,
    split_shape,
    export_dir='./temp/'):
    
    prefix = dem_file.split('/')[-1].split('_')[0]
    file_suffix = dem_file.split('.')[-1]
    dem_data = demfile_io(dem_file)
    
    H,W = dem_data.shape
    # H_interv, W_interv = split_shape[0]/4, split_shape[1]/4
    dem_slices = np.stack(
        np.meshgrid(
        range(0, H-split_shape[0], split_shape[0]),
        range(0, W-split_shape[1], split_shape[1])),
        axis=-1
        ).reshape([-1,2])

    filenames = []
    export_dir = Path(export_dir).expanduser()
    export_dir.mkdir(parents=True, exist_ok=True)
    for p_start in dem_slices:
        p_end = p_start + split_shape
        split_data = dem_data[p_start[0]:p_end[0], p_start[1]:p_end[1]]
        split_file = export_dir/f'{prefix}_{p_start[0]}x{p_start[1]}.{file_suffix}'
        filenames.append(str(split_file))
        if file_suffix == 'tif':
            imageio.imsave(split_file, split_data)
        elif file_suffix == 'dem':
            np.savetxt(split_file, split_data, fmt='%g', delimiter=',')

    return filenames

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirRawDataset', type=str)
    opt = parser.parse_args()

    # dirRawDataset = '/data/syao/Datasets/Terrains/pyrenees_raw'
    # dirRawDataset = '/data/syao/Datasets/Terrains/tyrol_raw'
    dirRawDataset = opt.dirRawDataset
    regexp = dirRawDataset+'/*2m.dem'
    rawDEMs = glob.glob(regexp)

    if len(rawDEMs)==0:
        raise Exception(f"Wrong regular expression {regexp}.")
    
    for dem_file in tqdm(rawDEMs):
        # dem_data = demfile_io(dem_file)
        # print(dem_file, [x/1 for x in dem_data.shape])

        split_single_dem(
            dem_file,
            split_shape=[96,96]
        )