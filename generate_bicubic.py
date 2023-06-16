from tqdm import tqdm
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import glob
import imageio
import cv2
from dem_utils.dem_cubic import DEMBicubic

if __name__ == '__main__':
    '''
    python generate_bicubic.py --fregexp multi-exp_interpolator-tif/tfasr-identity-x1/*/*/*.tif \
                               --scale 2 \
                               --save_dir ./temp_tfasr-bicubic-d2
    python generate_bicubic.py --fregexp multi-exp_interpolator-tif/tfasr-identity-x1/*/*/*.tif \
                               --scale 4 \
                               --save_dir ./temp_tfasr-bicubic-d4 

    python generate_bicubic.py --fregexp multi-exp_interpolator-tif/pyrenees96-identity-x1/*/*/*.tif \
                               --scale 2 \
                               --save_dir ./temp_pyrenees96-bicubic-d2
    python generate_bicubic.py --fregexp multi-exp_interpolator-tif/pyrenees96-identity-x1/*/*/*.tif \
                               --scale 4 \
                               --save_dir ./temp_pyrenees96-bicubic-d4
    python generate_bicubic.py --fregexp multi-exp_interpolator-tif/pyrenees96-identity-x1/*/*/*.tif \
                               --scale 6 \
                               --save_dir ./temp_pyrenees96-bicubic-d6
    python generate_bicubic.py --fregexp multi-exp_interpolator-tif/pyrenees96-identity-x1/*/*/*.tif \
                               --scale 8 \
                               --save_dir ./temp_pyrenees96-bicubic-d8

    
    python generate_bicubic.py --fregexp multi-exp_interpolator-tif/tyrol96-identity-x1/*/*/*.tif \
                               --scale 2 \
                               --save_dir ./temp_tyrol96-bicubic-d2
    python generate_bicubic.py --fregexp multi-exp_interpolator-tif/tyrol96-identity-x1/*/*/*.tif \
                               --scale 4 \
                               --save_dir ./temp_tyrol96-bicubic-d4 
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--main_dir', type=str, default="/home/syao/Programs/Experiments/EBCF-CDEM")
    parser.add_argument('--save_dir', type=str, default="./temp_dem-bicubic")
    parser.add_argument('--fregexp', type=str, default="multi-exp_interpolator-tif/tfasr-identity-x1/*/*/*.tif")
    parser.add_argument('--scale', type=int, default=2)
    opt = parser.parse_args()

    if opt.main_dir is not None:
        file_regexp = os.path.join(opt.main_dir, opt.fregexp)
    else:
        file_regexp = opt.fregexp

    try:
        filenames = glob.glob(file_regexp)
    except:
        print("Wrong reg exp: {}".format(file_regexp))
        exit(1)

    print("Files num: {}".format(len(filenames)))

    save_dir = opt.save_dir
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    cut_edge = 2

    sr_scale = opt.scale
    for dem_file in tqdm(filenames):
        dem_data = imageio.v2.imread(dem_file)
        H, W = dem_data.shape
        lr_gt = cv2.resize(dem_data, (H // sr_scale, W // sr_scale),
                        interpolation=cv2.INTER_NEAREST)
        hr_dem_cubic = DEMBicubic(
            lr_gt.reshape([*lr_gt.shape, 1]),
            sr_scale
        )[:,:,0].astype(dem_data.dtype)

        filename = dem_file.split('/')[-1].split('_')[0]
        bicubic_file = os.path.join(save_dir, filename+'_bicubic.tif')

        imageio.imsave(bicubic_file, hr_dem_cubic)