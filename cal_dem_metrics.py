import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import matplotlib.pyplot as plt
import imageio
import glob
from pathlib import Path
from omegaconf import OmegaConf
import math
import numpy as np
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from collections import defaultdict
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
from tqdm import tqdm
import cv2

from dem_utils import cmp_TerrainAttribute, cmp_DEMFeature, cmp_DEMParameter, cmp_dem_extractor
from dem_utils.dem_cubic import DEMBicubic


if __name__ == '__main__':
    # cfg = OmegaConf.load('rec-tif.yaml')
    # cfg = OmegaConf.load('rec-tfasr30to10.yaml')
    # cfg = OmegaConf.load('rec-pyrenees96.yaml')
    # cfg = OmegaConf.load('encoders-tif.yaml')
    # cfg = OmegaConf.load('cross-tif.yaml')
    # cfg = OmegaConf.load('rec-tif-tfasr.yaml')

    cfg = OmegaConf.load('sr-tif.yaml')
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cell_size=30.0
    if cfg['test_dataset'] == 'pyrenees':
        cell_size=2.0

    rec_dirs = cfg['rec_dirs']
    sr_scale = cfg['scale']
    # cut_edge = None
    cut_edge = 2
    print("Dataset: {}, sr_scale: {}, cut_edge: {}.".format(
        cfg['test_dataset'],sr_scale,cut_edge))
    
    method_list = [
        "sr"
    ]
    method_dict = {k:defaultdict(list) for k in method_list}

    # Add DEM Cubic Method, very slow
    method_dict['DEMCubic']=defaultdict(list)

    filenames = glob.glob(cfg['gt_dir']+'*_inp.tif')
    for fname in tqdm(filenames):
        fstem = fname.split('/')[-1]
        fidx = fstem.split('_')[0]
        file_gt_reg = cfg['gt_dir']+f'{fidx}_inp*'
        file_gt_pth = glob.glob(file_gt_reg)[0]
        im_gt_origin = imageio.v2.imread(file_gt_pth)

        if method_dict.get('DEMCubic', None) is not None:
            H, W = im_gt_origin.shape
            lr_gt = cv2.resize(im_gt_origin, (H // sr_scale, W // sr_scale),
                            interpolation=cv2.INTER_NEAREST)
            hr_dem_cubic = DEMBicubic(
                lr_gt.reshape([*lr_gt.shape, 1]),
                sr_scale
            )[:,:,0].astype(im_gt_origin.dtype)
            demcubic_res = cmp_DEMFeature(
                im_gt_origin,
                hr_dem_cubic,
                cut_edge,
                False
            )
            for m, v in demcubic_res.items():
                method_dict['DEMCubic'][m].append(v)
            demcubic_diff = im_gt_origin-hr_dem_cubic
            if cut_edge:
                demcubic_diff = demcubic_diff[cut_edge:-cut_edge,cut_edge:-cut_edge]
            method_dict['DEMCubic']['mae'].append( np.abs(demcubic_diff).mean() )
            method_dict['DEMCubic']['mse'].append(((demcubic_diff)**2).mean())
            method_dict['DEMCubic']['rmse'].append((((demcubic_diff)**2).mean())**0.5)

        if cut_edge:
            im_gt=im_gt_origin[cut_edge:-cut_edge,cut_edge:-cut_edge]
        else:
            im_gt=im_gt_origin

        mae_png_list = []
        for k in method_list:
            v = rec_dirs[f'{k}_dir']
            file_regexr = v+f'{fidx}_*'
            try:
                file_pth = glob.glob(file_regexr)[0]
            except:
                print(file_regexr)
                exit(1)
            im_sr_origin = imageio.v2.imread(file_pth)
            
            # cmp_dem_extractor(file_gt_pth)
            dem_res_feature = cmp_DEMFeature(
                file_gt_pth, 
                file_pth, 
                cut_edge
                ) # assume cell size is 1
            for m,v in dem_res_feature.items():
                method_dict[k][m].append(v)

            # dem_res_attrib = cmp_TerrainAttribute(file_gt_pth, file_pth, cell_size, cut_edge)
            # for m,v in dem_res_attrib.items():
            #     method_dict[k][m].append(v)

            # dem_res_params = cmp_DEMParameter(file_gt_pth, file_pth, cell_size, cut_edge)
            # for m,v in dem_res_params.items():
            #     method_dict[k][m].append(v)

            
            if cut_edge:
                im_sr=im_sr_origin[cut_edge:-cut_edge,cut_edge:-cut_edge]
            else:
                im_sr=im_sr_origin

            method_dict[k]['mae'].append( np.abs(im_gt-im_sr).mean() )
            method_dict[k]['mse'].append( ((im_gt-im_sr)**2).mean() )
            method_dict[k]['rmse'].append( (((im_gt-im_sr)**2).mean())**0.5 )
            
    for n, r_dict in method_dict.items():
        for m, v in r_dict.items():
            average_m = np.mean(v)
            average_std = np.std(v)
            print("{}-{}: {:.4f} +- {:.4f}".format(
                n, m, average_m, average_std))
        print('')
