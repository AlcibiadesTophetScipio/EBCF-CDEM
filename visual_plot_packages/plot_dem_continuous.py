from tqdm import tqdm
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import glob
import imageio
import torch
import richdem as rd
from omegaconf import OmegaConf
import cv2
import matplotlib as mpl


from dem_utils import Slope_net, Aspect_net
from dem_utils.aspect_colormap import my_aspect_bounds,\
                                    my_aspect_norm,\
                                    my_aspect_cmap
from dem_utils.slope_colormap import my_slope_cmap

if __name__ == '__main__':
    main_dir="/home/syao/Programs/Experiments/EBCF-CDEM"
    test_dataset="pyrenees96"
    sr_scales = [2,4,6,8]

    # test_dataset="tfasr"
    # sr_scales = [2,4]

    gt_dir= f"{main_dir}/multi-exp_interpolator-tif/{test_dataset}-identity-x1/*/*/"
    

    method_dict = {
    "bicubic": f"{main_dir}/rec-tifs/temp_{test_dataset}-bicubic",
    "tfasr": f"{main_dir}/rec-tifs/multi-{test_dataset}_tfasr",
    "ebcf": f"{main_dir}/rec-tifs/multi-{test_dataset}_edsrB-ebcf-nearest-pe16_best/{test_dataset}",
    # "ebcf": f"{main_dir}/rec-tifs/multi-{test_dataset}x4_edsrB-ebcf-nearest-pe16_best/{test_dataset}", # special for tfasr
    }

    # cut_edge = None
    cut_edge = 2
    print("Dataset: {}, sr_scales: {}, cut_edge: {}.".format(
        test_dataset,sr_scales,cut_edge))
    
    plot_num = len(sr_scales)+1

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="./temp_continuous")
    parser.add_argument('--type', type=str, default="original-n")
    opt = parser.parse_args()

    file_regexp=gt_dir+'*_inp.tif'
    print(file_regexp)
    try:
        filenames = glob.glob(file_regexp)
    except:
        print("Wrong reg exp: {}".format(file_regexp))
        exit(1)
    print("Files num: {}".format(len(filenames)))

    save_dir = opt.save_dir
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        print("Make dir at: {}".format(save_dir))
    else:
        print("Using current dir: {}".format(save_dir))
    try:
        import pycpt
        topocmap = pycpt.load.cmap_from_cptcity_url('wkp/schwarzwald/wiki-schwarzwald-cont.cpt')
    except:
        topocmap = 'Spectral_r'
        topocmap = plt.get_cmap(topocmap)

    fig = plt.figure(figsize=(10,9))
    col_num = len(sr_scales)
    if opt.type == 'aspect':
        cmap=my_aspect_cmap
        norm=my_aspect_norm
        ticks=my_aspect_bounds
        plt_vmax = None
    elif opt.type == 'slope':
        cmap=my_slope_cmap
        norm=None
    elif opt.type == 'original':
        cmap="terrain"
        col_num+=1
    else:
        cmap=topocmap
        norm=None
        ticks=None
        plt_vmax = None

    for dem_file in tqdm(filenames):
        fig.clear()
        fidx = dem_file.split('/')[-1].split('_')[0]
        save_file = os.path.join(save_dir, fidx+f'-{opt.type}.png')

        # check show
        # if fidx not in ['2004','2024','2026','2034','2039']:
        if fidx not in ['666','811']:
            continue
        
        dem_gt = imageio.v2.imread(dem_file)
        gt_max_value = np.max(dem_gt)
        gt_min_value = np.min(dem_gt)

        H, W = dem_gt.shape
        for i,scale in enumerate(sr_scales):
            lr_gt = cv2.resize(dem_gt, (H // scale, W // scale),
                            interpolation=cv2.INTER_NEAREST)
            plt.subplot(len(method_dict)+1, col_num, i+1)
            plt.imshow(lr_gt, cmap ='terrain', vmax=gt_max_value, vmin=gt_min_value)
            plt.title(f'input-x{scale}')
            plt.axis('off')

        record_max_value=-1.0
        method_results = {k:{} for k in method_dict.keys()}
        for k,d in method_dict.items():
            
            for i,scale in enumerate(sr_scales):
                if k == 'bicubic':
                    file_regexp = d+f'-d{scale}/{fidx}_*'
                elif k == 'tfasr':
                    file_regexp = d+f'_dx{scale}/{test_dataset}-x{scale}/*/*/{fidx}_*'
                else:
                    file_regexp = d+f'-x{scale}/*/*/{fidx}_*'

                try:
                    file_pth = glob.glob(file_regexp)[0]
                except:
                    print("Wrong file reg exp:", file_regexp)
                    exit(1)
                dem_sr = imageio.v2.imread(file_pth)

                if opt.type == 'mae':
                    sr_res = np.abs(dem_sr-dem_gt)
                elif opt.type == "original":
                    sr_res = dem_sr
                elif opt.type == "original-n":
                    sr_res = dem_sr
                else:
                    raise Exception("Wront option type.")

                if cut_edge:
                    sr_res = sr_res[cut_edge:-cut_edge,cut_edge:-cut_edge]
                method_results[k][scale]=sr_res
                if np.max(sr_res)>record_max_value:
                    record_max_value = np.max(sr_res)
        
        row_idx = 0
        plt_vmax = record_max_value
        for m,sr_results in method_results.items():
            row_idx += 1
            for i,scale in enumerate(sr_scales):
                sr_res = sr_results[scale]

                plot_idx = row_idx*col_num+i+1
                plt.subplot(len(method_dict)+1, col_num, plot_idx)
                if opt.type == "original":
                    im = plt.imshow(sr_res, cmap ='terrain', vmax=gt_max_value, vmin=gt_min_value)
                elif opt.type == "original-n":
                    im = plt.imshow(sr_res, cmap ='terrain', vmax=gt_max_value, vmin=gt_min_value)
                else:
                    im = plt.imshow(sr_res, cmap = cmap, vmax=plt_vmax)
                # plt.title(f'{m}-x{scale}')
                plt.axis('off')
        if opt.type == "original":
            plt.subplot(len(method_dict)+1, col_num, col_num)
            im = plt.imshow(dem_gt, cmap=cmap)
            plt.axis('off')
            plt.title("HR")
            fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)
            cbar_ax = fig.add_axes([0.8, 0.2, 0.02, 0.4])
        else:
            fig.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)
            cbar_ax = fig.add_axes([0.93, 0.2, 0.02, 0.6])

        fig.colorbar(im, cax=cbar_ax)
        plt.savefig(save_file, dpi=500)
        continue
