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
    cfg = OmegaConf.load('rec-tif.yaml')
    # cfg = OmegaConf.load('rec-pyrenees96.yaml')
    cfg = OmegaConf.to_container(cfg, resolve=True)
    rec_dirs = cfg['rec_dirs']
    sr_scale = cfg['scale']
    # cut_edge = None
    cut_edge = 2
    print("Dataset: {}, sr_scale: {}, cut_edge: {}.".format(
        cfg['test_dataset'],sr_scale,cut_edge))
    
    method_list = [
        'bicubic', 

        # 'original-tfasrx4',
        # 'tfasr_dx2',
        'tfasr_dx4',
        # 'tfasr_dx6',
        # 'tfasr_dx8',
        # 'tfasr_dx8_120',
        # 'tfasr_dx16',
        # 'tfasr_dx16_120',

        # 'ebcf-nearest',
        # 'ebcf-nearest-best',
        # 'ebcf-nearest-pe16',
        'ebcf-nearest-pe16-best',
        # 'ebcf-nearest-pe16-s16-best',

        # 'ebcf-none',
        # 'ebcf-none-best',
        # 'ebcf-none-pe16-best',
        # 'ebcf-none-pe16-s16-best',
    ]

    method_dict = {}
    plot_num = len(method_list)+1

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="./temp_comparison")
    parser.add_argument('--type', type=str, default="original")
    opt = parser.parse_args()

    file_regexp=cfg['gt_dir']+'*_inp.tif'
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

    fig = plt.figure()
    if opt.type == 'aspect':
        cmap=my_aspect_cmap
        norm=my_aspect_norm
        ticks=my_aspect_bounds
        plt_vmax = None
    elif opt.type == 'slope':
        cmap=my_slope_cmap
        norm=None
    else:
        cmap=topocmap
        norm=None
        ticks=None
        plt_vmax = None


    for dem_file in tqdm(filenames):
        fidx = dem_file.split('/')[-1].split('_')[0]

        # check show
        # if fidx not in ['2026']:
        #     continue

        dem_gt = imageio.v2.imread(dem_file)
        H, W = dem_gt.shape
        lr_gt = cv2.resize(dem_gt, (H // sr_scale, W // sr_scale),
                        interpolation=cv2.INTER_NEAREST)
        

        if opt.type == 'slope':
            dem_tensor = torch.from_numpy(dem_gt).cuda().unsqueeze(0).unsqueeze(0)
            dem_slope = Slope_net(dem_tensor).squeeze().detach().cpu().numpy()
            gt_res = dem_slope
        elif opt.type == 'aspect':
            dem_tensor = torch.from_numpy(dem_gt).cuda().unsqueeze(0).unsqueeze(0)
            dem_aspect = Aspect_net(dem_tensor).squeeze().detach().cpu().numpy()
            gt_res = dem_aspect
        else:
            gt_res = dem_gt

        if cut_edge:
            gt_res = gt_res[cut_edge:-cut_edge,cut_edge:-cut_edge]

        max_value_method=''
        record_max_value=-1.0
        for k in method_list:
            v = rec_dirs[f'{k}_dir']
            file_regexr = v+f'{fidx}_*'
            try:
                file_pth = glob.glob(file_regexr)[0]
            except:
                print(file_regexr)
                exit(1)
            dem_sr = imageio.v2.imread(file_pth)

            if opt.type == 'slope':
                dem_tensor = torch.from_numpy(dem_sr).cuda().unsqueeze(0).unsqueeze(0)
                dem_slope = Slope_net(dem_tensor).squeeze().detach().cpu().numpy()
                sr_res = dem_slope
            elif opt.type == 'aspect':
                dem_tensor = torch.from_numpy(dem_sr).cuda().unsqueeze(0).unsqueeze(0)
                dem_aspect = Aspect_net(dem_tensor).squeeze().detach().cpu().numpy()
                sr_res = dem_aspect
            elif opt.type == 'value':
                sr_res = np.abs(dem_sr-dem_gt)
            elif opt.type == "original":
                sr_res = dem_sr
            else:
                raise Exception("Wrong generated type: {}".format(opt.type))

            if cut_edge:
                sr_res = sr_res[cut_edge:-cut_edge,cut_edge:-cut_edge]
            method_dict[k] = sr_res

            if np.max(sr_res)>record_max_value:
                max_value_method = k
                record_max_value = np.max(sr_res)

        save_file = os.path.join(save_dir, fidx+f'-{opt.type}.png')
        fig.clear()
        plt.subplot(2, plot_num//2, 1)
        if opt.type =='aspect':
            im = plt.imshow(
                gt_res, 
                vmax=plt_vmax,
                cmap=cmap,
                norm=norm,
                )
        elif opt.type == 'slope':
            plt_vmax = np.max(gt_res)
            slope_bounds = [i*plt_vmax/9 for i in range(10)]
            plt_vmax=None
            norm = mpl.colors.BoundaryNorm(slope_bounds, 9)
            ticks=slope_bounds
            im = plt.imshow(gt_res, cmap=cmap, vmax=plt_vmax, norm=norm)
        elif opt.type == 'original':
            im = plt.imshow(
                gt_res, 
                vmax=np.max(gt_res),
                vmin=np.min(gt_res),
                cmap='terrain'
            )
        else:
            plt.imshow(gt_res, cmap ='gray')
            im = None
            plt_vmax = record_max_value

        plt.title('Original')
        plt.axis('off')
        p_iter = 1
        for k,v in method_dict.items():
            p_iter +=1
            plt.subplot(2, plot_num//2, p_iter)

            if opt.type == "original":
                plt.imshow(v, cmap='terrain',vmax=np.max(gt_res),vmin=np.min(gt_res),)
            elif (im is None) and (k == max_value_method):
                # find which image provide the colorbar
                im = plt.imshow(v, cmap=topocmap, vmax=plt_vmax)
            else:
                # plt.imshow(v, cmap=topocmap, vmax=plt_vmax)
                plt.imshow(
                    v,
                    vmax=plt_vmax,
                    cmap=cmap,
                    norm=norm,
                )
            if 'tfasr' in k:
                plt_name = 'tfasr'
            elif 'ebcf' in k:
                plt_name = 'ebcf'
            else:
                plt_name = k
            plt.title(plt_name)
            plt.axis('off')

        # fig.colorbar(im)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.025, 0.7])
        fig.colorbar(im, cax=cbar_ax, ticks=ticks)
        plt.savefig(save_file, dpi=500)