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

from dem_utils import Slope_net, Aspect_net

if __name__ == '__main__':
    '''
    python plot_dem_features.py --fregexp multi-exp_interpolator-tif/tfasr-identity-x1/*/*/*.tif \
                                --save_dir ./vis_tfasr-gt \
                                --type aspect
    python plot_dem_features.py --save_dir ./vis_tfasr_ebcf-nearest_dx2 \
                                --type aspect \
                                --fregexp rec-tifs/multi-tfasrx4_edsrB-ebcf-nearest-pe16_best/tfasr-x2/*/*/*.tif
    python plot_dem_features.py --save_dir ./vis_tfasr_ebcf-nearest_dx4 \
                                --type aspect \
                                --fregexp rec-tifs/multi-tfasrx4_edsrB-ebcf-nearest-pe16_best/tfasr-x4/*/*/*.tif
    python plot_dem_features.py --save_dir ./vis_tfasr_tfasr_dx2 \
                                --type aspect \
                                --fregexp rec-tifs/multi-tfasr_tfasr_dx2/tfasr-x2/*/*/*.tif
    python plot_dem_features.py --save_dir ./vis_tfasr_tfasr_dx4 \
                                --type aspect \
                                --fregexp rec-tifs/multi-tfasr_tfasr_dx4/tfasr-x4/*/*/*.tif


    python plot_dem_features.py --save_dir ./vis_tfasr_bicubic_dx4 \
                                --type aspect \
                                --fregexp rec-tifs/temp_tfasr-bicubic-d4/*.tif
    python plot_dem_features.py --save_dir ./vis_tfasr_bicubic_dx2 \
                                --type aspect \
                                --fregexp rec-tifs/temp_tfasr-bicubic-d2/*.tif
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--main_dir', type=str, default="/home/syao/Programs/Experiments/EBCF-CDEM")
    parser.add_argument('--save_dir', type=str, default="./temp_dem-features")
    parser.add_argument('--type', type=str, default="aspect")
    parser.add_argument('--fregexp', type=str, default="multi-exp_interpolator-tif/tfasr-identity-x1/*/*/*.tif")
    opt = parser.parse_args()

    if opt.main_dir is not None:
        file_regexp = os.path.join(opt.main_dir, opt.fregexp)
    else:
        file_regexp = opt.fregexp
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
    cut_edge = 2
    try:
        import pycpt
        topocmap = pycpt.load.cmap_from_cptcity_url('wkp/schwarzwald/wiki-schwarzwald-cont.cpt')
    except:
        topocmap = 'Spectral_r'
        topocmap = plt.get_cmap(topocmap)

    fig = plt.figure()
    for dem_file in tqdm(filenames):
        dem_data = imageio.v2.imread(dem_file)
        # if cut_edge:
        #     dem_data = dem_data[cut_edge:-cut_edge,cut_edge:-cut_edge]
        with torch.no_grad():
            dem_tensor = torch.from_numpy(dem_data).cuda().unsqueeze(0).unsqueeze(0)
            dem_slope = Slope_net(dem_tensor).squeeze().detach().cpu().numpy()
            dem_aspect = Aspect_net(dem_tensor).squeeze().detach().cpu().numpy()
        if cut_edge:
            dem_slope = dem_slope[cut_edge:-cut_edge,cut_edge:-cut_edge]
            dem_aspect = dem_aspect[cut_edge:-cut_edge,cut_edge:-cut_edge]

        filename = dem_file.split('/')[-1].split('_')[0]
        imageio.imsave(os.path.join(save_dir, filename+'-aspect.tif'), dem_aspect)
        slop_file = os.path.join(save_dir, filename+'-slop.png')
        aspect_file = os.path.join(save_dir, filename+'-aspect.png')


        ##### Very slow
        # rd_slop = rd.rdarray(dem_slope, no_data=-9999)
        # rd.rdShow(rd_slop, cmap=topocmap, axes=False, show=False)
        # plt.savefig(slop_file)

        # rd_aspect = rd.rdarray(dem_aspect, no_data=-9999)
        # rd.rdShow(rd_aspect, cmap=topocmap, axes=False, show=False)
        # plt.savefig(aspect_file)

        if opt.type == 'slope':
            fig.clear()
            im = plt.imshow(dem_slope, cmap=topocmap)
            fig.colorbar(im)
            plt.axis('off')
            plt.savefig(slop_file)
        elif opt.type == 'aspect':
            fig.clear()
            im = plt.imshow(dem_aspect, cmap='hsv', vmax=360.0)
            fig.colorbar(im, ticks=[0,90,180,270,360])
            plt.axis('off')
            plt.savefig(aspect_file)
            # plt.imsave(aspect_file, dem_aspect, cmap='hsv', vmax=360)
        else:
            raise Exception("Wrong generated type: {}".format(opt.type))

        
