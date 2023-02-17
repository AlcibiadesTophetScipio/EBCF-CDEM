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

def calc_ssim(sr, hr, scale=None, eps=1.0e-6):
    # sr = (sr-hr.min()) / (hr.max()-hr.min()+eps)
    # hr = (hr-hr.min()) / (hr.max()-hr.min()+eps)
    if scale:
        sr = sr[scale:-scale, scale:-scale]
        hr = hr[scale:-scale, scale:-scale]
    mssim = structural_similarity(
        sr, hr, 
        data_range=hr.max()-hr.min(),
        gaussian_weights=True,
    )
    return mssim

def calc_psnr(sr, hr, scale=None, eps=1.0e-6):
    sr = (sr-hr.min()) / (hr.max()-hr.min()+eps)
    hr = (hr-hr.min()) / (hr.max()-hr.min()+eps)
    if scale:
        sr = sr[scale:-scale, scale:-scale]
        hr = hr[scale:-scale, scale:-scale]
    return peak_signal_noise_ratio(hr, sr)

    diff = (sr-hr)
    mse = np.mean(diff**2)
    # return -10*np.log10(mse)
    return 20*np.log10(1.0/np.sqrt(mse))

def data2heatmap(data, file_pth, max_value=1.0):
    heatmap_img = HeatmapsOnImage(data, shape=data.shape, max_value=max_value)
    imageio.imsave(file_pth,heatmap_img.draw(size=data.shape)[0])


if __name__ == '__main__':
    # cfg = OmegaConf.load('rec-tif.yaml')
    # cfg = OmegaConf.load('encoders-tif.yaml')
    cfg = OmegaConf.load('cross-tif.yaml')
    cfg = OmegaConf.to_container(cfg, resolve=True)

    if cfg.get('train_dataset') is None:
        save_dir = cfg['save_dir']+'/main-{}-x{}'.format(cfg['test_dataset'], cfg['scale'])
    else:
        save_dir = cfg['save_dir']+'/{}To{}-x{}'.format(cfg['train_dataset'], cfg['test_dataset'], cfg['scale'])
    print(save_dir.split('/')[-1])
    Path(save_dir).expanduser().mkdir(parents=True, exist_ok=True)

    rec_dirs = cfg['rec_dirs']
    sr_scale = cfg['scale']
    method_list = [
        'bilinear',
        'bicubic', 
        # 'mapif-global', 
        # 'mapif', 
        # 'mapif-sdf',
        # 'metasr',
        # 'metasr-sdf',
        # 'liif-naive',
        # 'liif',
        # 'liif-sdf',
        # 'sdfif-nearest','sdfif-bicubic',
        # 're-metasr',
        # 're-liif',
        # 're-sdfif-bicubic'

    ]
    method_dict = {k:defaultdict(list) for k in method_list}

    # specifical config for different encoders test
    if cfg.get('encoder') is not None:
        encoder_name = cfg['encoder']
        method_list = [encoder_name+'-'+net_name for net_name in
                       [
                           'metasr', 'liif', 'sdfif-nearest',
                           'sdfif-bicubic',
                       ]]
        method_dict = {k:defaultdict(list) for k in method_list}

    filenames = glob.glob(cfg['gt_dir']+'*_inp.tif')
    for fname in filenames:
        fstem = fname.split('/')[-1]
        fidx = fstem.split('_')[0]
        file_gt_reg = cfg['gt_dir']+f'{fidx}_inp*'
        file_gt_pth = glob.glob(file_gt_reg)[0]
        im_gt_origin = imageio.v2.imread(file_gt_pth)
        if sr_scale is not None:
            im_gt=im_gt_origin[sr_scale:-sr_scale,sr_scale:-sr_scale]

        mae_png_list = []
        for k in method_list:
            v = rec_dirs[f'{k}_dir']
            file_regexr = v+f'{fidx}_*'
            file_pth = glob.glob(file_regexr)[0]
            im_sr_origin = imageio.v2.imread(file_pth)

            if sr_scale is not None:
                im_sr=im_sr_origin[sr_scale:-sr_scale,sr_scale:-sr_scale]
            method_dict[k]['mae'].append( np.abs(im_gt-im_sr).mean() )
            method_dict[k]['mse'].append( ((im_gt-im_sr)**2).mean() )
            method_dict[k]['psnr_origin'].append( calc_psnr(hr=im_gt_origin, sr=im_sr_origin, scale=sr_scale) )
            # method_dict[k]['psnr'].append( calc_psnr(hr=im_gt, sr=im_sr) )
            method_dict[k]['ssim_origin'].append( calc_ssim(hr=im_gt_origin, sr=im_sr_origin, scale=sr_scale) )
            # method_dict[k]['ssim'].append( calc_ssim(hr=im_gt, sr=im_sr) )

            mae_png_list.append(np.abs(im_gt-im_sr))

        # # save mae png
        # mae_img = np.concatenate(mae_png_list, axis=1)
        # statics_msg = '{:.2f}_{:.2f}'.format(
        #     mae_img.max(), mae_img.mean()
        # )
        # save_pth = save_dir+f'/{fidx}-{statics_msg}.png'
        # # if cfg['scale']==8:
        # #     split_value=0.5*mae_img.max()
        # # elif cfg['scale']==4:
        # #     split_value=5.0     
        # # else:
        # split_value=1.0*mae_img.max()
        # data2heatmap(mae_img, save_pth, split_value)

    # print(f"SR scale: {sr_scale}")
    for n, r_dict in method_dict.items():
        for m, v in r_dict.items():
            average_m = np.mean(v)
            print("{}-{}: {:.4f}".format(n,m,average_m))
        print('')
