
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import numpy as np
from collections import defaultdict


import utils
from dem_utils.dem_cubic import DEMBicubic
from dem_utils import cmp_DEMFeature

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cfg = OmegaConf.load('./configs/datasets/tfasr_30to10.yaml')
    temp_pkgs = {}
    for key in cfg.keys():
        if 'dataset' not in key:
            temp_pkgs[key]=cfg[key]
            print(key)
    
    cfg['dataset_spec']=temp_pkgs
    cfg = OmegaConf.to_container(cfg, resolve=True)

    dataset_spec = cfg.get('train_dataset')
    train_dataset = utils.object_from_dict(dataset_spec)
    train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            num_workers=8,
            shuffle=True,
            pin_memory=True
        )
    
    dataset_spec = cfg.get('test_dataset')
    test_dataset = utils.object_from_dict(dataset_spec)
    test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            pin_memory=True
        )

    print("Length of train dataset: {}, Length of test dataset: {}".format(len(train_dataset), len(test_dataset)))

    # for batch,idx in tqdm(train_loader):
    #     pass

    cut_edge = 2
    # cut_edge = None
    method_list = [
    ]
    method_dict = {k:defaultdict(list) for k in method_list}
    method_dict['DEMCubic']=defaultdict(list)

    pbar = tqdm(test_loader)
    for batch,idx in pbar:
        ih, iw = batch['inp'].shape[-2:]
        scale = math.sqrt(batch['coord'].shape[1] / (ih * iw))
        hr_shape = [round(ih * scale), round(iw * scale)]

        lr_dem = batch['inp'].squeeze().numpy()
        hr_gt_dem = batch['gt'].reshape(hr_shape).numpy()

        if method_dict.get('DEMCubic', None) != None:
            hr_cubic_dem = DEMBicubic(
                    lr_dem.reshape([*lr_dem.shape, 1]),
                    scale
                )[:,:,0].astype(hr_gt_dem.dtype)
            
            dem_scale, dem_bias = batch['add_args'][...,0].numpy(), batch['add_args'][...,1].numpy()
            sr_dem = hr_cubic_dem*dem_scale + dem_bias

            demcubic_res = cmp_DEMFeature(
                    hr_gt_dem,
                    sr_dem,
                    cut_edge,
                    False
            )
            test_info = []
            for m, v in demcubic_res.items():
                method_dict['DEMCubic'][m].append(v)
                test_info.append("{}: {:.4f}".format(m,v))

            demcubic_diff = hr_gt_dem-sr_dem
            if cut_edge:
                demcubic_diff = demcubic_diff[cut_edge:-cut_edge,cut_edge:-cut_edge]
            method_dict['DEMCubic']['mae'].append( np.abs(demcubic_diff).mean() )
            method_dict['DEMCubic']['mse'].append(((demcubic_diff)**2).mean())
            method_dict['DEMCubic']['rmse'].append((((demcubic_diff)**2).mean())**0.5)
            
            test_info=', '.join(test_info)
            pbar.set_description(test_info)
        
        pass

    for n, r_dict in method_dict.items():
        for m, v in r_dict.items():
            average_m = np.mean(v)
            average_std = np.std(v)
            print("{}-{}: {:.4f} +- {:.4f}".format(
                n, m, average_m, average_std))
        print('')