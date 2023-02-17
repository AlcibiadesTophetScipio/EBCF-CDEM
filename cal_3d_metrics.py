from multiprocessing.util import is_exiting
import matplotlib.pyplot as plt
import imageio
import glob
from pathlib import Path
from omegaconf import OmegaConf
import math
import numpy as np
from collections import defaultdict
import trimesh
from pytorch3d.loss import chamfer_distance
import torch
from tqdm import tqdm
from vedo.utils import trimesh2vedo

def cal_perc_dis(distances, perc=0.5):
    d = np.array(distances)
    assert len(d.shape) == 1

    idxs = np.argsort(d)
    return d[idxs[int(idxs.size*perc)]]


def cal_perc_num(distances, threshold=0.01):
    d = np.array(distances)
    assert len(d.shape) == 1

    num = (d <= threshold).sum()
    return num / d.size


def cal_statics(distances):
    d = np.array(distances)
    assert len(d.shape) == 1

    return [d.mean(), d.std(), d.max(), d.min()]

def calc_chamferDis(pts_gt, pts_target):
    tPts_gt = torch.from_numpy(pts_gt).unsqueeze(0).cuda()
    tPts_target = torch.from_numpy(pts_target).unsqueeze(0).cuda()

    chamferDis, _ = chamfer_distance(tPts_gt, tPts_target)
    return chamferDis.item()

def get_pts_from_reg(
    pts_file_reg,
    mesh_file_reg,
    replace_reg,
    samples_num,
    dem_res,
    scale=8,
):
    npy_pth = glob.glob(pts_file_reg)
    if len(npy_pth) == 0:
        mesh_pth = glob.glob(mesh_file_reg)[0]
        pts_file = mesh_pth.replace(*replace_reg)[:-3]+'npy'
        Path(pts_file).parent.mkdir(parents=True, exist_ok=True)

        mesh_origin = trimesh.load(mesh_pth)
        pts = trimesh.sample.sample_surface(mesh_origin, int(samples_num*1.2))[0]

        x_slice = (pts[:,0]>scale).astype(int)+(pts[:,0]<255-scale).astype(int)
        y_slice = (pts[:,1]>scale).astype(int)+(pts[:,1]<255-scale).astype(int)
        idx_select=(x_slice+y_slice)>=4
        pts = pts[idx_select]
        while pts.shape[0]<int(samples_num):
            pts_supp = trimesh.sample.sample_surface(mesh_origin, int(samples_num*0.2))[0]
            x_slice = (pts_supp[:,0]>scale).astype(int)+(pts_supp[:,0]<255-scale).astype(int)
            y_slice = (pts_supp[:,1]>scale).astype(int)+(pts_supp[:,1]<255-scale).astype(int)
            idx_select=(x_slice+y_slice)>=4
            pts_supp = pts_supp[idx_select]
            pts = np.concatenate([pts, pts_supp], axis=0)
        pts = pts[:int(samples_num),:]
        np.save(pts_file, pts)
    else:
        pts = np.load(npy_pth[0])

    return pts*[dem_res, dem_res, 1]


if __name__ == '__main__':
    # cfg = OmegaConf.load('rec-ply.yaml')
    # cfg = OmegaConf.load('encoders-ply.yaml')
    cfg = OmegaConf.load('cross-ply.yaml')
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Path(save_dir).expanduser().mkdir(parents=True, exist_ok=True)
    rec_dirs = cfg['rec_dirs']
    sr_scale = cfg['scale']
    dem_res_dict = {
        'astgtmv003': 30,
        'pyrenees': 2,
        'tyrol': 2,
    }
    dem_res = dem_res_dict[cfg['test_dataset']]
    samples_num= 128**2
    if cfg["train_dataset"]!=cfg['test_dataset']:
            print(
            "From {} to {}, sr scale: {}, dem resolution: {}."
            .format(cfg['train_dataset'], cfg['test_dataset'], sr_scale, dem_res)
        )
    else:
        print(
            "test dataset: {}, sr scale: {}, dem resolution: {}."
            .format(cfg['test_dataset'], sr_scale, dem_res)
        )

    method_list = [
        'bilinear',
        'bicubic', 
        'mapif-global', 
        'mapif', 
        'mapif-sdf',
        'metasr',
        'metasr-sdf',
        # 'liif-naive',
        'liif',
        'liif-sdf',
        'sdfif-nearest','sdfif-bicubic'
    ]
    method_dict = {k:defaultdict(list) for k in method_list}

    # specifical config for different encoders test
    if cfg.get('encoder') is not None:
        encoder_name = cfg['encoder']
        method_list = [encoder_name+'-'+net_name for net_name in 
        [
            # 'metasr','liif','sdfif-nearest',
            'sdfif-bicubic',
            ]]
        method_dict = {k:defaultdict(list) for k in method_list}

    if cfg.get('gt_ply_dir') is None:
        gt_ply_dir = cfg['ply_dir']+'/'+rec_dirs['gt_dir']
        gt_npy_dir = cfg['npy_dir']+'/'+rec_dirs['gt_dir']
        Path(gt_npy_dir).mkdir(parents=True, exist_ok=True)
    else:
        gt_ply_dir = cfg['gt_ply_dir']
        gt_npy_dir = cfg['gt_npy_dir']

    filenames = glob.glob(gt_ply_dir+'*_inp.ply')
    for fname in tqdm(filenames):
        fstem = fname.split('/')[-1]
        fidx = fstem.split('_')[0]

        gt_npy_reg = gt_npy_dir+f'{fidx}_inp*'
        gt_ply_reg = gt_ply_dir+f'{fidx}_inp*'
        pts_gt = get_pts_from_reg(
            gt_npy_reg,
            gt_ply_reg,
            [cfg['ply_dir'], cfg['npy_dir']],
            samples_num,
            dem_res=dem_res,
            scale=cfg['scale']
        )
        gt_mesh_pth = glob.glob(gt_ply_reg)[0]
        gt_mesh = trimesh.load(gt_mesh_pth)
        gt_mesh.vertices = gt_mesh.vertices*[dem_res,dem_res,1]
        
        for k in method_list:
            target_npy_regexr = cfg['npy_dir']+'/'+rec_dirs[f'{k}_dir']+f'{fidx}_*'
            target_ply_regexr = cfg['ply_dir']+'/'+rec_dirs[f'{k}_dir']+f'{fidx}_*'
            pts_target = get_pts_from_reg(
                target_npy_regexr,
                target_ply_regexr,
                [cfg['ply_dir'], cfg['npy_dir']],
                samples_num,
                dem_res=dem_res,
                scale=cfg['scale']
            )
            # chamferDis = calc_chamferDis(pts_gt, pts_target)
            # method_dict[k]['chamfer_dis'].append(chamferDis)

            (closest_points,
            p2f_dis,
            triangle_id) = gt_mesh.nearest.on_surface(pts_target[:1000,:])
            method_dict[k]['p2f_dis'].append(p2f_dis.mean())


            ######################################## Statistic 
            # print(f"\r\n{k}-{fidx}:")
            # dis_perc = []
            # for perc in [0.1, 0.7, 0.9]:
            #     dis_perc.append(cal_perc_dis(p2f_dis, perc))
            # # print("dis perc(*100):", [round(d*1e6)/1e4 for d in dis_perc])
            # print("dis perc:", dis_perc)

            # num_perc = []
            # for t in [0.1, 0.5, 1.0]:
            #     num_perc.append(cal_perc_num(p2f_dis, t))
            # print("num perc:", [round(n*1e6)/1e4 for n in num_perc])

            # print("statics:", cal_statics(p2f_dis))
            

    for n, r_dict in method_dict.items():
        for m, v in r_dict.items():
            average_m = np.mean(v)
            print("{}-{}: {:.4f}".format(n,m,average_m))
        if len(r_dict)!=0:
            print('')
