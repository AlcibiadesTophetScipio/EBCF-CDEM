from multiprocessing.util import is_exiting
import matplotlib.pyplot as plt
import imageio
import glob
from pathlib import Path
from omegaconf import OmegaConf
import math
import numpy as np
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from collections import defaultdict
import trimesh
from pytorch3d.loss import chamfer_distance
import torch
from tqdm import tqdm
from vedo.utils import trimesh2vedo
import vedo
import copy
import random

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

def save_vedo_cmap(vedo_ply, vedo_png, gt_mesh, target_ply_regexr, dem_res, v_max):
    if Path(vedo_ply).is_file():
        return
    else:
        target_mesh_pth = glob.glob(target_ply_regexr)[0]
        target_mesh = trimesh.load(target_mesh_pth)
        target_mesh.vertices = target_mesh.vertices*[dem_res,dem_res,1]
        (closest_points,
        target_p2f_dis,
        triangle_id) = gt_mesh.nearest.on_surface(target_mesh.vertices)

        # visual mesh normalization
        visual_mesh=copy.deepcopy(gt_mesh)
        vtx_max=gt_mesh.vertices.max(axis=0)
        vtx_min=gt_mesh.vertices.min(axis=0)
        visual_mesh.vertices = ((visual_mesh.vertices-vtx_min)/(vtx_max-vtx_min) - 0.5)*2.0
        # visual_mesh.vertices = visual_mesh.vertices/[dem_res,dem_res,1]
        # print(visual_mesh.vertices[:,0].max(), visual_mesh.vertices[:,1].max(), v_max)

        vis_vedomesh = trimesh2vedo(visual_mesh)
        vis_vedomesh.cmap(
            input_array=target_p2f_dis,
            cname='jet',
            vmin=0.0, 
            vmax=v_max,
        ).addScalarBar()
        # vis_vedomesh.write(vedo_ply)
        vedomesh = vis_vedomesh
        
    vedo.show(
        vedomesh, 
    ).screenshot(vedo_png).clear()

if __name__ == '__main__':
    cfg = OmegaConf.load('rec-ply.yaml')
    # cfg = OmegaConf.load('encoders-ply.yaml')
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
    # dem_res = 1
    samples_num= 128**2
    print(
        "test dataset: {}, sr scale: {}, dem resolution: {}."
        .format(cfg['test_dataset'], sr_scale, dem_res)
    )

    v_max = 1.0
    if cfg['test_dataset'] == 'astgtmv003':
        if sr_scale==4:
            v_max = 20.0
        elif sr_scale==8:
            v_max = 40.0
        elif sr_scale==16:
            v_max = 80.0
    elif cfg['test_dataset'] == 'pyrenees':
        if sr_scale==4:
            v_max = 2.0
        elif sr_scale==8:
            v_max = 5.0
        elif sr_scale==16:
            v_max = 10.0

    method_list = [
        # 'bilinear',
        'bicubic', 
        # 'mapif-global', 
        # 'mapif', 
        # 'mapif-sdf',
        'metasr',
        # 'metasr-sdf',
        # 'liif-naive',
        'liif',
        # 'liif-sdf',
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

    vedo.show(
        axes=13,
        offscreen=True,
        roll=-90.0
    )
    filenames = glob.glob(gt_ply_dir+'*_inp.ply')
    selections=list(range(len(filenames)))
    selections=random.sample(selections, 10)
    selections=[str(s) for s in selections]
    print("Selected id:", selections)
    for fname in tqdm(filenames):
        fstem = fname.split('/')[-1]
        fidx = fstem.split('_')[0]

        ##### Select
        # if fidx not in ['8','145','27','34']:
        # if fidx not in selections:
        # if fidx not in ['145','126']:
        #     continue

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
            # target_npy_regexr = cfg['npy_dir']+'/'+rec_dirs[f'{k}_dir']+f'{fidx}_*'
            target_ply_regexr = cfg['ply_dir']+'/'+rec_dirs[f'{k}_dir']+f'{fidx}_*'
            # pts_target = get_pts_from_reg(
            #     target_npy_regexr,
            #     target_ply_regexr,
            #     [cfg['ply_dir'], cfg['npy_dir']],
            #     samples_num,
            #     dem_res=dem_res,
            #     scale=cfg['scale']
            # )

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
            
            ######################################## 3D error map
            # vedo_dir = "/data/syao/Exps/vedo-show"
            vedo_dir = cfg['vedo_dir']+f"/{cfg['test_dataset']}_x{sr_scale}_v{v_max}"
            Path(vedo_dir).mkdir(parents=True, exist_ok=True)
            vedo_ply=vedo_dir+f"/{cfg['test_dataset']}_{k}x{sr_scale}_{fidx}.ply"
            vedo_png=vedo_dir+f"/{cfg['test_dataset']}_{k}x{sr_scale}_{fidx}.png"
            save_vedo_cmap(vedo_ply, vedo_png, gt_mesh, target_ply_regexr, dem_res, v_max)
            print(f"Saved {vedo_png}.")

    for n, r_dict in method_dict.items():
        for m, v in r_dict.items():
            average_m = np.mean(v)
            print("{}-{}: {:.4f}".format(n,m,average_m))
        if len(r_dict)!=0:
            print('')
