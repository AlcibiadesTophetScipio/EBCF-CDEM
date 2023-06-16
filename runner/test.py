import torch
import logging
from tqdm import tqdm
from collections import defaultdict
from omegaconf import OmegaConf
import numpy as np
import os

###### custome package
import utils
from .utils import make_data_loader
from .pytorch_helper import model_state_dict_convert_auto

log = logging.getLogger(__name__)


def prepare_test(config, device):
    model_pth = config.test_spec.get('model_pth')
    if model_pth is None:
        raise Exception("Need to specify model_pth for loading model parameters.")
    elif model_pth == 'interpolator':
        model = utils.object_from_dict(
            OmegaConf.to_container(config.model_spec, resolve=True)
        ).to(device)
        return model, 0
    else:
        log.info(f"Loding file is {model_pth}")
        model_from = config.test_spec.get('model_from', 'load_file')
        if model_from == "load_file":
            sv_file = torch.load(model_pth)
            epoch = sv_file['epoch']
            log.info(f"Load history from {sv_file['epoch']}")
            model_spec = sv_file['model']
            model_state = model_spec.pop('sd')
            model = utils.object_from_dict(model_spec)
        elif model_from == 'config':
            model = utils.object_from_dict(
                OmegaConf.to_container(config.model_spec, resolve=True),
            )
            model_state = torch.load(model_pth, map_location='cpu')
            epoch=0
        
        model.load_state_dict(model_state_dict_convert_auto(model_state, os.environ["CUDA_VISIBLE_DEVICES"]))
        model.to(device)

        
        # else:
        #     model = utils.object_from_dict(
        #         OmegaConf.to_container(config.model_spec, resolve=True),
        #         sdfnet_from='from_old'
        #     ).to(device)
        #     model_sd_new = {}
        #     for k, v in model_sd.items():
        #         if k.find('imnet') != -1:
        #             k = k.replace('imnet', 'srnet')

        #         model_sd_new[k] = v
        #     model_sd = model_sd_new
        
        # model.load_state_dict(model_sd)

    return model, epoch

def v1(
    cfg,
    device,
    save_path,
    **kwargs
):
    ############ Preparing
    test_loader = make_data_loader(config=cfg, tag='test_dataset')
    model, load_epoch = prepare_test(config=cfg, device=device)

    visual_recs_dir = cfg.test_spec.get('visual_dir')
    if visual_recs_dir is not None:
        visual_recs_dir = os.path.join(save_path, visual_recs_dir+f'-{load_epoch}')
        if os.path.isdir(visual_recs_dir) is False:
            os.mkdir(visual_recs_dir)

    ############  Test procedure
    model.eval()
    testres_dict = defaultdict(list)
    pbar = tqdm(test_loader, leave=False, desc='train')
    for batch, batch_idx in pbar:
        # to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        # model val step, return val res
        with torch.no_grad(): 
            if hasattr(model, 'test_step'):
                test_res = model.test_step(
                    batch=batch,
                    batch_idx=batch_idx,
                    eval_bsize=cfg.test_spec.get('eval_bsize'),
                    save_dir=visual_recs_dir,
                )
            else:
                test_res = model(
                    batch=batch,
                    batch_idx=batch_idx,
                    flag='test',
                    eval_bsize=cfg.test_spec.get('eval_bsize'),
                    save_dir=visual_recs_dir,
                )
                # raise Exception(f"Need to specify test step method.")
        
        test_info = []
        if isinstance(test_res, dict):
            for k,v in test_res.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                
                test_info.append("{}: {:.4f}".format(k,v))
                testres_dict[k].append(v)

            test_info=', '.join(test_info)
        else:
            test_info = ''
        
        pbar.set_description(test_info)

    for k,v in testres_dict.items():
        log.info("{}: {:.4f}".format(k, np.mean(v)))
