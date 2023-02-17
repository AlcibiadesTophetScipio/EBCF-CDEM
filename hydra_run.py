
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf
import logging
import os
import shutil
import numpy as np
import torch

###### custome package
import utils

log = logging.getLogger(__name__)

def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

@hydra.main(
    version_base=None,
    config_path="configs/",
    config_name="config"
)
def hydra_app(cfg: DictConfig) -> None:
    save_path = os.getcwd()
    if cfg.get('ensure_path', False):
        ensure_path(save_path, remove=True)

    log.info(f'Working directory: {save_path}')
    log.info(f"Solvent Name: {cfg.exp_name}")
    log.info(f"Dataset name: {cfg.dataset_spec.name}")

    device = cfg.get('device')
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Runing on: {device}")
    else:
        if device.startswith('cuda:'):
            os.environ['CUDA_VISIBLE_DEVICES']=device.split(':')[-1].strip(' ')
            device = torch.device('cuda')
            log.info(f"Runing on CUDA: {os.environ['CUDA_VISIBLE_DEVICES']}")
        else:
            device = torch.device(device)
            log.info(f"Runing on: {device}")
        
    
    # with open_dict(cfg):
    #     cfg.device = device

    runner = utils.object_from_dict(
        OmegaConf.to_container(cfg.run_spec.runner, resolve=True),
        return_type=cfg.run_spec.get('return_type', 'object')
    )
    runner(cfg=cfg, device=device, save_path=save_path)


if __name__ == "__main__":
    hydra_app()
