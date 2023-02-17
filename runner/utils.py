
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import os
from collections import defaultdict
from typing import Tuple
import numpy as np
import logging 

###### custome package
import utils

log = logging.getLogger(__name__)


def make_data_loader(config, tag):
    dataset_spec = config.dataset_spec.get(tag)
    dataset = utils.object_from_dict(
            OmegaConf.to_container(dataset_spec, resolve=True)
    )
    log.info(f"The num of {tag} is {len(dataset)}")

    if tag == 'test_dataset':
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            pin_memory=True
        )
    else:
        return DataLoader(
            dataset,
            batch_size=config.train_spec.batch_size,
            num_workers=config.train_spec.num_workers,
            shuffle=(tag=='train_dataset'),
            pin_memory=True
        )

def check_value_and_save(
    res_dict,
    res_value,
    save_file,
    save_path,
    epoch,
    mod='min',
    tag='val',
) -> bool:
    if res_dict.get(tag) is None:
        raise Exception(f'Wrong tag({tag}) for checking val res.')

    file_path = os.path.join(save_path, 'epoch-{}_{}-{:.4f}.pth'.format(epoch, tag, res_value))
    save_flag = False
    if mod == 'min':
        if res_value <= np.min(res_dict[tag]):
            save_flag = True
    elif mod == 'max':
        if res_value >= np.max(res_dict[tag]):
            save_flag = True
    else:
        raise Exception(f'Wrong mod({mod}) for checking val res.')
    
    if save_flag:
        torch.save(save_file, file_path)
    
    return save_flag
                             
def records_and_description(
    info_object, 
    info_dict: defaultdict
    ) -> Tuple[str, defaultdict]:
    # procedure description and records
    proc_descri = ''
    if isinstance(info_object, dict):
        proc_descri = []
        for k,v in info_object.items():
            info_dict[k].append(v.item())
            proc_descri.append(
                '{}: {:.4f}'.format(k, np.mean(info_dict[k]))
            )
        proc_descri = ', '.join(proc_descri)
    elif isinstance(info_object, torch.Tensor):
        info_dict['loss'].append(info_object.item())
        proc_descri = 'loss: {:.4f}'.format(np.mean(info_dict['loss']))
    else:
        info_dict['loss'].append(info_object)
        proc_descri = 'loss: {:.4f}'.format(np.mean(info_dict['loss']))

    return proc_descri, info_dict

def monitor_do(
    monitor_records: defaultdict,
    monitor_spec: dict,
    info_dict: dict,
    **kwargs,
) -> Tuple[list, defaultdict]:
    log_info = []
    for k,v in info_dict.items():
        query_value = np.mean(v)
        monitor_records[k].append(query_value)
        if monitor_spec.get(k) is not None:
            log_info.append("{}: {:.4f}".format(k, query_value))

            check_value_and_save(
                res_dict=monitor_records,
                res_value=query_value,
                tag=k,
                mod=monitor_spec[k],
                # save_file=save_file,
                # save_path=save_path,
                # epoch=epoch,
                **kwargs,
            )

    return log_info, monitor_records


