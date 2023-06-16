import logging
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from functools import partial
from collections import defaultdict


import torch
from torch.utils.tensorboard import SummaryWriter

###### custome package
from .utils import \
    make_data_loader, \
    records_and_description, \
    monitor_do

import utils

log = logging.getLogger(__name__)


def prepare_training(config, device):
    train_resume_pth = config.train_spec.get('resume')
    if train_resume_pth is not None:
        sv_file = torch.load(train_resume_pth)
        log.info(f"Load history from {sv_file['epoch']}")

        model_spec = sv_file['model']
        model_sd = model_spec.pop('sd')
        model = utils.object_from_dict(model_spec).to(device)
        model.load_state_dict(model_sd)

        optimizer_spec = sv_file['optimizer']
        optimizer_sd = optimizer_spec.pop('sd')
        optimizer = utils.object_from_dict(
            optimizer_spec,
            params=model.parameters()
        )
        optimizer.load_state_dict(optimizer_sd)

        epoch_start = sv_file['epoch'] + 1
        if config.get('lr_scheduler_spec') is None:
            lr_scheduler = None
        else:
            lr_scheduler = utils.object_from_dict(
                OmegaConf.to_container(config.lr_scheduler_spec, resolve=True),
                optimizer=optimizer,
                last_epoch=sv_file['epoch']-1
            )
        
        log.info(f"Current learning rate is {optimizer.param_groups[0]['lr']}")
    else:
        model = utils.object_from_dict(
            OmegaConf.to_container(config.model_spec, resolve=True)
        ).to(device)
        optimizer = utils.object_from_dict(
            OmegaConf.to_container(config.optimizer_spec, resolve=True),
            params=model.parameters()
        )
        epoch_start = 1
        if config.get('lr_scheduler_spec') is None:
            lr_scheduler = None
        else:
            lr_scheduler = utils.object_from_dict(
                OmegaConf.to_container(config.lr_scheduler_spec, resolve=True),
                optimizer=optimizer
            )

    log.info('model: #type={}, #params={}'.format(
        type(model),
        utils.compute_num_params(model, text=True)
    ))
    return model, optimizer, epoch_start, lr_scheduler


def v1(
    cfg,
    device,
    save_path,
    # writer: SummaryWriter = None,
    writer_flag: bool = True
):
    ############ Preparing
    train_loader = make_data_loader(config=cfg, tag='train_dataset')
    if cfg.dataset_spec.get('val_dataset'):
        val_loader = make_data_loader(config=cfg, tag='val_dataset')
    else:
        val_loader = None
    
    model, optimizer, epoch_start, lr_scheduler = prepare_training(
        config=cfg,
        device=device
    )

    # Multi GPUs
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        model = torch.nn.parallel.DataParallel(model)

    # Epoch cycle
    timer = utils.Timer()
    epoch_val = cfg.train_spec.get('epoch_val', 1)
    epoch_save = cfg.train_spec.get('epoch_save', 10)
    epoch_max = cfg.train_spec.epoch_max

    writer = None
    if writer_flag:
        writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    monitor_spec = cfg.monitor_spec
    monitor_records = defaultdict(list)
    for tag, mod in monitor_spec.items():
        if mod not in ['max', 'min']:
            raise Exception(f"Wrong mod({tag}: {mod}) in monitor spec.")
        # # Don't add the init value 
        # if mod == 'max':
        #     monitor_records[tag].append(-1e18)
        # elif mod == 'min':
        #     monitor_records[tag].append(1e18)
        # else:
        #     raise Exception(f"Wrong mod({tag}: {mod}) in monitor spec.")

    for epoch in range(epoch_start, epoch_max+1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        ############  Training procedure
        loss_dict = defaultdict(list)
        pbar = tqdm(train_loader, leave=False, desc='train')
        for batch, batch_idx in pbar:
            # to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)

            # model train step, return loss
            model.train()
            if hasattr(model, 'training_step'):
                loss = model.training_step(
                    batch=batch,
                    batch_idx=batch_idx,
                )
            else:
                loss = model(batch, batch_idx, flag='train',epoch=epoch)
                for k,v in loss.items():
                    if len(v.shape)>0:
                        loss[k]=loss[k].mean()
                # raise Exception(f"Need to specify training step method.")
                # loss = torch.rand([1], requires_grad=True)

            # grad update
            optimizer.zero_grad()
            loss['loss'].backward()
            optimizer.step()

            # procedure description and records
            proc_descri, loss_dict = records_and_description(
                info_object=loss,
                info_dict=loss_dict
            )
            pbar.set_description(proc_descri)

        ############ LR scheduler step
        if lr_scheduler is not None:
            lr_scheduler.step()

        ############  Saving procedure
        if n_gpus > 1 and type(device)=='cuda':
            model_ = model.module
        else:
            model_ = model
        model_spec = OmegaConf.to_container(cfg.model_spec, resolve=True)
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = OmegaConf.to_container(cfg.optimizer_spec, resolve=True)
        optimizer_spec['sd'] = optimizer.state_dict()
        # lr_scheduler_spec = OmegaConf.to_container(cfg.lr_scheduler_spec, resolve=True)
        # lr_scheduler_spec['sd'] = lr_scheduler.state_dict()
        save_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            # 'scheduler': lr_scheduler_spec,
            'epoch': epoch
        }
        torch.save(save_file, os.path.join(save_path, 'epoch-last.pth'))
        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(save_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))
        
        # config monitor behaviors
        monitor_func = partial(
            monitor_do,
            save_file=save_file,
            save_path=save_path,
            epoch=epoch,
        )

        # monitor doing
        mesgs, monitor_records = monitor_func(
            monitor_records=monitor_records,
            monitor_spec=monitor_spec,
            info_dict=loss_dict,
        )
        for m in mesgs:
            log_info.append(m)


        ############  Validating procedure
        if (val_loader is not None) and (epoch_val is not None) and (epoch % epoch_val == 0):
            valres_dict = defaultdict(list)
            pbar = tqdm(val_loader, leave=False, desc='val')
            for batch, batch_idx in pbar:
                # to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)

                # model val step, return val res
                model.eval()
                with torch.no_grad(): 
                    if hasattr(model, 'validation_step'):
                        val_res = model.validation_step(
                            batch=batch,
                            batch_idx=batch_idx,
                        )
                    else:
                        val_res = model(batch,batch_idx,flag='val',)
                        for k,v in val_res.items():
                            if len(v.shape)>0:
                                val_res[k]=val_res[k].mean()
                        # raise Exception(f"Need to specify validation step method.")

                proc_descri, valres_dict = records_and_description(
                    info_object=val_res,
                    info_dict=valres_dict
                )
                pbar.set_description(proc_descri)
            
            # monitor doing
            mesgs, monitor_records = monitor_func(
                monitor_records=monitor_records,
                monitor_spec=monitor_spec,
                info_dict=valres_dict,
            )
            for m in mesgs:
                log_info.append(m)


        ############  Records
        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
        log.info(', '.join(log_info))
        if writer is not None:
            # lr 
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            # loss
            for k,v in monitor_records.items():
                if k.find('loss') != -1:
                    writer.add_scalar('loss/'+k, v[-1], epoch)
                else:
                    writer.add_scalar(k, v[-1], epoch)
            writer.flush()
