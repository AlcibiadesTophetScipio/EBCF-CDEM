import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import transforms

import utils
from .meta_cls import DemRec_IF

class InterPolator(nn.Module, DemRec_IF):
    def __init__(self, interpolation='bicubic'):
        super().__init__()

        if interpolation == 'bicubic':
            self.interpolation = transforms.InterpolationMode.BICUBIC
        elif interpolation == 'bilinear':
            self.interpolation = transforms.InterpolationMode.BILINEAR
        elif interpolation == 'identity':
            self.interpolation = 'identity'
        else:
            raise Exception('Please align interpolation method')

    def forward(self, target_shape, lr):
        # sr = transforms.Resize(hr.shape[-2:], self.interpolation)(lr)
        if self.interpolation == 'identity':
            return lr
        sr = transforms.Resize(target_shape, self.interpolation)(lr)
        return sr

    def test_step(
        self,
        batch,
        batch_idx,
        eval_bsize=None,
        save_dir=None,
        **kwargs
    ):
        inp = batch['inp']

        # reshape for evaluating
        ih, iw = batch['inp'].shape[-2:]
        s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
        # shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), batch['gt'].shape[-1]]
        # gt_elev = batch['gt'].view(*shape) \
        #         .permute(0, 3, 1, 2).contiguous()

        # get interpolation results
        sr = self.forward(
                    target_shape=[round(ih * s), round(iw * s)],
                    lr=inp
                )

        return self.recfunc(
                batch=batch,
                batch_idx=batch_idx,
                pred_elev=sr,
                save_dir=save_dir,
            )

        # dem_scale, dem_bias = batch['add_args'][...,0], batch['add_args'][...,1]
        # rec = sr*dem_scale + dem_bias
        # original =  gt_elev*dem_scale + dem_bias

        # # statistics
        # shave=int(s)
        # psnr=utils.calc_psnr(
        #     sr=sr,
        #     hr=gt_elev,
        #     scale=int(s)
        # )
        # diff = (rec-original)[..., shave:-shave, shave:-shave]

        # # save recs
        # if save_dir is not None:
        #     # utils.data2dem(original, save_dir+'/{}_gt.png'.format(batch_idx.item()))
        #     # utils.data2dem(rec, save_dir+'/{}_rec.png'.format(batch_idx.item()))
        #     utils.data2dem(
        #         torch.cat([rec, original], dim=-1),
        #         save_dir+'/{}_compos-psnr_{:.4f}-mae_{:.4f}-mse_{:.4f}.png'.format(
        #             batch_idx.item(), psnr.item(),
        #             diff.abs().mean().item(),
        #             diff.pow(2).mean().item()
        #         )
        #     )

        #     utils.data2dem(
        #         diff.abs(),
        #         save_dir+'/{}-heatmap_mae.png'.format(batch_idx.item()),
        #         heatmap_flag=True
        #     )
        #     utils.data2dem(
        #         diff.pow(2),
        #         save_dir+'/{}-heatmap_mse.png'.format(batch_idx.item()),
        #         heatmap_flag=True
        #     )

        # return {
        #     'psnr': psnr,
        #     'mae': diff.abs().mean(),
        #     'mse': diff.pow(2).mean()
        # }

        