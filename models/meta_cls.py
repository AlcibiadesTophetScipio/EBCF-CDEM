from abc import abstractmethod, ABC
import math
import torch
import torch.nn.functional as F

import utils

class DemRec_IF(ABC):
    @abstractmethod
    def forward(self):
        pass

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def gen_latvec(self, sample_grid, mode='bicubic'):
        feat = self.feat

        grid_ = sample_grid.clone()
        gen_latvec = F.grid_sample(
            feat, grid_.flip(-1).unsqueeze(1),
            mode=mode, align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)

        return gen_latvec


    def recfunc(
        self, 
        batch: dict,
        batch_idx: torch.Tensor,
        pred_elev: torch.Tensor,
        save_dir: str=None,
        use_original_gt=False,
        )->dict:

        # reshape for evaluating
        ih, iw = batch['inp'].shape[-2:]
        s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
        shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), batch['gt'].shape[-1]]
        pred_elev = pred_elev.view(*shape) \
            .permute(0, 3, 1, 2).contiguous()
        gt_elev = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()

        # from (0,1) to real value
        dem_scale, dem_bias = batch['add_args'][...,0], batch['add_args'][...,1]
        rec = pred_elev*dem_scale + dem_bias
        if use_original_gt:
            original = gt_elev
        else:
            original =  gt_elev*dem_scale + dem_bias

        # statistics
        scale=int(s)
        psnr=utils.calc_psnr(
            sr=pred_elev,
            hr=gt_elev,
            scale=scale
        )
        shave=scale
        diff = (rec-original)[..., shave:-shave, shave:-shave]

        # save recs
        if save_dir is not None:
            # utils.data2dem(batch['inp'], save_dir+'/{}_inp.png'.format(batch_idx.item()))
            # utils.data2dem(original, save_dir+'/{}_gt.png'.format(batch_idx.item()))
            # utils.data2dem(rec, save_dir+'/{}_rec-psnr_{:.4f}-mae_{:.4f}-mse_{:.4f}.png'.format(
            #     batch_idx.item(), psnr.item(),
            #         diff.abs().mean().item(),
            #         diff.pow(2).mean().item()
            #     )
            # )
            # utils.data2heatmap(diff.abs(), save_dir+'/{}_rec-psnr_{:.4f}-mae_{:.4f}-mse_{:.4f}.png'.format(
            #     batch_idx.item(), psnr.item(),
            #         diff.abs().mean().item(),
            #         diff.pow(2).mean().item()
            #     ),
            #     max_value=10.0
            # )

            # if scale == 1:
            if hasattr(self, 'interpolation') and self.interpolation == 'identity':
                utils.data2tif(batch['inp']*dem_scale + dem_bias, save_dir+'/{}_inp.tif'.format(batch_idx.item()))
            else:
                utils.data2tif(rec, save_dir+'/{}_rec-psnr_{:.4f}-mae_{:.4f}-mse_{:.4f}.tif'.format(
                    batch_idx.item(), psnr.item(),
                        diff.abs().mean().item(),
                        diff.pow(2).mean().item()
                    )
                )
            
            
            # utils.data2dem(
            #     torch.cat([rec, original], dim=-1),
            #     save_dir+'/{}_compos-psnr_{:.4f}-mae_{:.4f}-mse_{:.4f}.png'.format(
            #         batch_idx.item(), psnr.item(),
            #         diff.abs().mean().item(),
            #         diff.pow(2).mean().item()
            #     )
            # )

            # utils.data2dem(
            #     diff.abs(),
            #     save_dir+'/{}-heatmap_mae.png'.format(batch_idx.item()),
            #     heatmap_flag=True
            # )
            # utils.data2dem(
            #     diff.pow(2),
            #     save_dir+'/{}-heatmap_mse.png'.format(batch_idx.item()),
            #     heatmap_flag=True
            # )

        return {
            'psnr': psnr,
            'mae': diff.abs().mean(),
            'mse': diff.pow(2).mean()
        }
