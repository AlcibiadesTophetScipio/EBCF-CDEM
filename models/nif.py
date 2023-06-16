
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import default_collate as collate_fn

import utils
from datasets.utils import make_coord

from .meta_cls import DemRec_IF

class EBCFF(nn.Module, DemRec_IF):
    def __init__(
        self,
        encoder,
        biasnet,
        data_sub=[0.5], data_div=[0.5],
        make_coord_local=False,
        interp_mode='none',
        posEmbeder=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.data_sub = torch.FloatTensor(data_sub)
        self.data_div = torch.FloatTensor(data_div)
        self.make_coord_local = make_coord_local
        self.interp_mode = interp_mode

        if posEmbeder is None:
            biasnet_in_dim = self.encoder.out_dim +2
            self.posEmbeder = None
        else:
            self.posEmbeder = utils.object_from_dict(
                posEmbeder['spec'],
                )
            biasnet_in_dim = self.posEmbeder.get_output_dim(2) + self.encoder.out_dim
            
        self.biasnet = utils.object_from_dict(
            biasnet['spec'],
            in_dim=biasnet_in_dim
        )

    def query_bias(self, q_samples):
        bs, q = q_samples.shape[:2]
        pred = self.biasnet(q_samples.view(bs * q, -1)).view(bs, q, -1)
        return pred

    def pred(self, inp, global_coord, local_coord):
        self.gen_feat(inp)

        if self.make_coord_local:
            latvec = self.gen_latvec(
                sample_grid=local_coord,
            )
            coord_ = local_coord.clone()
            feat_coord = make_coord(self.feat.shape[-2:],flatten=False).to(inp.device) \
                .permute(2,0,1) \
                .unsqueeze(0).expand(self.feat.shape[0], 2, *self.feat.shape[-2:])
            q_coord = F.grid_sample(
                feat_coord, coord_.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0,2,1)
            rel_coord = local_coord-q_coord
            rel_coord[:,:,0] *= self.feat.shape[-2]
            rel_coord[:,:,1] *= self.feat.shape[-1]
            if self.posEmbeder:
                rel_coord = self.posEmbeder(rel_coord)
            q_samples = torch.cat([latvec, rel_coord], dim=-1)
        else:
            latvec = self.gen_latvec(
                sample_grid=local_coord,
            )
            rel_coord = local_coord
            if self.posEmbeder:
                rel_coord = self.posEmbeder(rel_coord)
            q_samples = torch.cat([latvec, rel_coord], dim=-1)
            # q_samples = torch.cat([latvec, global_coord], dim=-1)

        bias_value = self.query_bias(q_samples)

        if self.interp_mode !='none':
            # get horiz line for preding sdf
            grid_ = local_coord.clone()
            horiz_line = F.grid_sample(
                inp, grid_.flip(-1).unsqueeze(1),
                mode=self.interp_mode, align_corners=False
            )[:, :, 0, :].permute(0, 2, 1)

            elev_value = horiz_line+bias_value

            return {
                'pred_bias': bias_value,
                'pred_elev': elev_value,
                'horiz_line': horiz_line,
            }
        else:
            return {'pred_elev': bias_value,}

    def forward(self, batch, batch_idx, flag, **kwargs):
        data_sub = self.data_sub.to(batch['inp'].device)
        data_div = self.data_div.to(batch['inp'].device)
        inp= (batch['inp'] - data_sub) / data_div

        preds = self.pred(
            inp=inp,
            global_coord=batch['global_coord'],
            local_coord=batch['coord']
        )

        if flag == 'train':
            gt_elev = (batch['gt'] - data_sub) / data_div
            return self.train_mod(preds=preds, gt_elev=gt_elev)
        elif flag == 'val':
            normal_pred = preds['pred_elev']*data_div + data_sub
            normal_gt = batch['gt']
            return self.val_mod(normal_pred, normal_gt)
        elif flag == 'test':
            normal_pred = preds['pred_elev']*data_div + data_sub
            normal_pred.clamp_(0,1)

            return self.recfunc(
                batch=batch,
                batch_idx=batch_idx,
                pred_elev=normal_pred,
                save_dir=kwargs['save_dir'],
            )
            
        else:
            raise Exception('Wrong flag in model.')

    def train_mod(self, preds, gt_elev):
        sr_loss = F.l1_loss(preds['pred_elev'], gt_elev)

        if self.interp_mode == 'none':
            return {
                'loss': sr_loss,
                'sr_loss': sr_loss,
            }
        else:
            gt_bias = gt_elev - preds['horiz_line']
            bias_loss = F.l1_loss(preds['pred_bias'], gt_bias)

            loss = bias_loss

            return {
                'loss': loss,
                'sr_loss': sr_loss,
                'bias_loss': bias_loss,
            }

    def val_mod(self, normal_pred, normal_gt):
        psnr = utils.calc_psnr(normal_pred, normal_gt)

        return {
            'psnr': psnr
        }


