import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import utils
from datasets.utils import make_coord

from .meta_cls import DemRec_IF


class MetaSR(nn.Module, DemRec_IF):
    def __init__(
        self,
        encoder,
        srnet,
        data_sub=[0.5], data_div=[0.5],
        interp_mode=None,
        loss_method=None,
    ):
        super().__init__()
        self.data_sub = torch.FloatTensor(data_sub)
        self.data_div = torch.FloatTensor(data_div)
        self.interp_mode = interp_mode
        self.loss_method = loss_method

        self.encoder = encoder

        if hasattr(self.encoder,'args'):
            self.co_mat = self.encoder.args.n_colors
        else:
            self.co_mat = self.encoder.in_dim
            
        self.srnet = utils.object_from_dict(
                srnet['spec'],
                in_dim=3,
                out_dim=self.encoder.out_dim*9*self.co_mat
            )

    def query_elev(self, origin_inp, coord, cell):
        feat = self.feat
        feat = F.unfold(feat, 3, padding=1).view(
            feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda()
        feat_coord[:, :, 0] -= (2 / feat.shape[-2]) / 2
        feat_coord[:, :, 1] -= (2 / feat.shape[-1]) / 2
        feat_coord = feat_coord.permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        coord_ = coord.clone()
        coord_[:, :, 0] -= cell[:, :, 0] / 2
        coord_[:, :, 1] -= cell[:, :, 1] / 2
        coord_q = (coord_ + 1e-6).clamp(-1 + 1e-6, 1 - 1e-6)
        q_feat = F.grid_sample(
            feat, coord_q.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)
        q_coord = F.grid_sample(
            feat_coord, coord_q.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1)

        rel_coord = coord_ - q_coord
        rel_coord[:, :, 0] *= feat.shape[-2] / 2
        rel_coord[:, :, 1] *= feat.shape[-1] / 2

        r_rev = cell[:, :, 0] * (feat.shape[-2] / 2)

        # add pos encoding module
        # if self.pos_embedder is not None:
        #     rel_coord = self.pos_embedder(rel_coord)

        inp = torch.cat([rel_coord, r_rev.unsqueeze(-1)], dim=-1)

        bs, q = coord.shape[:2]
        pred = self.srnet(inp.view(bs * q, -1)).view(bs * q, feat.shape[1], self.co_mat)
        pred = torch.bmm(q_feat.contiguous().view(bs * q, 1, -1), pred)
        pred = pred.view(bs, q, -1)

        # get horiz line for preding bias
        if self.interp_mode is not None:
            grid_ = coord.clone()
            horiz_line = F.grid_sample(
                origin_inp, grid_.flip(-1).unsqueeze(1),
                mode=self.interp_mode, align_corners=False
            )[:, :, 0, :].permute(0, 2, 1)

            elev_value = horiz_line+pred
            return {
                'pred_sdf': pred,
                'pred_elev': elev_value,
                'horiz_line': horiz_line,
            }
        else:
            return {'pred_elev': pred}

    def forward(self, batch, batch_idx, flag, **kwargs):
        data_sub = self.data_sub.to(batch['inp'].device)
        data_div = self.data_div.to(batch['inp'].device)
        inp= (batch['inp'] - data_sub) / data_div

        self.gen_feat(inp)
        preds = self.query_elev(
            origin_inp=inp,
            coord=batch['coord'],
            cell=batch['cell']
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

        if self.interp_mode is None:
            return {
                'loss': sr_loss,
                'sr_loss': sr_loss,
            }
        else:
            gt_bias = gt_elev - preds['horiz_line']
            max_gt = gt_bias.max(dim=-2,keepdim=True)[0]
            min_gt = gt_bias.min(dim=-2,keepdim=True)[0]

            scale = 1/(max_gt-min_gt+1.0e-10)
            re_bias = torch.where(scale>1000, gt_bias, ((gt_bias-min_gt)*scale - 0.5)*2.0)
            re_pred = torch.where(scale>1000, preds['pred_sdf'], ((preds['pred_sdf']-min_gt)*scale- 0.5)*2.0)
            # bias_loss = F.mse_loss(re_pred, re_bias)
            bias_loss = F.l1_loss(re_pred, re_bias)
            if self.loss_method is None:
                loss = bias_loss
            elif self.loss_method == 'compos':
                loss = sr_loss+0.1*bias_loss
            else:
                raise("Wrong loss method.")

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