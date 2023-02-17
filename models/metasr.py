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
        sdfnet,
        data_sub=[0.5], data_div=[0.5],
        loss_method = 'naive',
    ):
        super().__init__()
        self.data_sub = torch.FloatTensor(data_sub)
        self.data_div = torch.FloatTensor(data_div)
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
        
        if sdfnet is None:
            self.sdfnet = None
        else:
            sdfnet_in_dim = self.encoder.out_dim
            sdfnet_in_dim += 3 # dem 3D coords
            self.sdfnet = utils.object_from_dict(
                sdfnet['spec'],
                in_dim=sdfnet_in_dim
            )

    def query_elev(self, coord, cell):
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
        return pred

    def query_sdf(
            self,
            sdf_coord,
            query_elev,
            liif_coord=None,
            cell=None
    ):
        feat = self.feat.detach()
        compos_coord = torch.cat([sdf_coord, query_elev], dim=-1)
        

        coord_ = liif_coord.clone()
        q_feat = F.grid_sample(
            feat, coord_.flip(-1).unsqueeze(1),
            mode='bicubic', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
        inp = torch.cat([q_feat, compos_coord], dim=-1)


        bs, q = sdf_coord.shape[:2]
        pred = self.sdfnet(inp.view(bs * q, -1)).view(bs, q, -1)

        return pred

    def forward(
        self, 
        inp, 
        cell,
        local_coord, 
        global_coord,
        ):

        self.gen_feat(inp)
        pred_elev = self.query_elev(local_coord, cell)

        if self.sdfnet is None:
            return {
                'pred_elev': pred_elev,
            }

        pred_sdf = self.query_sdf(
            sdf_coord=global_coord,
            query_elev=pred_elev,
            liif_coord=local_coord,
            cell=cell,
        )
        return {
            'pred_elev': pred_elev,
            'pred_sdf': pred_sdf,
            }

    def training_step(self, batch, **kwargs):
        data_sub = self.data_sub.to(batch['inp'].device)
        data_div = self.data_div.to(batch['inp'].device)

        inp= (batch['inp'] - data_sub) / data_div
        train_res = self.forward(
            inp=inp,
            cell=batch['cell'],
            global_coord=batch['global_coord'],
            local_coord=batch['coord']
        )

        gt_elev = (batch['gt'] - data_sub) / data_div
        sr_loss = F.l1_loss(train_res['pred_elev'], gt_elev)
        if self.sdfnet is None:
            return {
                'loss': sr_loss,
                'sr_loss': sr_loss,
            }

        gt_sdf = train_res['pred_elev'] - gt_elev
        sdf_loss = F.mse_loss(train_res['pred_sdf'], gt_sdf)

        compos_loss = F.l1_loss(train_res['pred_elev']-train_res['pred_sdf'], gt_elev)

        if self.loss_method == 'naive':
            loss = compos_loss
        elif self.loss_method == 'compos':
            loss = compos_loss + sr_loss
        
        return {
            'loss': loss,
            'sr_loss': sr_loss,
            'sdf_loss': sdf_loss,
            'compos_loss': compos_loss,
        }

    def validation_step(self, batch, **kwargs):

        data_sub = self.data_sub.to(batch['inp'].device)
        data_div = self.data_div.to(batch['inp'].device)

        inp= (batch['inp'] - data_sub) / data_div
        model_output = self.forward(
            inp=inp,
            cell=batch['cell'],
            global_coord=batch['global_coord'],
            local_coord=batch['coord'],
        )
        pred_elev = model_output['pred_elev']

        if self.sdfnet is None:
            pred_elev = pred_elev*data_div + data_sub
            psnr_pred = utils.calc_psnr(sr=pred_elev.clamp(0,1), hr=batch['gt'])
            return {
                'psnr_pred': psnr_pred,
            }

        else:
            pred_sdf = model_output['pred_sdf']
            enhance_elev = pred_elev-pred_sdf

            # recover
            pred_elev = pred_elev*data_div + data_sub
            enhance_elev = enhance_elev*data_div + data_sub
            enhance_elev.clamp_(0, 1)

            psnr_pred = utils.calc_psnr(sr=pred_elev, hr=batch['gt'])
            psnr_enhance = utils.calc_psnr(sr=enhance_elev, hr=batch['gt'])

            return {
                'psnr_pred': psnr_pred,
                'psnr_enhance': psnr_enhance,
            }

    def test_step(
        self,
        batch,
        batch_idx,
        eval_bsize=None,
        save_dir=None,
        **kwargs
    ):
        data_sub = self.data_sub.to(batch['inp'].device)
        data_div = self.data_div.to(batch['inp'].device)
        inp= (batch['inp'] - data_sub) / data_div

        model_output = self.forward(
                inp=inp,
                cell=batch['cell'],
                global_coord=batch['global_coord'],
                local_coord=batch['coord']
            )

        pred_elev = model_output['pred_elev']
        if self.sdfnet is None:
            pred_elev = pred_elev*data_div + data_sub
            return self.recfunc(
                batch=batch,
                batch_idx=batch_idx,
                pred_elev=pred_elev.clamp(0, 1),
                save_dir=save_dir,
            )
        else:
            pred_sdf = model_output['pred_sdf']
            enhance_elev = pred_elev-pred_sdf

            # from (-1,1) to (0,1)
            enhance_elev = enhance_elev*data_div + data_sub
            enhance_elev.clamp_(0,1)

            return self.recfunc(
                batch=batch,
                batch_idx=batch_idx,
                pred_elev=enhance_elev,
                save_dir=save_dir,
            )