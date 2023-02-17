import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import utils
from datasets.utils import make_coord

from .meta_cls import DemRec_IF

class LIIF(nn.Module, DemRec_IF):
    def __init__(
        self,
        encoder,
        srnet,
        sdfnet,
        local_ensemble=True, feat_unfold=True, cell_decode=True,
        sdfnet_from: str=None,
        data_sub=[0.5], data_div=[0.5],
        loss_method = 'naive',
    ):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.data_sub = torch.FloatTensor(data_sub)
        self.data_div = torch.FloatTensor(data_div)
        self.sdfnet_from = sdfnet_from
        self.loss_method = loss_method

        self.encoder = encoder

        if not isinstance(srnet, nn.Module):
            srnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                srnet_in_dim *= 9
            # srnet_in_dim += pebd_dims # attach coord
            if self.cell_decode:
                srnet_in_dim += 2
            
            srnet_in_dim += 2 # image 2D coords
            self.srnet = utils.object_from_dict(
                srnet['spec'],
                in_dim=srnet_in_dim
            )
        else:
            self.srnet = srnet

        if sdfnet is None:
            self.sdfnet = None
        elif not isinstance(sdfnet, nn.Module):
            if sdfnet_from is None:
                sdfnet_in_dim = self.encoder.out_dim
                sdfnet_in_dim += 3 # dem 3D coords
            elif sdfnet_from == 'from_old':
                sdfnet_in_dim = srnet_in_dim+1
            
            self.sdfnet = utils.object_from_dict(
                sdfnet['spec'],
                in_dim=sdfnet_in_dim
            )
        else:
            self.sdfnet = sdfnet

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_elev(self, coord, cell):
        feat = self.feat

        if self.srnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(coord.device) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                # add pos encoding module
                # if self.pos_embedder is not None:
                #     rel_coord = self.pos_embedder(rel_coord)
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.srnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def query_sdf(
            self,
            sdf_coord,
            query_elev,
            liif_coord=None,
            cell=None
    ):
        feat = self.feat.detach()
        compos_coord = torch.cat([sdf_coord, query_elev], dim=-1)

        
        if self.sdfnet_from is None:
            coord_ = liif_coord.clone()
            q_feat = F.grid_sample(
                feat, coord_.flip(-1).unsqueeze(1),
                mode='bicubic', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
            inp = torch.cat([q_feat, compos_coord], dim=-1)
        else:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            coord_ = liif_coord.clone()
            q_feat = F.grid_sample(
                feat, coord_.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
            rel_cell = cell.clone()
            rel_cell[:, :, 0] *= feat.shape[-2]
            rel_cell[:, :, 1] *= feat.shape[-1]
            inp = torch.cat([q_feat, compos_coord, rel_cell], dim=-1)

        bs, q = sdf_coord.shape[:2]
        pred = self.sdfnet(inp.view(bs * q, -1)).view(bs, q, -1)

        return pred

    def forward(
        self, 
        inp, liif_coord, cell, # original need
        sdf_coord=None,
        ):

        self.gen_feat(inp)
        pred_elev = self.query_elev(liif_coord, cell)

        if self.sdfnet is None:
            return {'pred_elev': pred_elev}

        pred_sdf = self.query_sdf(
            sdf_coord=sdf_coord,
            query_elev=pred_elev,
            liif_coord=liif_coord,
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
            liif_coord=batch['coord'],
            cell=batch['cell'],
            sdf_coord=batch['global_coord']
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

        # 
        loss = F.l1_loss(train_res['pred_elev']-train_res['pred_sdf'], gt_elev)

        if self.loss_method == 'naive':
            loss = compos_loss
        elif self.loss_method == 'idpt':
            loss = sr_loss+sdf_loss
        elif self.loss_method == 'compos':
            loss = compos_loss + sr_loss
        elif self.loss_method == 'all':
            loss = compos_loss + sr_loss + sdf_loss
        elif self.loss_method == 'new':
            # new
            diff_sr = (train_res['pred_elev']-gt_elev).abs()
            # sdf_mask = torch.where(
            #     diff_sr>diff_sr.mean(),
            #     True,
            #     False
            # )
            # sdf_loss = F.mse_loss(train_res['pred_sdf'][sdf_mask], gt_sdf[sdf_mask])
            sdf_focus = torch.where(
                diff_sr>diff_sr.mean(),
                gt_sdf,
                torch.zeros_like(gt_sdf)
            )
            sdf_loss = F.mse_loss(train_res['pred_sdf'], sdf_focus)

            
            compos_loss = F.l1_loss(train_res['pred_elev']-train_res['pred_sdf'], gt_elev)
            loss = compos_loss + sr_loss 


        # loss = torch.rand([1], requires_grad=True)
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
            liif_coord=batch['coord'],
            cell=batch['cell'],
            sdf_coord=batch['global_coord']
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

        if hasattr(self.encoder,'args') and self.encoder.args.n_colors == 3:
            scale = batch['add_args'][..., -2]
            scale = scale.view(-1,1,1,1).expand_as(inp)
            bias = batch['add_args'][..., -1]
            bias = bias.view(-1,1,1,1).expand_as(inp)
            inp = torch.cat([inp, 1.0/scale, 1.0/bias], dim=1)

        model_output = self.forward(
            inp=inp,
            liif_coord=batch['coord'],
            cell=batch['cell'],
            sdf_coord=batch['global_coord']
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
