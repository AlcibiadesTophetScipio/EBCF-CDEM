
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import default_collate as collate_fn

import utils
from datasets.utils import make_coord

from .meta_cls import DemRec_IF

class MAPIF(nn.Module, DemRec_IF):
    def __init__(
        self,
        encoder,
        mapif,
        data_sub=[0.5], data_div=[0.5],
        make_coord_local=False,
        **kwargs,
    ):
        super().__init__()
        self.make_coord_local = make_coord_local
        self.data_sub = torch.FloatTensor(data_sub)
        self.data_div = torch.FloatTensor(data_div)
        self.encoder = encoder

        mapif_in_dim = self.encoder.out_dim+2
        self.mapif = utils.object_from_dict(
                mapif['spec'],
                in_dim=mapif_in_dim
            )

    # def gen_feat(self, inp):
    #     self.feat = self.encoder(inp)
    #     return self.feat

    # def gen_latvec(self, sample_grid, mode='bicubic'):
    #     feat = self.feat

    #     grid_ = sample_grid.clone()
    #     gen_latvec = F.grid_sample(
    #         feat, grid_.flip(-1).unsqueeze(1),
    #         mode=mode, align_corners=False
    #     )[:, :, 0, :].permute(0, 2, 1)

    #     return gen_latvec
    
    def gen_query_value(self, q_samples):
        bs, q = q_samples.shape[:2]
        pred = self.mapif(q_samples.view(bs * q, -1)).view(bs, q, -1)
        return pred

    def forward(self, inp, global_coord, local_coord):
        self.gen_feat(inp)

        if self.make_coord_local:
            latvec = self.gen_latvec(
                sample_grid=local_coord,
                mode='nearest',
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
            q_samples = torch.cat([latvec, rel_coord], dim=-1)
        else:
            latvec = self.gen_latvec(sample_grid=local_coord)
            q_samples = torch.cat([latvec, global_coord], dim=-1)

        q_value = self.gen_query_value(q_samples)

        return {'pred_elev': q_value}

    def training_step(self, batch, **kwargs):
        self.data_sub = self.data_sub.to(batch['inp'].device)
        self.data_div = self.data_div.to(batch['inp'].device)

        inp= (batch['inp'] - self.data_sub) / self.data_div
        pred = self.forward(
            inp=inp,
            global_coord=batch['global_coord'],
            local_coord=batch['coord']
        )
        gt_normal = (batch['gt'] - self.data_sub) / self.data_div
        loss = F.l1_loss(pred['pred_elev'], gt_normal)

        return {
            'loss': loss
        }

    def validation_step(self, batch, **kwargs):
        self.data_sub = self.data_sub.to(batch['inp'].device)
        self.data_div = self.data_div.to(batch['inp'].device)

        inp= (batch['inp'] - self.data_sub) / self.data_div
        pred = self.forward(
            inp=inp,
            global_coord=batch['global_coord'],
            local_coord=batch['coord']
        )

        # recover
        pred_01 = pred['pred_elev']*self.data_div + self.data_sub
        psnr = utils.calc_psnr(pred_01, batch['gt'])

        return {
            'psnr': psnr
        }

    def test_step(
        self,
        batch,
        batch_idx,
        eval_bsize=None,
        save_dir=None,
        **kwargs
    ):
        self.data_sub = self.data_sub.to(batch['inp'].device)
        self.data_div = self.data_div.to(batch['inp'].device)

        inp= (batch['inp'] - self.data_sub) / self.data_div

        if self.encoder.args.n_colors == 3:
            scale = batch['add_args'][..., -2]
            scale = scale.view(-1,1,1,1).expand_as(inp)
            bias = batch['add_args'][..., -1]
            bias = bias.view(-1,1,1,1).expand_as(inp)
            inp = torch.cat([inp, 1.0/scale, 1.0/bias], dim=1)

        model_output = None
        if eval_bsize is None:
            model_output = self.forward(
                inp=inp,
                global_coord=batch['global_coord'],
                local_coord=batch['coord']
            )['pred_elev']
        else:
            self.gen_feat(inp)
            n = batch['coord'].shape[1]
            ql = 0
            elevs = []
            while ql < n:
                qr = min(ql + eval_bsize, n)

                latvec = self.gen_latvec(
                    sample_grid=batch['coord'][:, ql: qr, :])
                q_samples = torch.cat([latvec, batch['global_coord'][:, ql: qr, :]], dim=-1)
                q_value = self.gen_query_value(q_samples)
                
                elevs.append(q_value)
                ql = qr
            # compos output
            model_output = torch.cat(elevs, dim=1)

        pred_elev = model_output

        # from (-1,1) to (0,1)
        pred_elev = pred_elev*self.data_div + self.data_sub
        pred_elev.clamp_(0,1)
        
        return self.recfunc(
            batch=batch,
            batch_idx=batch_idx,
            pred_elev=pred_elev,
            save_dir=save_dir,
        )

class MAPIF_ENHANCE(MAPIF):
    def __init__(
        self,
        loss_method = 'naive',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.loss_method = loss_method

    def training_step(self, batch, **kwargs):
        self.data_sub = self.data_sub.to(batch['inp'].device)
        self.data_div = self.data_div.to(batch['inp'].device)

        inp= (batch['inp'] - self.data_sub) / self.data_div
        pred = self.forward(
            inp=inp,
            global_coord=batch['global_coord'],
            local_coord=batch['coord']
        )
        gt_normal = (batch['gt'][...,0].unsqueeze(-1) - self.data_sub) / self.data_div
        loss = F.l1_loss(pred['pred_elev'], gt_normal)

        return {
            'loss': loss
        }

    def validation_step(self, batch, **kwargs):
        self.data_sub = self.data_sub.to(batch['inp'].device)
        self.data_div = self.data_div.to(batch['inp'].device)

        inp= (batch['inp'] - self.data_sub) / self.data_div
        pred = self.forward(
            inp=inp,
            global_coord=batch['global_coord'],
            local_coord=batch['coord']
        )

        # recover
        pred_01 = pred['pred_elev']*self.data_div + self.data_sub
        psnr = utils.calc_psnr(pred_01, batch['gt'][...,0].unsqueeze(-1))

        return {
            'psnr': psnr
        }

    def test_step(
        self,
        batch,
        batch_idx,
        eval_bsize=None,
        save_dir=None,
        **kwargs
    ):
        self.data_sub = self.data_sub.to(batch['inp'].device)
        self.data_div = self.data_div.to(batch['inp'].device)

        inp= (batch['inp'] - self.data_sub) / self.data_div
        pred = self.forward(
            inp=inp,
            global_coord=batch['global_coord'],
            local_coord=batch['coord']
        )

        pred_elev = pred['pred_elev']
        # from (-1,1) to (0,1)
        pred_elev = pred_elev*self.data_div + self.data_sub
        pred_elev.clamp_(0,1)

        # adjust batch gt
        batch['gt']=batch['gt'][...,0].unsqueeze(-1)
        return self.recfunc(
            batch=batch,
            batch_idx=batch_idx,
            pred_elev=pred_elev,
            save_dir=save_dir,
        )


class MAPIF_SDF(MAPIF):
    def __init__(
        self,
        # encoder,
        # mapif,
        sdfnet,
        loss_method = 'naive',
        feat_detach = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.loss_method = loss_method
        self.feat_detach = feat_detach

        if not isinstance(sdfnet, nn.Module):
            sdfnet_in_dim = self.encoder.out_dim
            sdfnet_in_dim += 3 # dem 3D coords
            
            self.sdfnet = utils.object_from_dict(
                sdfnet['spec'],
                in_dim=sdfnet_in_dim
            )
        else:
            self.sdfnet = sdfnet

    def query_sdf(self, q_samples):
        bs, q = q_samples.shape[:2]
        pred = self.sdfnet(q_samples.view(bs * q, -1)).view(bs, q, -1)
        return pred

    def forward(self, inp, global_coord, local_coord):
        self.gen_feat(inp)

        if self.make_coord_local:
            latvec = self.gen_latvec(
                sample_grid=local_coord,
                mode='nearest',
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
            samples_mapif = torch.cat([latvec, rel_coord], dim=-1)
            elev = self.gen_query_value(samples_mapif)

            latvec_sdf = self.gen_latvec(
                sample_grid=local_coord,
                mode='bicubic',
                )
            if self.feat_detach:
                latvec_sdf = latvec_sdf.detach()

            samples_sdfnet = torch.cat([latvec_sdf, global_coord, elev], dim=-1)
            sdfv = self.query_sdf(samples_sdfnet)

        else:
            latvec = self.gen_latvec(sample_grid=local_coord)
            samples_mapif = torch.cat([latvec, global_coord], dim=-1)
            elev = self.gen_query_value(samples_mapif)

            if self.feat_detach:
                samples_sdfnet = torch.cat([latvec.detach(), global_coord.detach(), elev.detach()], dim=-1)
            else:
                samples_sdfnet = torch.cat([latvec, global_coord, elev], dim=-1)
            sdfv = self.query_sdf(samples_sdfnet)

        return {
            'pred_elev': elev,
            'pred_sdf': sdfv,
        }

    def training_step(self, batch, **kwargs):
        data_sub = self.data_sub.to(batch['inp'].device)
        data_div = self.data_div.to(batch['inp'].device)

        inp= (batch['inp'] - data_sub) / data_div
        train_res = self.forward(
            inp=inp,
            global_coord=batch['global_coord'],
            local_coord=batch['coord']
        )

        gt_elev = (batch['gt'] - data_sub) / data_div
        sr_loss = F.l1_loss(train_res['pred_elev'], gt_elev)

        gt_sdf = train_res['pred_elev'] - gt_elev
        sdf_loss = F.mse_loss(train_res['pred_sdf'], gt_sdf)

        compos_loss = F.l1_loss(train_res['pred_elev']-train_res['pred_sdf'], gt_elev)


        if self.loss_method == 'naive':
            loss = compos_loss
        elif self.loss_method == 'all':
            loss = compos_loss + sr_loss + sdf_loss
        elif self.loss_method == 'focus':
            sdf_co = (train_res['pred_sdf'].abs()*100).exp()
            focus_loss = ((train_res['pred_elev']-train_res['pred_sdf'] - gt_elev)*sdf_co).abs().mean()
            loss = focus_loss + sdf_loss + sr_loss
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
            global_coord=batch['global_coord'],
            local_coord=batch['coord'],
        )
        pred_elev = model_output['pred_elev']
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
                global_coord=batch['global_coord'],
                local_coord=batch['coord']
            )

        pred_elev = model_output['pred_elev']
        pred_sdf = model_output['pred_sdf']
        enhance_elev = pred_elev-pred_sdf

        # from (-1,1) to (0,1)
        pred_elev = pred_elev*data_div + data_sub
        enhance_elev = enhance_elev*data_div + data_sub
        enhance_elev.clamp_(0,1)

        return self.recfunc(
            batch=batch,
            batch_idx=batch_idx,
            pred_elev=enhance_elev,
            save_dir=save_dir,
        )


class SDFIF(nn.Module, DemRec_IF):
    def __init__(
        self,
        encoder,
        sdfnet,
        data_sub=[0.5], data_div=[0.5],
        make_coord_local=False,
        interp_mode='nearest',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.data_sub = torch.FloatTensor(data_sub)
        self.data_div = torch.FloatTensor(data_div)
        self.make_coord_local = make_coord_local
        self.interp_mode = interp_mode

        sdfnet_in_dim = self.encoder.out_dim +2
            
        self.sdfnet = utils.object_from_dict(
            sdfnet['spec'],
            in_dim=sdfnet_in_dim
        )

    def query_sdf(self, q_samples):
        bs, q = q_samples.shape[:2]
        pred = self.sdfnet(q_samples.view(bs * q, -1)).view(bs, q, -1)
        return pred

    def pred(self, inp, global_coord, local_coord):
        self.gen_feat(inp)

        # get horiz line for preding sdf
        grid_ = local_coord.clone()
        horiz_line = F.grid_sample(
            inp, grid_.flip(-1).unsqueeze(1),
            mode=self.interp_mode, align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)

        # hr_sdf = hr_value - horiz_line

        if self.make_coord_local:
            latvec = self.gen_latvec(
                sample_grid=local_coord,
                mode=self.interp_mode,
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
            q_samples = torch.cat([latvec, rel_coord], dim=-1)
        else:
            latvec = self.gen_latvec(
                sample_grid=local_coord,
                mode=self.interp_mode
            )
            q_samples = torch.cat([latvec, global_coord], dim=-1)

        sdf_value = self.query_sdf(q_samples)
        elev_value = horiz_line+sdf_value

        return {
            'pred_sdf': sdf_value,
            'pred_elev': elev_value,
            'horiz_line': horiz_line,
        }

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

        
        gt_sdf = gt_elev - preds['horiz_line']
        sdf_loss = F.mse_loss(preds['pred_sdf'], gt_sdf)

        loss = sr_loss

        return {
            'loss': loss,
            'sr_loss': sr_loss,
            'sdf_loss': sdf_loss,
        }

    def val_mod(self, normal_pred, normal_gt):
        psnr = utils.calc_psnr(normal_pred, normal_gt)

        return {
            'psnr': psnr
        }


