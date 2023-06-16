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
        local_ensemble=True, feat_unfold=True, cell_decode=True,
        data_sub=[0.5], data_div=[0.5],
        interp_mode=None,
    ):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.data_sub = torch.FloatTensor(data_sub)
        self.data_div = torch.FloatTensor(data_div)
        self.interp_mode = interp_mode

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


    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_elev(self, origin_inp, coord, cell):
        feat = self.feat

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

        # get horiz line for preding bias
        if self.interp_mode is not None:
            grid_ = coord.clone()
            horiz_line = F.grid_sample(
                origin_inp, grid_.flip(-1).unsqueeze(1),
                mode=self.interp_mode, align_corners=False
            )[:, :, 0, :].permute(0, 2, 1)

            elev_value = horiz_line+ret
            return {
                'pred_sdf': ret,
                'pred_elev': elev_value,
                'horiz_line': horiz_line,
            }
        else:
            return {'pred_elev': ret}

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

    # def test_step(
    #     self,
    #     batch,
    #     batch_idx,
    #     eval_bsize=None,
    #     save_dir=None,
    #     **kwargs
    # ):
    #     data_sub = self.data_sub.to(batch['inp'].device)
    #     data_div = self.data_div.to(batch['inp'].device)

    #     inp= (batch['inp'] - data_sub) / data_div

    #     if hasattr(self.encoder,'args') and self.encoder.args.n_colors == 3:
    #         scale = batch['add_args'][..., -2]
    #         scale = scale.view(-1,1,1,1).expand_as(inp)
    #         bias = batch['add_args'][..., -1]
    #         bias = bias.view(-1,1,1,1).expand_as(inp)
    #         inp = torch.cat([inp, 1.0/scale, 1.0/bias], dim=1)

    #     model_output = self.forward(
    #         inp=inp,
    #         liif_coord=batch['coord'],
    #         cell=batch['cell'],
    #         sdf_coord=batch['global_coord']
    #     )

    #     pred_elev = model_output['pred_elev']
    #     pred_elev = pred_elev*data_div + data_sub
    #     return self.recfunc(
    #         batch=batch,
    #         batch_idx=batch_idx,
    #         pred_elev=pred_elev.clamp(0, 1),
    #         save_dir=save_dir,
    #     )

