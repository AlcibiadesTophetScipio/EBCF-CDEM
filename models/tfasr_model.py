# Code from: https://doi.org/10.6084/m9.figshare.19597201

# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dPack as DCN

from .meta_cls import DemRec_IF
from datasets.utils import resize_fn
from dem_utils.dem_features import Slope_net
import utils


def swish(x):
    return F.relu(x)

content_criterion = nn.MSELoss().cuda()

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.Dropout(p=0.5),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.Dropout(p=0.5),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
# weights from https://doi.org/10.6084/m9.figshare.19597201
extract_river_Weights = '/home/syao/Codes/ebcf-cdem/paper_pth/river_unet.pth'
unet = UNet(n_channels=1, n_classes=1)  # unet
unet.load_state_dict(torch.load(extract_river_Weights, map_location='cpu'))
unet.cuda()

def GetIOU(Pred, GT, NumClasses, ClassNames=[], DisplyResults=False):
    # Given A ground true and predicted labels per pixel return the intersection over union for each class
    # and the union for each class
    ClassIOU = np.zeros(NumClasses)  # Vector that Contain IOU per class
    ClassWeight = np.zeros(
        NumClasses)  # Vector that Contain Number of pixel per class Predicted U Ground true (Union for this class)
    for i in range(NumClasses):  # Go over all classes
        Intersection = np.float32(np.sum((Pred == GT) * (GT == i)))  # Calculate class intersection
        Union = np.sum(GT == i) + np.sum(Pred == i) - Intersection  # Calculate class Union
        if Union > 0:
            ClassIOU[i] = Intersection / Union  # Calculate intesection over union
            ClassWeight[i] = Union

    # ------------Display results (optional)-------------------------------------------------------------------------------------
    if DisplyResults:
        for i in range(len(ClassNames)):
            print(ClassNames[i] + ") " + str(ClassIOU[i]))
        print("Mean Classes IOU) " + str(np.mean(ClassIOU)))
        print("Image Predicition Accuracy)" + str(np.float32(np.sum(Pred == GT)) / GT.size))
    # -------------------------------------------------------------------------------------------------

    return ClassIOU, ClassWeight

class RiverLoss(nn.Module):
    def __init__(self):
        super(RiverLoss, self).__init__()

    def forward(self, input, targets):
        batchSize, channel, H, W = input.shape

        fake_river_heatmap = unet(input)
        criterion = nn.BCEWithLogitsLoss().cuda()
        loss = criterion(fake_river_heatmap, targets)

        fake_river = 1.0 * (F.sigmoid(fake_river_heatmap).detach().cpu() > 0.5).numpy()
        miou_0 = 0
        miou_1 = 0
        Miou = 0
        for j in range(batchSize):
            real_j = np.reshape(targets[j].detach().cpu(), (H, W)).numpy()
            fake_j = np.reshape(fake_river[j].data, (H, W))
            iou = GetIOU(real_j, fake_j, 2)
            miou_0 = miou_0 + iou[0][0]
            miou_1 = miou_1 + iou[0][1]
            Miou = Miou + np.mean(iou[0])

        miou_0 = miou_0 / batchSize
        miou_1 = miou_1 / batchSize
        Miou = Miou / batchSize
        ####################################################################################s
        return loss, miou_0, miou_1, Miou

river_conterion = RiverLoss()
river_conterion.cuda()

class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        y = swish(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x


class Generator(nn.Module, DemRec_IF):
    def __init__(
            self, 
            n_residual_blocks, 
            upsample_factor,
            data_sub=[0.5], data_div=[0.5],
            paper_test=False,
            ):
        super(Generator, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor
        self.data_sub = torch.FloatTensor(data_sub)
        self.data_div = torch.FloatTensor(data_div)
        self.paper_test = paper_test

        self.conv1 = nn.Conv2d(1, 64, 9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.dconv2_2 = DCN(64, 64, 3, 1, 1)
        self.dconv2_3 = DCN(64, 64, 3, 1, 1)
        self.dconv2_4 = DCN(64, 1, 3, 1, 1)
        self.conv4 = nn.Conv2d(1, 1, 1, stride=1, padding=0)

    def pred(self, x):
        #########################original version########################
        x = self.conv1(x)

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)

        x = self.bn2(self.conv2(y)) + x

        x = swish(self.dconv2_2(x))
        x = swish(self.dconv2_3(x))
        return {'pred_elev': self.conv4(self.dconv2_4(x))}
    
    def forward(self, batch, batch_idx, flag, epoch=0, **kwargs):
        data_sub = self.data_sub.to(batch['inp'].device)
        data_div = self.data_div.to(batch['inp'].device)
        inp= (batch['inp'] - data_sub) / data_div

        if flag == 'train':
            inp_resize = resize_fn(inp, [i *self.upsample_factor for i in inp.shape[-2:]])
            preds = self.pred(inp_resize)
            gt_elev = (batch['gt'] - data_sub) / data_div
            gt_elev.reshape(preds['pred_elev'].shape)
            return self.train_mod(preds=preds['pred_elev'], gt_elev=gt_elev, epoch=epoch)
        elif flag == 'val':
            inp_resize = resize_fn(inp, [i *self.upsample_factor for i in inp.shape[-2:]])
            preds = self.pred(inp_resize)
            normal_pred = preds['pred_elev']*data_div + data_sub
            normal_gt = batch['gt']
            return self.val_mod(normal_pred, normal_gt)
        elif flag == 'test':
            if self.paper_test:
                dem_scale, dem_bias = batch['add_args'][...,0], batch['add_args'][...,1]
                fake_inp = batch['inp']*dem_scale + dem_bias
                base_max = fake_inp.max(-1)[0].max(-1)[0].squeeze()
                base_min = fake_inp.min(-1)[0].min(-1)[0].squeeze()
                base_scale = base_max - base_min + 10

                fake_inp2 = (fake_inp-base_min)/base_scale
                batch['add_args'][...,0] = base_scale
                batch['add_args'][...,1] = base_min
                fake_inp3 = (fake_inp2 - data_sub) / data_div
                inp_resize = resize_fn(fake_inp3, [i *self.upsample_factor for i in fake_inp2.shape[-2:]])
                preds = self.pred(inp_resize)
            else:
                ih, iw = batch['inp'].shape[-2:]
                s = math.sqrt(batch['gt'].shape[1] / (ih * iw))
                inp_resize = resize_fn(inp, [round(i * s) for i in inp.shape[-2:]])
                preds = self.pred(inp_resize)

            normal_pred = preds['pred_elev']*data_div + data_sub
            # normal_pred.clamp_(0,1)

            return self.recfunc(
                batch=batch,
                batch_idx=batch_idx,
                pred_elev=normal_pred,
                save_dir=kwargs['save_dir'],
                use_original_gt=self.paper_test,
            )
            
        else:
            raise Exception('Wrong flag in model.')
        
    def train_mod(self, preds, gt_elev, epoch):
        high_res_real = gt_elev.reshape(preds.shape)
        high_res_fake = preds
        high_slope = Slope_net(high_res_real)
        fake_slope = Slope_net(high_res_fake)
        generator_slope_loss = content_criterion(high_slope, fake_slope)
        generator_content_loss = content_criterion(high_res_fake, high_res_real)

        high_river = 1.0 * (F.sigmoid(unet(high_res_real)).detach().cpu() > 0.5).numpy().astype(np.float32)
        high_river_heatmap = torch.tensor(high_river).to(gt_elev.device)
        generator_river_loss, miou_0, miou_1, Miou = river_conterion(
            high_res_fake,
            high_river_heatmap
        )
        slope_weight = 1.0
        if epoch >= 80:
            river_weight = 1e-3
        else:
            river_weight = 0.0

        generator_total_loss = generator_content_loss + slope_weight * generator_slope_loss + river_weight * generator_river_loss

        return {
                'loss': generator_total_loss,
            }

    def val_mod(self, normal_pred, normal_gt):
        psnr = utils.calc_psnr(normal_pred, normal_gt)

        return {
            'psnr': psnr
        }
