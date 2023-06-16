import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np
import math
from skimage import io as skio


class Slope(nn.Module):
    def __init__(self):
        super(Slope, self).__init__()
        weight1 = np.zeros(shape=(3, 3), dtype=np.float32)
        weight2 = np.zeros(shape=(3, 3), dtype=np.float32)

        weight1[0][0] = -1
        weight1[0][1] = 0
        weight1[0][2] = 1
        weight1[1][0] = -2
        weight1[1][1] = 0
        weight1[1][2] = 2
        weight1[2][0] = -1
        weight1[2][1] = 0
        weight1[2][2] = 1

        weight2[0][0] = -1
        weight2[0][1] = -2
        weight2[0][2] = -1
        weight2[1][0] = 0
        weight2[1][1] = 0
        weight2[1][2] = 0
        weight2[2][0] = 1
        weight2[2][1] = 2
        weight2[2][2] = 1

        weight1 = np.reshape(weight1, (1, 1, 3, 3))
        weight2 = np.reshape(weight2, (1, 1, 3, 3))
        weight1 = weight1 / (8 * 30)
        weight2 = weight2 / (8 * 30)
        self.weight1 = nn.Parameter(torch.tensor(weight1))  # 自定义的权值
        self.weight2 = nn.Parameter(torch.tensor(weight2))
        self.bias = nn.Parameter(torch.zeros(1))  # 自定义的偏置

    def forward(self, x):
        dx = F.conv2d(x, self.weight1, self.bias, stride=1, padding=1)
        dy = F.conv2d(x, self.weight2, self.bias, stride=1, padding=1)
        ij_slope = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))
        ij_slope = torch.arctan(ij_slope) * 180 / math.pi
        return ij_slope

class Aspect(nn.Module):
    def __init__(self):
        super(Aspect, self).__init__()
        weight1 = np.zeros(shape=(3, 3), dtype=np.float32)
        weight2 = np.zeros(shape=(3, 3), dtype=np.float32)

        weight1[0][0] = -1
        weight1[0][1] = 0
        weight1[0][2] = 1
        weight1[1][0] = -2
        weight1[1][1] = 0
        weight1[1][2] = 2
        weight1[2][0] = -1
        weight1[2][1] = 0
        weight1[2][2] = 1

        weight2[0][0] = -1
        weight2[0][1] = -2
        weight2[0][2] = -1
        weight2[1][0] = 0
        weight2[1][1] = 0
        weight2[1][2] = 0
        weight2[2][0] = 1
        weight2[2][1] = 2
        weight2[2][2] = 1

        weight1 = np.reshape(weight1, (1, 1, 3, 3))
        weight2 = np.reshape(weight2, (1, 1, 3, 3))
        weight1 = weight1 / (8)
        weight2 = weight2 / (8)
        self.weight1 = nn.Parameter(torch.tensor(weight1))  # 自定义的权值
        self.weight2 = nn.Parameter(torch.tensor(weight2))
        self.bias = nn.Parameter(torch.zeros(1))  # 自定义的偏置

    def forward(self, x):
        dx = F.conv2d(x, self.weight1, self.bias, stride=1, padding=1)
        dy = F.conv2d(x, self.weight2, self.bias, stride=1, padding=1)
        batchsize, channel, row, col = x.shape
        aspect = 57.29578 * torch.atan2(dy, -dx)
        # a = np.array(aspect[1][0].cpu())
        aspect = torch.where(aspect > 90, 360 - aspect + 90, 90 - aspect)
        return aspect

content_criterion_mse = nn.MSELoss().cuda()
content_criterion_mae = nn.L1Loss().cuda()
Slope_net = Slope().cuda()
Aspect_net = Aspect().cuda()

def cmp_DEMFeature(demA_file, demB_file, scale=None, file_flag=True):
    if file_flag:
        demA = skio.imread(demA_file)
        demB = skio.imread(demB_file)
    else:
        demA = demA_file
        demB = demB_file

    if scale:
        demA = demA[scale:-scale, scale:-scale]
        demB = demB[scale:-scale, scale:-scale]

    demA_tensor = torch.from_numpy(demA).cuda().unsqueeze(0).unsqueeze(0)
    demB_tensor = torch.from_numpy(demB).cuda().unsqueeze(0).unsqueeze(0)

    high_slope = Slope_net(demA_tensor)
    fake_slope = Slope_net(demB_tensor)
    high_aspect = Aspect_net(demA_tensor)
    fake_aspect = Aspect_net(demB_tensor)

    return {
        'slope_1_mae': content_criterion_mae(high_slope, fake_slope).detach().cpu().numpy(),
        'aspect_1_mae': content_criterion_mae(high_aspect, fake_aspect).detach().cpu().numpy(),
        'slope_1_rmse': math.sqrt(content_criterion_mse(high_slope, fake_slope)),
        'aspect_1_rmse': math.sqrt(content_criterion_mse(high_aspect, fake_aspect)),
    }