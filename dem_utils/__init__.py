from .dem_attribute import cmp_TerrainAttribute
from .dem_features import cmp_DEMFeature, Slope_net, Aspect_net
from .dem_parameters import aspect,slope, cmp_DEMParameter

import numpy as np
import torch
from skimage import io as skio
import rasterio
import richdem as rd

def cmp_dem_extractor(dem_file):
    dem=skio.imread(dem_file)
    dem_tensor = torch.from_numpy(dem).cuda().unsqueeze(0).unsqueeze(0)
    feature_aspect = Aspect_net(dem_tensor).squeeze()
    feature_slope = Slope_net(dem_tensor).squeeze()
    

    demA = rasterio.open(dem_file).read(1)
    dem_rd_A = rd.rdarray(demA, no_data=-9999)

    dem_rd_A.geotransform = [0,30,0,0,0,30] # 1,5 specify the size
    rd_aspect = rd.TerrainAttribute(dem_rd_A, 'aspect')
    rd_slope = rd.TerrainAttribute(dem_rd_A, 'slope_degrees')

    params_aspect = aspect(dem, 30) # size are not affect
    params_slope = slope(dem, 30) # size must be specified

    return