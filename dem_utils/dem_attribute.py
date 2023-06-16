import rasterio
import richdem as rd
import numpy as np

def cmp_TerrainAttribute(
    demA_file, demB_file, cell_size=1, scale=None,
    methods=['aspect', 'slope_degrees']
):
    demA = rasterio.open(demA_file).read(1)
    demB = rasterio.open(demB_file).read(1)

    if scale:
        demA = demA[scale:-scale, scale:-scale]
        demB = demB[scale:-scale, scale:-scale]

    dem_rd_A = rd.rdarray(demA, no_data=-9999)
    dem_rd_B = rd.rdarray(demB, no_data=-9999)
    
    # Top left cell's top let corner at <0,0>; cells are 1x1.
    # [1,5] specify the size
    dem_rd_A.geotransform = [0,cell_size,0,0,0,cell_size]
    dem_rd_B.geotransform = [0,cell_size,0,0,0,cell_size]

    results = {}
    for m in methods:
        A_value = rd.TerrainAttribute(dem_rd_A, m)
        B_value = rd.TerrainAttribute(dem_rd_B, m)
        results[m] = np.abs(A_value-B_value).mean()

    return results

