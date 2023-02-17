import numpy as np
import imageio

def demfile_io(dem_file: str):
    file_suffix = dem_file.split('.')[-1]
    if file_suffix == 'tif':
        dem_data = imageio.imread(dem_file)
    elif file_suffix == 'dem':
        dem_data = np.loadtxt(dem_file, dtype=np.float32, delimiter=',')
    
    return dem_data

def dem2one(dem_data, epsilon=1.0e-10):
    scale = dem_data.max()-dem_data.min()+epsilon
    bias = dem_data.min()
    the_one = (dem_data-bias)/scale
    return the_one, [scale, bias]

def dem2multi(dem_data, interval=500, max_elev=3000):
    split_elev_list = list(range(interval, max_elev+1, interval))

    dem_sp = dem_data.copy()
    aggre_channels = np.zeros_like(dem_data, dtype=np.float)
    for threshold_elev in split_elev_list[::-1]:

        new_c = np.zeros_like(dem_data)
        mask = dem_sp>threshold_elev
        new_c[mask]=(dem_sp-threshold_elev)[mask]
        aggre_channels[mask] = dem2one(new_c)[0][mask]
        # cover the value
        dem_sp[mask]=0.0

    # new_channels = np.stack(aggre_channels, axis=0)


    return aggre_channels