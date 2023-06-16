import matplotlib.pyplot as plt
import numpy as np
import imageio
import cv2

def normalize_1(data_array, eps=1.0e-8):
    data_max = np.max(data_array)
    data_min = np.min(data_array)
    norm_data = (data_array-data_min)/(data_max-data_min+eps)
    return norm_data

if __name__ == '__main__':
    dem_file = './demo_data/demo_dem.tif'
    dem_array = imageio.v2.imread(dem_file)

    sr_scale = 8
    H, W = dem_array.shape
    lr_dem = cv2.resize(dem_array, (H // sr_scale, W // sr_scale),
                    interpolation=cv2.INTER_NEAREST)
    hr_dem_nearest = cv2.resize(lr_dem, (H, W),
                    interpolation=cv2.INTER_NEAREST)

    select_x = int(H//2)
    bias_array = dem_array-hr_dem_nearest

    origin_line = dem_array[select_x]
    lr_line = hr_dem_nearest[select_x]
    bias_line = bias_array[select_x]
    coord_x = [i for i in range(len(origin_line))]

    plt.subplot(2,2,1)
    plt.imshow(dem_array, cmap='terrain')
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.plot(coord_x, normalize_1(origin_line))
    plt.subplot(2,2,3)
    plt.plot(coord_x, normalize_1(lr_line))
    plt.subplot(2,2,4)
    plt.plot(coord_x, normalize_1(bias_line))

    plt.savefig('./results/bias.png', dpi=500)
