import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import imageio.v2 as imageio


colors = ["red","orange","yellow","lime","cyan","DeepSkyBlue","blue","MediumOrchid","red"]
nodes = [
        0.0, # red, north
        22.5, # orange, northeast
        67.5, # yellow, east
        112.5, # lime, southeast
        157.5, # cyan, south
        202.5, # deepskyblue, southwest
        247.5, # blue, west
        292.5, # mediumorchid, northwest
        337.5, # red, north
        360.0,
        ]
# define the bins and normalize
# bounds = [i/360 for i in nodes]
my_aspect_bounds = nodes
my_aspect_norm = mpl.colors.BoundaryNorm(my_aspect_bounds, len(colors))

# nodes = [i/360 for i in range(len(nodes)-1)]
# my_cmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))
# my_cmap = LinearSegmentedColormap.from_list("mycmap", colors, len(colors))

my_aspect_cmap = ListedColormap(colors, name="my_aspect_cmap")
# mpl.colormaps.register(cmap=my_cmap)

# data = [[i/(len(colors)+1) for i in range(1, len(colors)+1) ]]
# plt.imshow(data, cmap=my_cmap, norm=norm)

if __name__ == '__main__':
    dem_data = imageio.imread("./temp_dem-features/2080-aspect.tif")
    fig = plt.figure()
    im = plt.imshow(dem_data, cmap=my_aspect_cmap, norm=my_aspect_norm)
    fig.colorbar(im, ticks=my_aspect_bounds)
    plt.axis('off')

    plt.savefig("temp.png")