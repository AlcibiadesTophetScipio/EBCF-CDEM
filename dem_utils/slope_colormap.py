import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import imageio.v2 as imageio


colors = ["Green","LimeGreen","LawnGreen","GreenYellow","Yellow","Orange","DarkOrange","OrangeRed","Red"]

my_slope_cmap = ListedColormap(colors, name="my_slope_cmap")

if __name__ == '__main__':

        data = [[i for i in range(1, len(colors)+1)]]
        plt.imshow(data, cmap=my_slope_cmap, norm=None)
        plt.savefig("temp.png")