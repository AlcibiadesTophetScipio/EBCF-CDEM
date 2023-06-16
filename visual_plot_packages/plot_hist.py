
from tqdm import tqdm
import math
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

from datasets.dem2be import demfile_io

if __name__ == '__main__':
    '''
    python plot_hist.py -d pyrenees -t test
    python plot_hist.py -d pyrenees -t train
    python plot_hist.py -d tyrol -t test

    python plot_hist.py -d pyrenees96 -t test
    python plot_hist.py -d pyrenees96 -t train
    python plot_hist.py -d tyrol96 -t test
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default="pyrenees", help='dataset name.')
    parser.add_argument('-t', type=str, default="test", help='type of dataset.')
    opt = parser.parse_args()
    dataset = opt.d
    split_key = opt.t

    try:
        import pycpt
        topocmap = pycpt.load.cmap_from_cptcity_url('wkp/schwarzwald/wiki-schwarzwald-cont.cpt')
    except:
        topocmap = 'Spectral_r'
        topocmap = plt.get_cmap(topocmap)

    # dataset = "pyrenees"
    # split_key = "test"

    if dataset == "pyrenees":
        split_file = "/data/syao/Datasets/Terrains/Pyrenees_2m_split/pyrenees_r256.json"
    elif dataset == "tyrol":
        split_file = "/data/syao/Datasets/Terrains/Tyrol_2m_split/tyrol_r128.json"
    elif dataset == "pyrenees96":
        split_file = "/data/syao/Datasets/Terrains/Pyrenees_2m_split/pyrenees_r96.json"
    elif dataset == "tyrol96":
        split_file = "/data/syao/Datasets/Terrains/Tyrol_2m_split/tyrol_r96.json"
    else:
        raise Exception("Wrong dataset name.")
    

    with open(split_file, 'r') as f:
        filenames = json.load(f)[split_key]

    dem_data_list = []
    for filename in tqdm(filenames):
        dem_data = demfile_io(filename)
        dem_data_list.append(dem_data)

    dem_array = np.stack(dem_data_list)

    vmin, vmax = dem_array.min(), dem_array.max()
    print(vmin, vmax)
    # ax = sns.histplot(dem_array.ravel(), kde=True, stat="density", kde_kws=dict(cut=3), bins=50)
    ax = sns.histplot(dem_array.ravel(), kde=True, stat="count", kde_kws=dict(cut=3), bins=50)
    ax.set_xlim(1000,4000)
    ax.set_xticks([x for x in range(1000, 4001, int(3000/10)) ])

    y_num = 4
    y_min, y_max = ax.get_ylim()
    print(y_min, y_max)
    y_interval = int(y_max/y_num)
    print(y_interval)
    y_scale = 1000 #if y_interval<10000 else 10000
    y_interval = int(round(y_interval / y_scale)*y_scale)
    current_values = [x for x in range(0, int(y_num*y_interval)+1, y_interval) ]
    print(current_values)
    ax.set_yticks(current_values)
    current_values = ax.get_yticks()
    ax.set_yticklabels(['{:,.0f}'.format(x) for x in current_values])

    ax.set_aspect(4000/y_max/6)
    ax.set(xlabel='Elevation (m)', ylabel='Cell count')
    _ = [patch.set_color(topocmap(plt.Normalize(vmin=vmin, vmax=vmax)(patch.xy[0]))) for patch in ax.patches]
    _ = [patch.set_alpha(1) for patch in ax.patches]
    ax.get_figure().savefig('./results/hist_{}_{}-{}.png'.format(dataset, split_key, len(filenames)), dpi=500)

    print("Finish!")