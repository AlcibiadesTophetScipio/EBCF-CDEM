#%%
import json
import glob
import os
import numpy as np
import random
from pathlib import Path

#%% Generate split file for tfasr dataset.
scan_dir = '/data/syao/Datasets/dataset_TfaSR/(60mor120m)to30m/DEM_Test/'
filenames_test = glob.glob(scan_dir+'/*/*.TIF')
scan_dir = '/data/syao/Datasets/dataset_TfaSR/(60mor120m)to30m/DEM_Train/'
filenames_train = glob.glob(scan_dir+'/*/*.TIF')

with open('TFASR.json', 'w') as f:
    json.dump(
        {
            "train": filenames_train,
            "test": filenames_test,
        },
        f
    )


# %% Generate split file for tfasr_30to10 dataset. 
scan_dir = '/data/syao/Datasets/dataset_TfaSR/30mto10m/train/train_10m/'
filenames_train_HR = glob.glob(scan_dir+'/*/*.TIF')
scan_dir = '/data/syao/Datasets/dataset_TfaSR/30mto10m/train/train_30m/'
filenames_train_LR = glob.glob(scan_dir+'/*/*.TIF')

scan_dir = '/data/syao/Datasets/dataset_TfaSR/30mto10m/test/test_10m/'
filenames_test_HR = glob.glob(scan_dir+'/*/*.TIF')
scan_dir = '/data/syao/Datasets/dataset_TfaSR/30mto10m/test/test_30m/'
filenames_test_LR = glob.glob(scan_dir+'/*/*.TIF')

with open('TFASR_HR.json', 'w') as f:
    json.dump(
        {
            "train": sorted(filenames_train_HR),
            "test": sorted(filenames_test_HR),
        },
        f
    )

with open('TFASR_LR.json', 'w') as f:
    json.dump(
        {
            "train": sorted(filenames_train_LR),
            "test": sorted(filenames_test_LR),
        },
        f
    )

# %% Generate split file for Pyrenees and Tyrol datasets. 
def generate_json(dataset_json, regexp, just_test=False):
    filenames = glob.glob(regexp)
    if len(filenames)==0:
        raise Exception(f"Wrong regular expression {regexp}.")
    random.shuffle(filenames)

    train_pos = np.floor(len(filenames)*.9).astype(int)
    with open(dataset_json, 'w') as f:
        if just_test:
            json.dump(
                {
                    "test": filenames,
                },
                f)
        else:
            json.dump(
                {
                    "train": filenames[:train_pos],
                    "test": filenames[train_pos:],
                },
                f)

    return True

# dataset_json = './pyrenees_r96.json'
# dirRawDataset = '/data/syao/Datasets/Terrains/Pyrenees_2m_split/r96'
dataset_json = './tyrol_r96.json'
dirRawDataset = '/data/syao/Datasets/Terrains/Tyrol_2m_split/r96'
regexp = dirRawDataset+'/*.dem'
generate_json(dataset_json, regexp, True)
