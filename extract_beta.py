#!/usr/bin/env python3
import os
import sys
import pickle
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm
from calc_stat import Stats
from config import *


def load_mask (path=ROI_PATH): #aparc_path=APARC_PATH, regions=EARLY_VISUAL_CORTEX):
    mask = np.asanyarray(nib.load(path).dataobj).astype(np.uint16)
    v, c = np.unique(mask, return_counts=True)
    for a, b in zip(v, c):
        print(a, '->', b)
    return np.logical_and(mask > 0, mask < 65535)

def extract (ts, mask):
    v = []
    for i in range(ts.shape[3]):
        v.append(ts[:, :, :, i][mask].flatten())
    v = np.stack(v)
    return v

if __name__ == '__main__':
    visual_dir = 'data/betas/visual'
    os.makedirs(visual_dir, exist_ok=True)

    visual_mask = load_mask().astype(bool)
    print("DIM:", np.sum(visual_mask))
    print("Extracting betas features..")

    input_paths = glob(BETA_INPUT_PATTERN)
    print(len(input_paths))
    for input_path in tqdm(input_paths):
        #print(input_path)
        ts = nib.load(input_path)
        #ts = ts.get_data()
        ts = np.asanyarray(ts.dataobj)
        #print(ts.dtype)
        #print(v.shape)
        visual = extract(ts, visual_mask)
        np.savez(os.path.join(visual_dir, os.path.basename(input_path)), visual)
    #with open('features.pkl', 'wb') as f:
    #    pickle.dump(outputs, f)

