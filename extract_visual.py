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


def load_mask (aparc_path=APARC_PATH, regions=EARLY_VISUAL_CORTEX):
    aparc = np.asanyarray(nib.load(aparc_path).dataobj).astype(np.uint16)
    mask = np.zeros_like(aparc)
    for v in regions:
        mask += (aparc == v)
    assert (mask >= 0).all() and (mask <= 1).all()
    assert np.sum(mask) == VISUAL_DIM
    return mask

if __name__ == '__main__':
    out_dir = 'visual'
    os.makedirs(out_dir, exist_ok=True)

    mask = load_mask().astype(bool)
    print("Extracting %d-D visual features.." % VISUAL_DIM)

    input_paths = glob(CORR_INPUT_PATTERN)
    print(len(input_paths))
    for input_path in tqdm(input_paths):
        #print(input_path)
        ts = nib.load(input_path)
        #ts = ts.get_data()
        ts = np.asanyarray(ts.dataobj)
        #print(ts.dtype)
        v = []
        for i in range(ts.shape[3]):
            v.append(ts[:, :, :, i][mask].flatten())
        v = np.stack(v)
        #print(v.shape)
        np.savez(os.path.join(out_dir, os.path.basename(input_path)), v)
    #with open('features.pkl', 'wb') as f:
    #    pickle.dump(outputs, f)

