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
    #assert np.sum(mask) == VISUAL_DIM
    return mask

def extract (ts, mask):
    v = []
    for i in range(ts.shape[3]):
        v.append(ts[:, :, :, i][mask].flatten())
    v = np.stack(v)
    return v

if __name__ == '__main__':
    lower_dir = 'betas/lower'
    higher_dir = 'betas/higher'
    both_dir = 'betas/both'
    os.makedirs(lower_dir, exist_ok=True)
    os.makedirs(higher_dir, exist_ok=True)
    os.makedirs(both_dir, exist_ok=True)

    lower_mask = load_mask(regions=EARLY_VISUAL_CORTEX).astype(bool)
    higher_mask = load_mask(regions=HIGHER_VISUAL_CORTEX).astype(bool)
    both_mask = np.logical_or(lower_mask, higher_mask)
    print("Lower:", np.sum(lower_mask))
    print("Higher:", np.sum(higher_mask))
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
        lower = extract(ts, lower_mask)
        higher = extract(ts, higher_mask)
        both = extract(ts, both_mask)
        np.savez(os.path.join(lower_dir, os.path.basename(input_path)), lower)
        np.savez(os.path.join(higher_dir, os.path.basename(input_path)), higher)
        np.savez(os.path.join(both_dir, os.path.basename(input_path)), both)
    #with open('features.pkl', 'wb') as f:
    #    pickle.dump(outputs, f)

