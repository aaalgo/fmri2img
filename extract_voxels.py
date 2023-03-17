#!/usr/bin/env python3
import os
import pickle
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm
from calc_stat import Stats
from config import *

if __name__ == '__main__':
    out_dir = 'features'
    os.makedirs(out_dir, exist_ok=True)
    with open('stats.pkl', 'rb') as f:
        stats = pickle.load(f)

    m1 = stats.m1
    m2 = stats.m2
    sigma = np.sqrt(stats.count * m2 - np.square(m1)) / stats.count
    mask = sigma > SIGMA_THRESHOLD
    D = np.sum(mask)
    print("Extracting %d dimensions..." % D)

    input_paths = glob(CORR_INPUT_PATTERN)
    print(len(input_paths))
    #sys.exit(0)

    #input_paths = list(input_paths)[:3]
    #outputs = {}
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

