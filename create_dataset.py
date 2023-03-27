#!/usr/bin/env python3
import sys
import os
import pickle
from glob import glob
import numpy as np
import h5py
from collections import defaultdict
from tqdm import tqdm
from config import *

if __name__ == '__main__':

    samples = defaultdict(lambda: [])
    total = []
    for session in range(1,38):
        images = []
        for run in range(1, 20):
            design_path = os.path.join(NSD_ROOT, 'nsddata_timeseries/ppdata/subj%02d/%smm' % (SUBJECT, FUNC_SPACE), 'design/design_session%02d_run%02d.tsv' % ( session, run))
            if not os.path.exists(design_path):
                continue
            design = np.loadtxt(design_path, dtype=np.int32, usecols=0)
            for v in design:
                if v != 0:
                    images.append(v-1)
        betas_path = 'data/betas/%02d/visual/betas_session%02d.nii.gz.npz' % (SUBJECT, session)
        betas = np.load(betas_path)['arr_0']
        assert len(images) == betas.shape[0]
        for i, v in enumerate(images):
            samples[v].append(betas[i, :])
            total.append(betas[i, :])
    out = []
    total = np.stack(total)
    mean = np.mean(total, axis=0)
    std = np.std(total, axis=0)
    print(total.shape, mean.shape, std.shape)
    for k, v in samples.items():
        out.append({
            'image_id': k,
            'fmri': [(x - mean) /std for x in v]
            })
    with open('data/samples%02d.pkl' % SUBJECT, 'wb') as f:
        pickle.dump(out, f)

