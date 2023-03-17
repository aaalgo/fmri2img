#!/usr/bin/env python3
import pickle
import numpy as np
import nibabel as nib
from tqdm import tqdm
from calc_stat import Stats
from config import *

if __name__ == '__main__':
    with open('stats0.pkl', 'rb') as f:
        stats = pickle.load(f)

    m1 = stats.m1
    m2 = stats.m2
    nsigma = np.sqrt(stats.count * m2 - np.square(m1))
    corr = stats.count * stats.cross

    l0, l1, l2 = m1.shape

    center1 = m1[1:-1, 1:-1, 1:-1]
    center2 = m2[1:-1, 1:-1, 1:-1]
    center_nsigma = nsigma[1:-1, 1:-1, 1:-1]
    for i, (d0, d1, d2) in enumerate(tqdm(NEIGHBORS_DELTA)):
        shift1 = m1[(1+d0):(l0-1+d0),
                    (1+d1):(l1-1+d1),
                    (1+d2):(l2-1+d2)]
        shift2 = m2[(1+d0):(l0-1+d0),
                    (1+d1):(l1-1+d1),
                    (1+d2):(l2-1+d2)]
        shift_nsigma = np.sqrt(stats.count * shift2 - np.square(shift1))
        corr[i, :, :, :] -= center1 * shift1
        corr[i, :, :, :] /= center_nsigma * shift_nsigma
    sigma = nsigma/stats.count
    print(corr.shape)

    smin = np.amin(sigma)
    smax = np.amax(sigma)
    print("SIGMA RANGE:", smin, smax)

    #cmin = np.nanmin(corr)
    #cmax = np.nanmax(corr)
    #print(cmin, cmax)

    u16 = np.clip(np.rint(65535 * (sigma - smin) / (smax - smin)), 0, 65535).astype(np.uint16)
    u16[~(sigma > SIGMA_THRESHOLD)] = 0
    img = nib.Nifti1Image(u16, np.eye(4))
    nib.save(img, 'sigma.nii.gz')

    center_OK = sigma[1:-1, 1:-1, 1:-1] > SIGMA_THRESHOLD

    a, b, c = center_OK.shape
    print(a*b*c)
    print(np.sum(center_OK))
    print(1.0 * np.sum(center_OK)/(a*b*c))

    for i, (d0, d1, d2) in enumerate(tqdm(NEIGHBORS_DELTA)):
        #shift1 = m1[(1+d0):(l0-1+d0),
        #            (1+d1):(l1-1+d1),
        #            (1+d2):(l2-1+d2)]
        #shift2 = m2[(1+d0):(l0-1+d0),
        #            (1+d1):(l1-1+d1),
        #            (1+d2):(l2-1+d2)]
        c = corr[i, :, :, :]



