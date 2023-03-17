#!/usr/bin/env python3
import pickle
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm
from config import *

# Correlation calculation
#   c(xy) = [n s(xy) - s(x) * s(y)]
#   r(xy) = c(xy) / sqrt[c(xx) * c(yy)]

# So we need:
#   - sx, sy
#   - sxx, syy, sxy

class Stats:
    # we are to calculate
    #   - information entropy of [1:-1, 1:-1, 1:-1]
    #   - correlation of each pixel with its neighbors
    def __init__ (self, shape=TIMESERIES_SHAPE):
        self.volume_shape = shape[:3]
        d0, d1, d2 = shape[:3]
        self.cropped_shape = (d0-2, d1-2, d2-2)
        self.m1 = np.zeros(self.volume_shape, dtype=np.float64)
        self.m2 = np.zeros(self.volume_shape, dtype=np.float64)
        self.cross = np.zeros((len(NEIGHBORS_DELTA),) + self.cropped_shape, dtype=np.float64)
        self.count = 0
        pass

    def update (self, volume):
        assert volume.shape == self.volume_shape
        if False:
            amin = np.amin(volume)
            amax = np.amax(volume)
            print(amin, amax)
        self.count += 1
        self.m1 += volume
        self.m2 += np.square(volume)

        l0, l1, l2 = volume.shape

        center = volume[1:-1, 1:-1, 1:-1]
        for i, (d0, d1, d2) in enumerate(NEIGHBORS_DELTA):
            shift = volume[(1+d0):(l0-1+d0),
                           (1+d1):(l1-1+d1),
                           (1+d2):(l2-1+d2)]
            self.cross[i, :, :, :] += center * shift

if __name__ == '__main__':

    input_paths = glob(CORR_INPUT_PATTERN)

    stats = Stats()
    shape = TIMESERIES_SHAPE
    #input_paths = list(input_paths)
    for input_path in tqdm(input_paths):
        ts = nib.load(input_path)
        if shape is None:
            shape = ts.shape
            print(type(shape), shape)
        else:
            assert shape == ts.shape

        ts = ts.get_fdata()
        for i in range(ts.shape[3]):
            stats.update(ts[:, :, :, i])

    with open('stats.pkl', 'wb') as f:
        pickle.dump(stats, f)


