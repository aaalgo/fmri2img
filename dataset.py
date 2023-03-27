import os
import h5py
import pickle
import imgaug.augmenters as iaa
import torch
from torch.utils.data import Dataset
from config import *

class Fmri2ImageDataset (Dataset):
    def __init__(self, path, image_size, duplicate=1, is_train=True):
        stimuli_path=os.path.join(NSD_ROOT, 'nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5')
        self.imgBrick = h5py.File(stimuli_path, 'r')['imgBrick']
        _, H, W, _ = self.imgBrick.shape
        assert H == W
        self.samples = []
        #self.imageCache = {}
        with open(path, 'rb') as f:
            for one in pickle.load(f):
                k = one['image_id']
                for v in one['fmri']:
                    self.samples.append((k, v))
        self.is_train = is_train
        if is_train:
            self.aug = iaa.Sequential([
                iaa.GaussianBlur(sigma=(0.0, 3.0)),
                iaa.Resize({"height": image_size, "width": image_size}),
                iaa.Affine(scale=(1.0, 1.2)),
                iaa.CropToFixedSize(image_size, image_size)
                ])
        else:
            self.aug = iaa.Sequential([
                iaa.Resize({"height": image_size, "width": image_size})
                ])
        self.duplicate = duplicate

    def __len__(self):
        return len(self.samples) * self.duplicate

    def __getitem__(self, i):
        k, v = self.samples[i % len(self.samples)]
        image = self.aug(image=self.imgBrick[k, :, :, :])
        image = image / 127.5 - 1.0
        v = torch.from_numpy(v).float()
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        assert v.shape == (DIM,)
        assert not torch.any(torch.isnan(v))
        assert not torch.any(torch.isnan(image))
        return {
                'fmri': v,
                'pixel_values': image
                }

