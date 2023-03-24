#!/usr/bin/env python3
import sys
import numpy as np
import nibabel as nib
sys.path.insert(0, 'nsdcode')
from nsdcode.nsd_mapdata import NSDmapdata

# process 0pt8
ref = nib.load('ppdata/subj01/anat/T2_0pt8_masked.nii.gz')
aparc = nib.load('aparc.nii.gz')
arr = np.asanyarray(aparc.dataobj)
arr = np.moveaxis(arr, 1, 2)
arr = np.flip(arr, axis=0)
arr = np.flip(arr, axis=2)
arr2 = np.zeros_like(arr)
arr2[1:, :, 1:] = arr[:-1, :, :-1]
aparc2 = nib.Nifti1Image(arr2, affine=ref.affine, header=ref.header)
src = 'aparc2.nii.gz'
nib.save(aparc2, 'aparc2.nii.gz')

mapper = NSDmapdata('.')
output = mapper.fit(1, 'anat0pt8', 'func1pt8', src, 'nearest', 0, None, 'single', None)
ref = nib.load('ppdata/subj01/func1pt8mm/mean_session23.nii.gz')
mask = nib.Nifti1Image(np.copy(output, order='C'), affine=ref.affine, header=ref.header)
nib.save(mask, 'aparc3.nii.gz')

