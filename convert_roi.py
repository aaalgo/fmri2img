#!/usr/bin/env python3
import sys
import numpy as np
import nibabel as nib
sys.path.insert(0, 'nsdcode')
from nsdcode.nsd_mapdata import NSDmapdata
from config import *


if __name__ == '__main__':
    #subjroi2func(1, 'prf-visualrois.nii.gz', 'subj01_visual.nii.gz', 'anat0pt8', 'func1pt8')
    roi_path = '%s/nsddata/ppdata/subj%02d/anat/roi/%s' % (NSD_ROOT, SUBJECT, VISUAL_ROI_INPUT)
    mapper = NSDmapdata('data/raw')
    mapper.fit(SUBJECT, ANAT_SPACE, FUNC_SPACE, roi_path, 'nearest', 0, VISUAL_ROI_PATH, 'single', None)

