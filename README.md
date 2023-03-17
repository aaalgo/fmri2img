FMRI2IMG
==========

# 1. Intro

This is an effort to reproduce the results of Takagi and Nishimoto
(https://sites.google.com/view/stablediffusion-with-brain/).

Currently the implementation is very preliminary.


# 2. Method

## Overview

The current implementation is heavily simplified:

- The brain voxels with high standard deviation through multiple fMRI
  scanns are identified.  Each such voxel contributes one dimension to
  the fMRI feature.
- The fMRI feature is `encoded` into hidden state via a single linear
  layer.
- The hidden state is fed into the Stable Diffusion pipeline in place of
  the prompt embeddings.
- The training code is based on the huggingface textual inversion script
  (in Links session).

## Workflow

This is in a very early stage.  Don't expect this to work directly.
You'll need a GPU with 40+GB memory.
The dataset is very big; a single subjects' low-resolution
data is over 50GB.

```

# 0. Update configuration
# edit local-config.py to override  default options in config.py.

# 1. Scan the fMRI data and calculate statistics.
./calc_stat.py

# 2. Extract feature.
./extract_voxels

# 3. Split to training and test set.
./split.py

# 4. Train
./train.py      # or accelerate launch ./train.py

# 5. Test
./test.py

```

# 3. Links

Dataset:

    - Paper: https://www.biorxiv.org/content/10.1101/2021.02.22.432340v1.full.pdf
    - Download: https://cvnlab.slite.page/p/CT9Fwl4_hc/NSD-Data-Manual
    - For now we only need the `nsddata_timeseries/ppdata/*/func1pt8mm/timeseries` of one subject.  Subject 1 or 2 have the best correct scores; use one of these.


Textual Inversion:

    - https://huggingface.co/docs/diffusers/training/text_inversion
    - Script: https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion

