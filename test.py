#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import glob
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (
    DiffusionPipeline
)
import imgaug.augmenters as iaa
from models import FmriEncoder
from train import Fmri2ImageDataset, make_image
import PIL
import random
from gallery.gallery import Gallery
from config import *

#SIZE=256

def glob_newest (pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    newest_file = max(files, key=os.path.getctime)
    print("Using", newest_file)
    return newest_file

device =  torch.device('cuda:0')

#pipeline = DiffusionPipeline.from_pretrained(
#    "runwayml/stable-diffusion-v1-5",
#    torch_dtype=torch.float16
#)

def dummy(images, **kwargs):
    return images, False

#pipeline.safety_checker = dummy


#pipeline.to('cuda')

test_ds = Fmri2ImageDataset('data/test.pkl', is_train=False)
#random.seed(1999)
random.shuffle(test_ds.samples)
encoder = FmriEncoder(DIM)
encoder.load_state_dict(torch.load(glob_newest('output/fmri2image-70.bin')))
encoder.to(device)

DUP = 1
COLS = 5

gal = Gallery('test_out', cols= (1 + DUP) * COLS)
for _ in range(COLS):
    gal.text("What Subject Sees")
    for i in range(DUP):
        gal.text("Decoded Image %d" % (i+1))

def make_image (tensor):
    v = ((tensor.clamp(-1,1) + 1.0) * 127.5).detach().cpu().permute(1,2,0).numpy()
    v = np.clip(np.rint(v), 0, 255).astype(np.uint8)
    return v #return PIL.Image.fromarray(v)

for i in tqdm(range(64)):
    sample = test_ds[i]
    PIL.Image.fromarray(make_image(sample['pixel_values'])).save(gal.next())
    #target.resize((SIZE, SIZE)).save(gal.next())
    with torch.no_grad():
        images = encoder(sample['fmri'].reshape((1, -1)).to(device))
        #images = make_images(images)
        #images = pipeline(prompt_embeds=encode, negative_prompt=None, latents=latents, num_inference_steps=1, num_images_per_prompt=DUP).images
        #images = pipeline(prompt="", latents=latents, num_inference_steps=50, num_images_per_prompt=DUP).images
    PIL.Image.fromarray(make_image(images[0])).save(gal.next())
    gal.flush()

