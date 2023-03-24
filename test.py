#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import glob
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

SIZE=256

def glob_newest (pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    newest_file = max(files, key=os.path.getctime)
    print("Using", newest_file)
    return newest_file


pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

def dummy(images, **kwargs):
    return images, False

pipeline.safety_checker = dummy


pipeline.to('cuda')

test_ds = Fmri2ImageDataset('test.pkl', is_train=False)
#random.seed(1999)
random.shuffle(test_ds.samples)
encoder = FmriEncoder(DIM, ENCODE_DIM)
encoder.load_state_dict(torch.load(glob_newest('output/fmri2image-*.bin')))
encoder.to(pipeline.device)

DUP = 1
COLS = 3

gal = Gallery('test_out', cols= (2 + DUP) * COLS)
for _ in range(COLS):
    gal.text("What Subject Sees")
    for i in range(DUP):
        gal.text("Decoded Image %d" % (i+1))
    gal.text("||||||||||||")

for i in range(64):
    sample = test_ds[i]
    target = PIL.Image.fromarray(make_image(sample['pixel_values']))
    target.resize((SIZE, SIZE)).save(gal.next())
    with torch.no_grad():
        encode, latents = encoder(sample['fmri'].reshape((1, -1)).to(pipeline.device))
        encode = encode.to(dtype=torch.float16)
        print(torch.amin(latents), torch.amax(latents))
        latents = latents.to(dtype=torch.float16)
        latents *= pipeline.vae.config.scaling_factor #* 0.05
        if DUP > 1:
            latents = latents.tile((DUP, 1, 1, 1))
        #negative = -encode #torch.zeros_like(encode, dtype=torch.float16, device=pipeline.device)
        #negative = torch.zeros_like(encode, dtype=torch.float16, device=pipeline.device)
        #image = pipeline(prompt_embeds=encode, num_inference_steps=51, negative_prompt_embeds=negative).images[0]
        images = pipeline(prompt_embeds=encode, negative_prompt=None, latents=latents, num_inference_steps=1, num_images_per_prompt=DUP).images
        #images = pipeline(prompt="", latents=latents, num_inference_steps=50, num_images_per_prompt=DUP).images
    for image in images:
        image.resize((SIZE, SIZE)).save(gal.next())
    gal.text("||||||||||||")
    gal.flush()

