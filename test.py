#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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
encoder.load_state_dict(torch.load('output/fmri2image-4.bin'))
encoder.to(pipeline.device)

gal = Gallery('test_out', cols=9)
for i in range(32):
    sample = test_ds[i]
    target = PIL.Image.fromarray(make_image(sample['pixel_values']))
    with torch.no_grad():
        encode = encoder(sample['fmri'].reshape((1, -1)).to(pipeline.device)).to(dtype=torch.float16)[:, :77, :]
        #negative = -encode #torch.zeros_like(encode, dtype=torch.float16, device=pipeline.device)
        #negative = torch.zeros_like(encode, dtype=torch.float16, device=pipeline.device)
        #image = pipeline(prompt_embeds=encode, num_inference_steps=50, negative_prompt_embeds=negative).images[0]
        image = pipeline(prompt_embeds=encode, negative_prompt=None, num_inference_steps=50).images[0]
    target.resize((SIZE, SIZE)).save(gal.next())
    image.resize((SIZE, SIZE)).save(gal.next())
    gal.text("||||||||||||")
    gal.flush()

