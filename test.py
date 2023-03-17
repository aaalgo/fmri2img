#!/usr/bin/env python3
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

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

def dummy(images, **kwargs):
    return images, False

pipeline.safety_checker = dummy


pipeline.to('cuda')

test_ds = Fmri2ImageDataset('test.pkl', is_train=False)
random.seed(1999)
random.shuffle(test_ds.samples)
encoder = FmriEncoder(DIM, ENCODE_DIM)
encoder.load_state_dict(torch.load('output/fmri2image-0.bin'))
encoder.to(pipeline.device)

#self.aug = iaa.Sequential([
#    iaa.Resize({"height": 256, "width": 256})
#    ])

gal = Gallery('test_out', cols=4)
for i in range(4):
    sample = test_ds[i]
    target = PIL.Image.fromarray(make_image(sample['pixel_values']))
    with torch.no_grad():
        encode = encoder(sample['fmri'].reshape((1, -1)).to(pipeline.device)).to(dtype=torch.float16)
        encode = F.relu(encode)
        image = pipeline(prompt_embeds=encode, num_inference_steps=20).images[0]
    target.save(gal.next())
    image.save(gal.next())
gal.flush()

