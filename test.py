#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import glob
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from models import Fmri2Image
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

def make_image (tensor):
    v = ((tensor.clamp(-1,1) + 1.0) * 127.5).detach().cpu().permute(1,2,0).numpy()
    v = np.clip(np.rint(v), 0, 255).astype(np.uint8)
    return v #return PIL.Image.fromarray(v)



#pipeline.to('cuda')

#encoder = FmriEncoder(DIM)
model = Fmri2Image(DIM, ENCODE_DIM)
model.encoder.load_state_dict(torch.load(glob_newest('output/fmri2image-*.bin')))
model.to(device)

DUP = 1
COLS = 5

for split in ['test', 'train']:
    test_ds = Fmri2ImageDataset('data/%s.pkl' % split, is_train=False)
    #random.seed(1999)
    random.shuffle(test_ds.samples)

    gal = Gallery('%s_out'  % split , cols= COLS)

    for i in tqdm(range(128)):
        sample = test_ds[i]
        #PIL.Image.fromarray(make_image(sample['pixel_values'])).save(gal.next())
        #target.resize((SIZE, SIZE)).save(gal.next())
        with torch.no_grad():
            images = model(sample['fmri'].reshape((1, -1)).to(device))
            #images = make_images(images)
            #images = pipeline(prompt_embeds=encode, negative_prompt=None, latents=latents, num_inference_steps=1, num_images_per_prompt=DUP).images
            #images = pipeline(prompt="", latents=latents, num_inference_steps=50, num_images_per_prompt=DUP).images
        out = np.concatenate([make_image(sample['pixel_values']), (make_image(images[0]))], axis=1)
        PIL.Image.fromarray(out).save(gal.next())
        gal.flush()

