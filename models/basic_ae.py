#!/usr/bin/env python3
import torch.nn as nn
from diffusers import AutoencoderKL
from config import *

class FmriEncoder (nn.Module):
    def __init__ (self, Din):
        super().__init__()
        self.fc1 = nn.Linear(Din, 512)
        self.fc2 = nn.Linear(512, 256)
        self.deconv1 = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, 4, kernel_size=4, stride=2, padding=1)

    def forward (self, X):
        x = self.fc1(X)
        x = nn.functional.gelu(x)
        x = self.fc2(x)
        x = nn.functional.gelu(x)
        x = x.view(-1, 16, 4, 4)
        x = self.deconv1(x)
        x = nn.functional.gelu(x)
        x = self.deconv2(x)
        return x

class Fmri2Image (nn.Module):
    def __init__ (self, input_dim,
                        pretrained="runwayml/stable-diffusion-v1-5",
                        pretrained_revision=None):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae", revision=pretrained_revision)
        self.vae.requires_grad_(False)
        self.encoder = FmriEncoder(input_dim)

    def forward (self, fmri):
        latents = self.encoder(fmri)
        return self.vae.decode(latents.to(dtype=self.vae.dtype)).sample.float()

