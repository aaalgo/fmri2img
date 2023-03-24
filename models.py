#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from config import *



class FmriEncoder (torch.nn.Module):
    def __init__ (self, Din):
        super().__init__()
        self.fc1 = nn.Linear(Din, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        #self.fc3 = nn.Linear(4096, 4096)
        # 16x16
        self.deconv1 = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1)
        # 32x32
        self.deconv2 = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1)
        # 64x64
        self.deconv3 = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)
        # 128x128
        #self.deconv4 = nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1)

    def forward (self, X):
        x = self.fc1(X)
        x = nn.functional.gelu(x)
        x = self.fc2(x)
        x = nn.functional.gelu(x)
        #x = self.fc3(x)
        #x = nn.functional.gelu(x)
        x = x.view(-1, 16, 16, 16)
        x = self.deconv1(x)
        x = nn.functional.gelu(x)
        x = self.deconv2(x)
        x = nn.functional.gelu(x)
        x = self.deconv3(x)
        #output = nn.functional.gelu(x)
        return x

class Fmri2Image (torch.nn.Module):
    def __init__ (self, input_dim, encode_dim,
                        pretrained="runwayml/stable-diffusion-v1-5",
                        pretrained_revision=None):
        super().__init__()
        #self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained, subfolder="scheduler")
        #self.vae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae", revision=pretrained_revision)
        #self.unet = UNet2DConditionModel.from_pretrained(pretrained, subfolder="unet", revision=pretrained_revision)
        #self.vae.requires_grad_(False)
        #self.unet.requires_grad_(False)
        self.encoder = FmriEncoder(input_dim)
        #self.noise_scheduler.config.prediction_type = 'v_prediction'
        #self.noise_scheduler.config.num_train_timesteps = TRAIN_TIMESTEPS

    def decode (self, images):
        #latents = 1 / self.vae.config.scaling_factor * latents
        #image = self.vae.decode(latents).sample
        #image = image.clamp(-1, 1)
        #image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        return images

    def forwardTrain (self, fmri, targets):
        images = self.encoder(fmri)
        #latents.to(dtype=self.vae.dtype)
        #images = self.vae.decode(latents).sample
        #with torch.no_grad():
             #targets = self.vae.encode(image).latent_dist.sample().detach()
#            assert not torch.any(torch.isnan(latents))
#            latents = latents * self.vae.config.scaling_factor
#            noise = torch.randn_like(latents)
#            bsz = latents.shape[0]
#            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
#            timesteps = timesteps.long()
#            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps).to(dtype=image.dtype)
#
#        assert not torch.any(torch.isnan(encoder_hidden_states))
#        assert not torch.any(torch.isnan(noisy_latents))
#        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
#        # Get the target for loss depending on the prediction type
#        if self.noise_scheduler.config.prediction_type == "epsilon":
#            target = noise
#        elif self.noise_scheduler.config.prediction_type == "v_prediction":
#            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
#        else:
#            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
#        assert not torch.any(torch.isnan(model_pred))
#        assert not torch.any(torch.isnan(target))
        loss = F.mse_loss(images.float(), targets.float(), reduction="mean")
        return {
                'loss': loss,
                'images': images,
                'targets': targets}


if __name__ == '__main__':
    from config import *
    model = Fmri2Image(DIM)
