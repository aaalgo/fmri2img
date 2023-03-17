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

class FmriEncoder (torch.nn.Module):
    def __init__ (self, Din, Dout, seqlen=77):
        super().__init__()
        self.seqlen = seqlen
        self.dim = Dout
        self.linear = torch.nn.Linear(Din, 8192)
        self.linear2 = torch.nn.Linear(8192, Dout * seqlen)
        self.final_layer_norm = nn.LayerNorm(Dout)
    
    def forward (self, X):
        hidden_state = F.relu(self.linear.forward(X))
        hidden_state = self.linear2.forward(hidden_state)
        hidden_state = hidden_state.reshape((-1, self.seqlen, self.dim))
        hidden_state = self.final_layer_norm(hidden_state)
        return hidden_state

class Fmri2Image (torch.nn.Module):
    def __init__ (self, input_dim, encode_dim,
                        pretrained="runwayml/stable-diffusion-v1-5",
                        pretrained_revision=None):
        super().__init__()
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae", revision=pretrained_revision)
        self.unet = UNet2DConditionModel.from_pretrained(pretrained, subfolder="unet", revision=pretrained_revision)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.encoder = FmriEncoder(input_dim, encode_dim)
        #self.noise_scheduler.config.prediction_type = 'v_prediction'

    def decode (self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = image.clamp(-1, 1)
        #image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        return image

    def forwardTrain (self, fmri, image):
        with torch.no_grad():
            latents = self.vae.encode(image).latent_dist.sample().detach()
            assert not torch.any(torch.isnan(latents))
            latents = latents * self.vae.config.scaling_factor
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps).to(dtype=image.dtype)

        encoder_hidden_states = self.encoder(fmri).to(dtype=latents.dtype)
        assert not torch.any(torch.isnan(encoder_hidden_states))
        assert not torch.any(torch.isnan(noisy_latents))
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
        assert not torch.any(torch.isnan(model_pred))
        assert not torch.any(torch.isnan(target))
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return {
                'loss': loss,
                'image': image,
                'latents': model_pred}


if __name__ == '__main__':
    from config import *
    model = Fmri2Image(DIM)
