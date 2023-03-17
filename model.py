import torch
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
    def __init__ (self, Din, Dout):
        self.linear = torch.nn.Linear(Din, Dout)
    
    def forward (self, X):
        return self.linear(X)

class Fmri2Image (torch.nn.Module):
    def __init__ (self, D,
                        pretrained="runwayml/stable-diffusion-v1-5",
                        pretrained_revision=None):
        super().__init__()
        self.noise_scheduler = DDPMScheduler.from_pretrained(pretrained, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae", revision=pretrained_revision)
        self.unet = UNet2DConditionModel.from_pretrained(pretrained, subfolder="unet", revision=pretrained_revision)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.encoder = FmriEncoder(D, 1000)

    def forwardTrain (self, fmri, image):
        with torch.no_grad():
            latents = self.vae.encode(image).latent_dist.sample().detach()
            latents = latents * self.vae.config.scaling_factor
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        encoder_hidden_states = self.encoder(fmri) #text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
        return F.mse_loss(model_pred.float(), target.float(), reduction="mean")


if __name__ == '__main__':
    from config import *
    model = Fmri2Image(DIM)
