#!/usr/bin/env python3

import logging
import math
import os
import random
import pickle
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
import h5py
import models
import PIL
import imgaug.augmenters as iaa
import wandb
from config import *

logger = get_logger(__name__)


class Fmri2ImageDataset (Dataset):
    def __init__(self, path, is_train=True, size=IMAGE_SIZE,
        stimuli_path='data/nsd_stimuli.hdf5'
    ):
        self.imgBrick = h5py.File(stimuli_path, 'r')['imgBrick']
        _, H, W, _ = self.imgBrick.shape
        assert H == W
        self.samples = []
        #self.imageCache = {}
        with open(path, 'rb') as f:
            for one in pickle.load(f):
                k = one['image_id']
                for v in one['features']:
                    self.samples.append((k, v))
        self.size = size
        self.is_train = is_train
        if is_train:
            self.aug = iaa.Sequential([
                iaa.GaussianBlur(sigma=(0.0, 3.0)),
                iaa.Resize({"height": IMAGE_SIZE, "width": IMAGE_SIZE}),
                iaa.Affine(scale=(1.0, 1.2)),
                iaa.CropToFixedSize(IMAGE_SIZE, IMAGE_SIZE)
                ])
        else:
            self.aug = iaa.Sequential([
                iaa.Resize({"height": IMAGE_SIZE, "width": IMAGE_SIZE})
                ])
        self.factor = 10

    def __len__(self):
        return len(self.samples) * self.factor

    def __getitem__(self, i):
        k, v = self.samples[i % len(self.samples)]
        image = self.aug(image=self.imgBrick[k, :, :, :])
        image = (image / 127.5 - 1.0).astype(np.float32)
        v = torch.from_numpy(v).float()
        image = torch.from_numpy(image).permute(2, 0, 1)
        assert v.shape == (DIM,)
        assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert not torch.any(torch.isnan(v))
        assert not torch.any(torch.isnan(image))
        return {
                'fmri': v,
                'pixel_values': image
                }

def make_image (tensor):
    v = ((tensor + 1.0) * 127.5).detach().cpu().permute(1,2,0).numpy()
    v = np.clip(np.rint(v), 0, 255).astype(np.uint8)
    return v #return PIL.Image.fromarray(v)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, default='data/train.pkl', help='')
    parser.add_argument("--stimuli", type=str, default='data/nsd_stimuli.hdf5', help='')
    parser.add_argument("--output_dir", type=str, default='output', help='')
    parser.add_argument("--learning_rate", type=float, default=1e-3, help='')
    parser.add_argument("--epochs", type=int, default=10000, help='')
    args = parser.parse_args()

    assert is_wandb_available()

    accelerator = Accelerator(
        mixed_precision=MIXED_PRECISION,
        gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
        log_with=REPORT_TO,
        project_dir=args.output_dir
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    model = models.Fmri2Image(DIM, ENCODE_DIM)

    '''
    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        model.unet.train()
        model.unet.enable_gradient_checkpointing()
    '''

    if SCALE_LR:
        args.learning_rate = args.learning_rate * GRADIENT_ACCUMULATION_STEPS * BATCH_SIZE * accelerator.num_processes

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.encoder.parameters(),  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(ADAM_BETA1, ADAM_BETA2),
        weight_decay=ADAM_WEIGHT_DECAY,
        eps=ADAM_EPSILON
    )

    # Dataset and DataLoaders creation:
    train_dataset = Fmri2ImageDataset(args.samples, stimuli_path=args.stimuli)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=DATALOADER_NUM_WORKERS
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / GRADIENT_ACCUMULATION_STEPS)
    max_train_steps = args.epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        LR_SCHEDULER,
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS * GRADIENT_ACCUMULATION_STEPS,
        num_training_steps=max_train_steps * GRADIENT_ACCUMULATION_STEPS
    )

    model.encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model.encoder, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the unet and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    #if accelerator.mixed_precision == "fp16":
    #    weight_dtype = torch.float16
    #else:
    #    assert False

    # Move vae and unet to device and cast to weight_dtype
    #model.unet.to(accelerator.device, dtype=weight_dtype)
    model.vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    '''
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    '''

    if accelerator.is_main_process:
        accelerator.init_trackers("fmri2image", config=vars(args))

    total_batch_size = BATCH_SIZE * accelerator.num_processes * GRADIENT_ACCUMULATION_STEPS

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")

    def save (output):
        logger.info("Saving model to %s ..." % output)
        torch.save(accelerator.unwrap_model(model.encoder).state_dict(), output)

    global_step = 0
    for epoch in range(args.epochs):
        model.encoder.train()
        if accelerator.is_main_process:
            generator = tqdm(train_dataloader)
        else:
            generator = train_dataloader

        for batch in generator:

            with accelerator.accumulate(model.encoder):
                # Convert images to latent space
                out = model.forwardTrain(batch['fmri'], batch['pixel_values'].to(dtype=weight_dtype))
                loss = out['loss']
                if torch.isnan(loss):
                    continue

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                accelerator.log(logs, step=global_step)
                global_step += 1

        if accelerator.is_main_process:
            pred = make_image(out['images'][0])
            image = make_image(out['targets'][0])
            logs = {'image': wandb.Image(PIL.Image.fromarray(np.concatenate([image, pred], axis=1)))}
            accelerator.log(logs, step=global_step)

        if epoch % 1 == 0: 
            if accelerator.is_main_process:
                save(os.path.join(args.output_dir, f"fmri2image-{epoch}.bin"))

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save(os.path.join(args.output_dir, f"fmri2image.bin"))
    accelerator.end_training()


if __name__ == "__main__":
    main()
