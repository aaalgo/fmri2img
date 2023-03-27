#!/usr/bin/env python3

import logging
import math
import os
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm import tqdm
import diffusers
from diffusers.optimization import get_scheduler
import PIL
from dataset import Fmri2ImageDataset
from models.basic_ae import Fmri2Image
from config import *
if REPORT_TO == 'wandb':
    import wandb

DUPLICATE = 10

logger = get_logger(__name__)

def make_image (tensor):
    v = ((tensor + 1.0) * 127.5).detach().cpu().permute(1,2,0).numpy()
    v = np.clip(np.rint(v), 0, 255).astype(np.uint8)
    return v #return PIL.Image.fromarray(v)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, default='data/train%02d.pkl' % SUBJECT, help='')
    parser.add_argument("--output_dir", type=str, default='snapshots', help='')
    parser.add_argument("--learning_rate", type=float, default=1e-3, help='')
    parser.add_argument("--epochs", type=int, default=1000, help='')
    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
        log_with=REPORT_TO,
        project_dir=args.output_dir
    )

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

    model = Fmri2Image(VISUAL_DIM)

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
    train_dataset = Fmri2ImageDataset(args.samples, model.IMAGE_SIZE, duplicate=DUPLICATE)
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

    model.vae.to(accelerator.device)

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
                images = model.forward(batch['fmri'])
                targets = batch['pixel_values']

                loss = torch.nn.functional.mse_loss(images.float(), targets.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                logs = {"loss": loss.detach().item()}
                accelerator.log(logs, step=global_step)
                global_step += 1

        if accelerator.is_main_process and REPORT_TO == 'wandb':
            pred = make_image(images[0])
            image = make_image(targets[0])
            logs = {'image': wandb.Image(PIL.Image.fromarray(np.concatenate([image, pred], axis=1)))}
            accelerator.log(logs, step=global_step)

        if epoch % 1 == 0: 
            if accelerator.is_main_process:
                save(os.path.join(args.output_dir, f"fmri2img-{epoch}.bin"))

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save(os.path.join(args.output_dir, f"fmri2img.bin"))
    accelerator.end_training()


if __name__ == "__main__":
    main()
