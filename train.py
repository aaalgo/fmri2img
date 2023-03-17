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
from accelerate.utils import ProjectConfiguration, set_seed

from torch.utils.data import Dataset
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available

logger = get_logger(__name__)

def save_progress(text_encoder, placeholder_token_id, accelerator, args, save_path):
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)

class Fmri2ImageDataset (Dataset):
    def __init__(self, path, is_train=True, size=512,
        stimuli_path='data/nsd_stimuli.hdf5'
    ):
        self.imgBrick = h5py.File(stimuli_path, 'r')['imgBrick']
        self.samples = []
        #self.imageCache = {}
        with open(path, 'rb') as f:
            for one in pickle.load(f):
                k = one['image_id']
                v = one['features']
                for feature in v:
                    self.samples.append((k, v))
                        'image_id': k,
                        'feature': v
                        }
        self.size = size
        self._length = len(self.samples)

        if set == "train":
            random.shuffle(self.samples)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        k, v = self.samples[i]
        image = self.imgBrick[k, :, :, :]

        image = (image / 127.5 - 1.0).astype(np.float32)


        if is_train:
            # augment
            pass
        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return {
                'fmri': v,
                'pixel_values': image
                }

def main():
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    model = models.Fmri2Image(DIM)

    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        model.unet.train()
        model.unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.encoder.parameters(),  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = Fmri2ImageDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=args.placeholder_token,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        set="train",
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the unet and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    model.unet.to(accelerator.device, dtype=weight_dtype)
    model.vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()

    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(model.encoder):
                # Convert images to latent space
                loss = model.forwardTrain(batch['fmri'], batch['image'])
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"learned_embeds-steps-{global_step}.bin")
                    save_progress(text_encoder, placeholder_token_id, accelerator, args, save_path)

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "learned_embeds.bin")
        save_progress(text_encoder, placeholder_token_id, accelerator, args, save_path)
    accelerator.end_training()


if __name__ == "__main__":
    main()
