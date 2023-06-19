#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import StableUnCLIPImageNormalizer

from controlnet.validation import log_validation, log_validation_pipeline_unclip
from controlnet.pipeline import StableDiffusionControlNetUnCLIPPipeline
from controlnet.args_parser import DictAction, config_merge_dict
from controlnet.dataset import make_train_dataset_embed, collate_fn_embed, ControlNetDepthUnCLIPDataset
from controlnet.utils import import_model_class_from_model_name_or_path, get_full_repo_name

import copy
import ipdb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0.dev0")

logger = get_logger(__name__)



def main(cfg):
    logging_dir = Path(cfg.output_dir, cfg.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=cfg.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

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
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.push_to_hub:
            if cfg.hub_model_id is None:
                repo_name = get_full_repo_name(Path(cfg.output_dir).name, token=cfg.hub_token)
            else:
                repo_name = cfg.hub_model_id
            create_repo(repo_name, exist_ok=True, token=cfg.hub_token)
            repo = Repository(cfg.output_dir, clone_from=repo_name, token=cfg.hub_token)

            with open(os.path.join(cfg.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif cfg.output_dir is not None:
            os.makedirs(cfg.output_dir, exist_ok=True)

    # Load the tokenizer
    if cfg.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, revision=cfg.revision, use_fast=False)
    elif cfg.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=cfg.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(cfg.pretrained_model_name_or_path, cfg.revision)

    # image encoding components
    feature_extractor = CLIPImageProcessor.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="feature_extractor")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="image_encoder")
    # image noising components
    image_normalizer = StableUnCLIPImageNormalizer.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="image_normalizer")
    image_noising_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="image_noising_scheduler")
    
    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="text_encoder", revision=cfg.revision
    )
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_name_or_path, subfolder="vae", revision=cfg.revision)
    unet = UNet2DConditionModel.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="unet", revision=cfg.revision
    )

    if cfg.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(cfg.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet)

    validation_pipeline = StableDiffusionControlNetUnCLIPPipeline.from_pretrained(
        cfg.pretrained_model_name_or_path,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        image_normalizer=image_normalizer,
        image_noising_scheduler=image_noising_scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        # scheduler: is in the validation func
        controlnet=controlnet,
    )

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            i = len(weights) - 1

            while len(weights) > 0:
                weights.pop()
                model = models[i]

                sub_dir = "controlnet"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

                i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)

    controlnet.train()

    if cfg.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if cfg.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.scale_lr:
        cfg.learning_rate = (
            cfg.learning_rate * cfg.gradient_accumulation_steps * cfg.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if cfg.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    train_dataset = ControlNetDepthUnCLIPDataset(cfg, tokenizer, feature_extractor)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn_embed,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if cfg.max_train_steps is None:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * cfg.gradient_accumulation_steps,
        num_training_steps=cfg.max_train_steps * cfg.gradient_accumulation_steps,
        num_cycles=cfg.lr_num_cycles,
        power=cfg.lr_power,
    )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if overrode_max_train_steps:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfg.num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # ipdb.set_trace()
    if accelerator.is_main_process:
        # ipdb.set_trace()
        # tracker_config = dict(vars(cfg)) ## for args use this
        tracker_config = dict(copy.deepcopy(cfg)) ## for dict cfg use this
        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")
        tracker_config.pop("validation_image_embed")

        accelerator.init_trackers(cfg.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = cfg.train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {cfg.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            path = os.path.basename(cfg.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(cfg.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{cfg.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            cfg.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(cfg.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step * cfg.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, cfg.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, cfg.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                with torch.autocast("cuda"):
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # ipdb.set_trace()
                    ## get image embeddings
                    images_preprocessed = batch["images_preprocessed"].to(dtype=weight_dtype)
                    image_embeds = image_encoder(images_preprocessed).image_embeds
                    ## add noise to image embeddings
                    if cfg.noise_level >= 1000:
                        train_noise = random.randint(0, 999)
                    else:
                        train_noise = cfg.noise_level

                    image_embeds = validation_pipeline.noise_image_embeddings(
                                    image_embeds=image_embeds,
                                    noise_level=train_noise,
                                    generator=None,
                                )

                    controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                    # ipdb.set_trace()
                    controlnet_image_embeds_type = cfg.get("controlnet_image_embeds_type", "empty")
                    if controlnet_image_embeds_type == "image":
                        controlnet_image_embeds = image_embeds
                    else:
                        controlnet_image_embeds = torch.zeros_like(image_embeds)

                    down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_latents,
                        timesteps,
                        class_labels=controlnet_image_embeds,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=controlnet_image,
                        return_dict=False,
                    )

                    # Predict the noise residual
                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        class_labels=image_embeds,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=cfg.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % cfg.checkpointing_steps == 0:
                        save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if cfg.validation_prompt is not None and global_step % cfg.validation_steps == 0:
                        log_validation_pipeline_unclip(
                            # logger,
                            validation_pipeline,
                            controlnet,
                            cfg,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= cfg.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = accelerator.unwrap_model(controlnet)
        controlnet.save_pretrained(cfg.output_dir)

        if cfg.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()


# if __name__ == "__main__":
#     args = parse_args()
#     main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/controlnet.yaml")
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction, ##NOTE cannot support multi-level config change
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')

    args = parser.parse_args()

    ## read from cmd line
    # ipdb.set_trace()
    # Load the YAML configuration file
    config = OmegaConf.load(args.config)
    # Merge the command-line arguments with the configuration file
    if args.options is not None:
        # config = OmegaConf.merge(config, cfg.options)
        config_merge_dict(args.options, config)
    
    main(config)
