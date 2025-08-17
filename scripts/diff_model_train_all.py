# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Heavily modified by Emerson Grabke based on MONAI
# Enables training base MAISI models as well as our variants based on "base_maisi" flag in config_maisi.json

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from torch import nn
import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

import monai
from monai.data import ThreadDataLoader, partition_dataset
from monai.transforms import Compose, MapTransform
from monai.utils import first
from monai.losses import FocalLoss

from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .utils import define_instance
import numpy as np

from typing import Union


def load_filenames(data_list_path: str) -> list:
    """
    Load filenames from the JSON data list.

    Args:
        data_list_path (str): Path to the JSON data list file.

    Returns:
        list: List of filenames.
    """
    if data_list_path.endswith('.json'):
        with open(data_list_path, "r") as f:
            filenames_train = json.load(f)
    else:
        raise ValueError("Data list file must be .json")
    return filenames_train

def prepare_data_base(
    train_files: list, device: torch.device, cache_rate: float, num_workers: int = 2, batch_size: int = 1, imkey: str = 'image'
) -> ThreadDataLoader:
    """
    Prepare training data per base MAISI

    Args:
        train_files (list): List of training files.
        device (torch.device): Device to use for training.
        cache_rate (float): Cache rate for dataset.
        num_workers (int): Number of workers for data loading.
        batch_size (int): Mini-batch size.

    Returns:
        DataLoader: Data loader for training.
    """

    def _load_data_from_file(file_path, key):
        with open(file_path) as f:
            return torch.FloatTensor(json.load(f)[key])

    train_transforms = Compose(
        [
            monai.transforms.LoadImaged(keys=[imkey]),
            monai.transforms.EnsureChannelFirstd(keys=[imkey]),
            monai.transforms.Lambdad(keys="top_region_index", func=lambda x: x * 1e2),
            monai.transforms.Lambdad(keys="bottom_region_index", func=lambda x: x * 1e2),
            monai.transforms.Lambdad(keys="spacing", func=lambda x: x * 1e2),
        ]
    )

    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers
    )

    return ThreadDataLoader(train_ds, num_workers=6, batch_size=batch_size, shuffle=True)

def prepare_data_text_class(
    train_files: list, device: torch.device, cache_rate: float, num_workers: int = 2, batch_size: int = 1, imkey: str = 'image'
) -> ThreadDataLoader:
    """
    Prepare training data.

    Args:
        train_files (list): List of training files.
        device (torch.device): Device to use for training.
        cache_rate (float): Cache rate for dataset.
        num_workers (int): Number of workers for data loading.
        batch_size (int): Mini-batch size.
        imkey (str): Image key to use for dataset.

    Returns:
        ThreadDataLoader: Data loader for training.
    """

    train_transforms = Compose(
        [
            monai.transforms.LoadImaged(keys=[imkey,'text']),
            monai.transforms.EnsureChannelFirstd(keys=[imkey]),
            monai.transforms.Lambdad(keys="pirads", func=lambda x: torch.FloatTensor(x)), # Label
            monai.transforms.Lambdad(keys="spacing", func=lambda x: torch.FloatTensor(x)),
            monai.transforms.Lambdad(keys=["pirads", "spacing"], func=lambda x: x * 1e2),
        ]
    )

    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers
    )

    return ThreadDataLoader(train_ds, num_workers=6, batch_size=batch_size, shuffle=True)

def prepare_data_text(
    train_files: list, device: torch.device, cache_rate: float, num_workers: int = 2, batch_size: int = 1, imkey: str = 'image'
) -> ThreadDataLoader:
    """
    Prepare training data.

    Args:
        train_files (list): List of training files.
        device (torch.device): Device to use for training.
        cache_rate (float): Cache rate for dataset.
        num_workers (int): Number of workers for data loading.
        batch_size (int): Mini-batch size.
        imkey (str): Image key to use for dataset.

    Returns:
        ThreadDataLoader: Data loader for training.
    """

    train_transforms = Compose(
        [
            monai.transforms.LoadImaged(keys=[imkey,'text']),
            monai.transforms.EnsureChannelFirstd(keys=[imkey]),
            monai.transforms.Lambdad(keys="pirads", func=lambda x: torch.FloatTensor(x)),
            monai.transforms.Lambdad(keys="spacing", func=lambda x: torch.FloatTensor(x)),
            monai.transforms.Lambdad(keys=["pirads", "spacing"], func=lambda x: x * 1e2),
        ]
    )

    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers
    )

    return ThreadDataLoader(train_ds, num_workers=6, batch_size=batch_size, shuffle=True)

def prepare_data_path(
    train_files: list, device: torch.device, cache_rate: float, num_workers: int = 2, batch_size: int = 1, imkey: str = 'image'
) -> ThreadDataLoader:
    """
    Prepare training data.

    Args:
        train_files (list): List of training files.
        device (torch.device): Device to use for training.
        cache_rate (float): Cache rate for dataset.
        num_workers (int): Number of workers for data loading.
        batch_size (int): Mini-batch size.
        imkey (str): Image key to use for dataset.

    Returns:
        ThreadDataLoader: Data loader for training.
    """

    train_transforms = Compose(
        [
            monai.transforms.LoadImaged(keys=[imkey]),
            monai.transforms.EnsureChannelFirstd(keys=[imkey]),
            monai.transforms.Lambdad(keys="pirads", func=lambda x: torch.FloatTensor(x)),
            monai.transforms.Lambdad(keys="spacing", func=lambda x: torch.FloatTensor(x)),
            monai.transforms.Lambdad(keys=["pirads", "spacing"], func=lambda x: x * 1e2),
        ]
    )

    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers
    )

    return ThreadDataLoader(train_ds, num_workers=6, batch_size=batch_size, shuffle=True)


def load_unet(args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> Union[torch.nn.Module, torch.nn.Module]:
    """
    Load the UNet model.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to load the model on.
        logger (logging.Logger): Logger for logging information.

    Returns:
        torch.nn.Module: Loaded UNet model.
    """
    unet = define_instance(args, "diffusion_unet_def").to(device)
    unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)
    checkpoint_unet=None

    if dist.is_initialized():
        unet = DistributedDataParallel(unet, device_ids=[device], find_unused_parameters=True)

    if args.existing_ckpt_filepath is None and not args.resume:
        logger.info("Training from scratch.")
    elif args.resume:
        checkpoint_unet = torch.load(f"{args.model_dir}/{args.model_filename}", map_location=device)
        if dist.is_initialized():
            unet.module.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        else:
            unet.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        logger.info(f"Resuming from previous at {args.model_dir}/{args.model_filename}")
    else:
        checkpoint_unet = torch.load(f"{args.existing_ckpt_filepath}", map_location=device)
        if dist.is_initialized():
            unet.module.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        else:
            unet.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        logger.info(f"Pretrained checkpoint {args.existing_ckpt_filepath} loaded.")

    return unet, checkpoint_unet

def load_unet_text(args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> Union[torch.nn.Module, torch.nn.Module]:
    """
    Load the UNet model.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to load the model on.
        logger (logging.Logger): Logger for logging information.

    Returns:
        torch.nn.Module: Loaded UNet model.
    """
    unet = define_instance(args, "diffusion_unet_def").to(device)
    unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)
    checkpoint_unet=None

    if dist.is_initialized():
        unet = DistributedDataParallel(unet, device_ids=[device], find_unused_parameters=True)

    if args.existing_ckpt_filepath is None and not args.resume:
        if args.use_pretrained_unet:
            pretrained_unet = torch.load(args.pretrained_unet_path, map_location=device)
            if dist.is_initialized():
                unet.module.unet.load_state_dict(pretrained_unet["unet_state_dict"], strict=True)
            else:
                unet.unet.load_state_dict(pretrained_unet["unet_state_dict"], strict=True)
            logger.info(f"Pretrained UNet loaded from {args.pretrained_unet_path}")
        else:
            logger.info("Training from scratch.")
    elif args.resume:
        checkpoint_unet = torch.load(f"{args.model_dir}/{args.model_filename}", map_location=device)
        if dist.is_initialized():
            unet.module.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        else:
            unet.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        logger.info(f"Resuming from previous at {args.model_dir}/{args.model_filename}")
    else:
        checkpoint_unet = torch.load(f"{args.existing_ckpt_filepath}", map_location=device)
        if dist.is_initialized():
            unet.module.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        else:
            unet.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        logger.info(f"Pretrained checkpoint {args.existing_ckpt_filepath} loaded.")

    if args.freeze_unet:
        if dist.is_initialized():
            for param in unet.module.unet.parameters():
                param.requires_grad = False
            unet.module.unet.eval()
        else:
            for param in unet.unet.parameters():
                param.requires_grad = False
            unet.unet.eval()

    return unet, checkpoint_unet

def calculate_scale_factor(
    train_loader: ThreadDataLoader, device: torch.device, logger: logging.Logger, imkey: str,
) -> Union[torch.Tensor,tuple]:
    """
    Calculate the scaling factor for the dataset.

    Args:
        train_loader (ThreadDataLoader): Data loader for training.
        device (torch.device): Device to use for calculation.
        logger (logging.Logger): Logger for logging information.

    Returns:
        torch.Tensor: Calculated scaling factor.
    """
    if imkey == 'multi':
        check_data = first(train_loader)
        z = check_data[imkey].to(device)
        axt2_latent = z[:,:4,:,:,:] # 0-3
        highb_latent = z[:,4:8,:,:,:] # 4-7
        adc_latent = z[:,8:,:,:,:] # 8-11

        axt2_scale, highb_scale, adc_scale = 1 / torch.std(axt2_latent), 1 / torch.std(highb_latent), 1 / torch.std(adc_latent)
        logger.info(f"Scaling factors set to {(axt2_scale, highb_scale, adc_scale)}.")

        if dist.is_initialized():
            dist.barrier()
            dist.all_reduce(axt2_scale, op=torch.distributed.ReduceOp.AVG)
            dist.all_reduce(highb_scale, op=torch.distributed.ReduceOp.AVG)
            dist.all_reduce(adc_scale, op=torch.distributed.ReduceOp.AVG)
        logger.info(f"scale_factors -> {(axt2_scale, highb_scale, adc_scale)}.")
        return axt2_scale, highb_scale, adc_scale
    else:
        check_data = first(train_loader)
        z = check_data[imkey].to(device)
        scale_factor = 1 / torch.std(z)
        logger.info(f"Scaling factor set to {scale_factor}.")

        if dist.is_initialized():
            dist.barrier()
            dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
        logger.info(f"scale_factor -> {scale_factor}.")
        return scale_factor


def create_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    """
    Create optimizer for training.

    Args:
        model (torch.nn.Module): Model to optimize.
        lr (float): Learning rate.

    Returns:
        torch.optim.Optimizer: Created optimizer.
    """
    return torch.optim.AdamW(params=model.parameters(), lr=lr)

def create_optimizer_text(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    """
    Create optimizer for training.

    Args:
        model (torch.nn.Module): Model to optimize.
        lr (float): Learning rate.

    Returns:
        torch.optim.Optimizer: Created optimizer.
    """
    return torch.optim.AdamW([{'params':model.module.unet.parameters()},
                              {'params': model.module.ella.parameters(), 'lr': lr/10}], lr=lr) #lr/10

def create_optimizer_textclass(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    """
    Create optimizer for training.

    Args:
        model (torch.nn.Module): Model to optimize.
        lr (float): Learning rate.

    Returns:
        torch.optim.Optimizer: Created optimizer.
    """
    return torch.optim.AdamW([{'params':model.module.unet.parameters()},
                              {'params': model.module.ella.parameters(), 'lr': lr/10}, # lr/10
                              {'params': model.module.classifier.parameters(), 'lr': lr/100}], lr=lr) # lr/100

def create_lr_scheduler(optimizer: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.PolynomialLR:
    """
    Create learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule.
        total_steps (int): Total number of training steps.

    Returns:
        torch.optim.lr_scheduler.PolynomialLR: Created learning rate scheduler.
    """
    return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)


def train_one_epoch(
    epoch: int,
    unet: torch.nn.Module,
    train_loader: ThreadDataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.PolynomialLR,
    loss_pt,
    scaler: GradScaler,
    scale_factor: Union[torch.Tensor,tuple],
    noise_scheduler: torch.nn.Module,
    num_images_per_batch: int,
    num_train_timesteps: int,
    device: torch.device,
    logger: logging.Logger,
    local_rank: int,
    imkey: str,
    base_maisi: bool,
    text_maisi: bool,
    tensorboard_writer: SummaryWriter | None,
    tfevents_dir: str,
    text_class_weight: float | None,
    text_class_loss: torch.nn.Module | None,
) -> torch.Tensor:
    """
    Train the model for one epoch.

    Args:
        epoch (int): Current epoch number.
        unet (torch.nn.Module): UNet model.
        train_loader (ThreadDataLoader): Data loader for training.
        optimizer (torch.optim.Optimizer): Optimizer.
        lr_scheduler (torch.optim.lr_scheduler.PolynomialLR): Learning rate scheduler.
        loss_pt (torch.nn.L1Loss): Loss function.
        scaler (GradScaler): Gradient scaler for mixed precision training.
        scale_factor (torch.Tensor): Scaling factor.
        noise_scheduler (torch.nn.Module): Noise scheduler.
        num_images_per_batch (int): Number of images per batch.
        num_train_timesteps (int): Number of training timesteps.
        device (torch.device): Device to use for training.
        logger (logging.Logger): Logger for logging information.
        local_rank (int): Local rank for distributed training.
        base_maisi (bool): Whether we're running with base MAISI.
        text_maisi (bool): Whether we're running with text MAISI.

    Returns:
        torch.Tensor: Training loss for the epoch.
    """
    if local_rank == 0:
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch + 1}, lr {current_lr}.")

    _iter = 0
    loss_torch = torch.zeros(2, dtype=torch.float, device=device)

    if not text_maisi:
        unet.train()
    total_step=0+epoch*len(train_loader)

    for train_data in train_loader:
        current_lr = optimizer.param_groups[0]["lr"]

        _iter += 1
        images = train_data[imkey].to(device)
        if imkey=='multi':
            images[:,:4,:,:,:]*=scale_factor[0] # 0-3
            images[:,4:8,:,:,:]*=scale_factor[1] # 4-7
            images[:,8:,:,:,:]*=scale_factor[2] # 8-11
        else:
            images = images * scale_factor

        if base_maisi:
            top_region_index_tensor = train_data["top_region_index"].to(device)
            bottom_region_index_tensor = train_data["bottom_region_index"].to(device)
            spacing_tensor = train_data["spacing"].to(device)
        elif text_maisi:
            pirads_tensor = train_data["pirads"].to(device)
            spacing_tensor = train_data["spacing"].to(device)
            text_encoding = train_data["text"].to(device)
        else:
            pirads_tensor = train_data["pirads"].to(device)
            spacing_tensor = train_data["spacing"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=True):
            n_channels = 12 if imkey=='multi' else 4
            noise = torch.randn(
                (images.size(0), n_channels, images.size(-3), images.size(-2), images.size(-1)), device=device
            )

            timesteps = torch.randint(0, num_train_timesteps, (images.shape[0],), device=images.device).long()

            noisy_latent = noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)

            class_loss = None

            if base_maisi:
                noise_pred = unet(
                    x=noisy_latent,
                    timesteps=timesteps,
                    top_region_index_tensor=top_region_index_tensor,
                    bottom_region_index_tensor=bottom_region_index_tensor,
                    spacing_tensor=spacing_tensor,
                )
            elif text_maisi:
                # text_encoding = text_encoder(text_embedding)
                if text_class_weight is not None:
                    noise_pred, class_pred = unet(
                        x=noisy_latent,
                        timesteps=timesteps,
                        spacing_tensor=spacing_tensor,
                        text_encoding=text_encoding,
                    )
                    if text_class_loss is not None:
                        class_loss = text_class_loss(class_pred, pirads_tensor)
                        isnull = train_data["text_isnull"].to(device)
                        isnull_unsqueeze = isnull.unsqueeze(1).repeat(1,2)
                        if torch.sum(1 - isnull) > 0:
                            class_loss = torch.sum(class_loss * (1 - isnull_unsqueeze)) / torch.sum(1 - isnull)
                        else:
                            class_loss = torch.tensor(0.0, device=device)
                else:
                    noise_pred = unet(
                        x=noisy_latent,
                        timesteps=timesteps,
                        pirads=pirads_tensor,
                        spacing_tensor=spacing_tensor,
                        text_encoding=text_encoding,
                    )
            else:
                noise_pred = unet(
                    x=noisy_latent,
                    timesteps=timesteps,
                    pirads=pirads_tensor,
                    spacing_tensor=spacing_tensor,
                )

            loss = loss_pt(noise_pred.float(), noise.float())
            if class_loss is not None and torch.sum(1 - isnull) > 0:
                loss += text_class_weight * class_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer) # unscale the gradients of optimizer's assigned params in-place
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1)  # clip gradient
        scaler.step(optimizer)
        scaler.update()

        lr_scheduler.step()

        loss_torch[0] += loss.item()
        loss_torch[1] += 1.0

        if local_rank == 0:
            logger.info(
                "[{0}] epoch {1}, iter {2}/{3}, loss: {4:.4f}, lr: {5:.12f}.".format(
                    str(datetime.now())[:19], epoch + 1, _iter, len(train_loader), loss.item(), current_lr
                )
            )
            tensorboard_writer.add_scalar("train_loss_iter", loss.detach().cpu().item(),total_step)
            tensorboard_writer.add_scalar("lr_iter", current_lr,total_step)
            total_step+=1

    if dist.is_initialized():
        dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)

    return loss_torch


def save_checkpoint(
    epoch: int,
    unet: torch.nn.Module,
    loss_torch_epoch: float,
    num_train_timesteps: int,
    scale_factor: Union[torch.Tensor,tuple],
    ckpt_folder: str,
    args: argparse.Namespace,
    optimizer=None,
    scheduler=None
) -> None:
    """
    Save checkpoint.

    Args:
        epoch (int): Current epoch number.
        unet (torch.nn.Module): UNet model.
        loss_torch_epoch (float): Training loss for the epoch.
        num_train_timesteps (int): Number of training timesteps.
        scale_factor (torch.Tensor): Scaling factor.
        ckpt_folder (str): Checkpoint folder path.
        args (argparse.Namespace): Configuration arguments.
    """
    unet_state_dict = unet.module.state_dict() if dist.is_initialized() else unet.state_dict()
    torch.save(
        {
            "epoch": epoch + 1,
            "loss": loss_torch_epoch,
            "num_train_timesteps": num_train_timesteps,
            "scale_factor": scale_factor, #?
            "unet_state_dict": unet_state_dict,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        },
        f"{ckpt_folder}/{args.model_filename}",
    )

class CustomLoss(torch.nn.Module):
    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.l1 = torch.nn.L1Loss(size_average, reduce, reduction)
        self.l2 = torch.nn.MSELoss(size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.l1(input, target) + self.l2(input, target)
        

def diff_model_train(env_config_path: str, model_config_path: str, model_def_path: str, num_gpus: int, resume: bool = False) -> None:
    """
    Main function to train a diffusion model.

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
        model_def_path (str): Path to the model definition file.
        num_gpus (int): Number of GPUs to use for training.
        resume (bool): Whether to resume training.
    """
    # wandb.login()
    with open(env_config_path, "r") as f:
        env_config_json = json.load(f)
    with open(model_config_path, "r") as f:
        model_config_json = json.load(f)
    with open(model_def_path, "r") as f:
        model_def_json = json.load(f)
    args = load_config(env_config_path, model_config_path, model_def_path)
    args.resume = resume
    local_rank, world_size, device = initialize_distributed(num_gpus)
    logger = setup_logging("training")
    tensorboard_path = os.path.join(args.tfevent_path, args.exp_name)
    if local_rank == 0:
        Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_path)

    if args.base_maisi:
        args.train_json = args.json_data_list

    args.text_maisi = args.use_text_conditioning

    class_loss=None


    logger.info(f"Using {device} of {world_size}")

    if local_rank == 0:
        logger.info(f"[config] ckpt_folder -> {args.model_dir}.")
        logger.info(f"[config] data_root -> {args.embedding_base_dir}.")
        logger.info(f"[config] data_list -> {args.train_json}.")
        logger.info(f"[config] lr -> {args.diffusion_unet_train['lr']}.")
        logger.info(f"[config] num_epochs -> {args.diffusion_unet_train['n_epochs']}.")
        logger.info(f"[config] num_train_timesteps -> {args.noise_scheduler['num_train_timesteps']}.")

        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    filenames_train = load_filenames(args.train_json)

    if local_rank == 0:
        logger.info(f"num_files_train: {len(filenames_train)}")
        if hasattr(args,'institute1') and args.institute1:
            print("Pruning non-Institute1 cases")

    train_files = []
    total_step = 0
    for _i in range(len(filenames_train)):
        file_data = filenames_train[_i]
        if hasattr(args,'institute1') and args.institute1 and not (file_data['dataset'] == "Institute1"): # Only use Institute1 data
            continue
        if args.diffusion_unet_train['imkey'] == 'multi':
            filename = file_data['axt2'].replace("axt2.nii.gz", "multi_emb.nii.gz")
        else:
            filename = file_data[args.diffusion_unet_train['imkey']].replace(".nii.gz", "_emb.nii.gz")
        str_img = os.path.join(args.embedding_base_dir, filename)
        if not os.path.exists(str_img):
            continue

        if args.base_maisi:
            train_files.append(
                {args.diffusion_unet_train['imkey']: str_img, "top_region_index": np.array([0,0,0,1],dtype=np.float32), "bottom_region_index": np.array([0,0,0,1],dtype=np.float32), "spacing": np.array([0.5,0.5,0.75],dtype=np.float32)}
            )
        elif args.text_maisi:
            text = file_data['text']
            if args.use_text_class_pred:
                train_files.append(
                    {args.diffusion_unet_train['imkey']: str_img, "pirads": file_data['pirads'], "spacing": file_data['spacing'], "text": text, "text_isnull": file_data['text_isnull']}
                )
            else:
                train_files.append(
                    {args.diffusion_unet_train['imkey']: str_img, "pirads": file_data['pirads'], "spacing": file_data['spacing'], "text": text}
                )
        else:
            train_files.append(
                {args.diffusion_unet_train['imkey']: str_img, "pirads": file_data['pirads'], "spacing": file_data['spacing']}
            )

    if local_rank==0:
        logger.info(f"num_files_train final: {len(train_files)}")

    if dist.is_initialized():
        train_files = partition_dataset(
            data=train_files, shuffle=True, num_partitions=dist.get_world_size(), even_divisible=True
        )[local_rank]

    if args.text_maisi:
        if args.use_text_class_pred:
            train_loader = prepare_data_text_class(
                train_files, device, args.diffusion_unet_train["cache_rate"], args.diffusion_unet_train["num_workers"], args.diffusion_unet_train["batch_size"], args.diffusion_unet_train['imkey']
            )
            class_loss = FocalLoss(alpha=0.75, use_softmax=True, reduction='none')
        else:
            train_loader = prepare_data_text(
                train_files, device, args.diffusion_unet_train["cache_rate"], args.diffusion_unet_train["num_workers"], args.diffusion_unet_train["batch_size"], args.diffusion_unet_train['imkey']
            )
        unet, checkpoint = load_unet_text(args, device, logger)
    else:
        if args.base_maisi:
            prepare_data_func = prepare_data_base
        else:
            prepare_data_func = prepare_data_path

        train_loader = prepare_data_func(
            train_files, device, args.diffusion_unet_train["cache_rate"], args.diffusion_unet_train["num_workers"], args.diffusion_unet_train["batch_size"], args.diffusion_unet_train['imkey']
        )

        unet, checkpoint = load_unet(args, device, logger)
    noise_scheduler = define_instance(args, "noise_scheduler")

    scale_factor = calculate_scale_factor(train_loader, device, logger, args.diffusion_unet_train['imkey'])

    
    if hasattr(args,'multiple_lr') and args.multiple_lr:
        if local_rank==0:
            logger.info(f"Using multiple LR")
        if args.text_maisi:
            if args.use_text_class_pred:
                optimizer = create_optimizer_textclass(unet, args.diffusion_unet_train["lr"])
            else:
                optimizer = create_optimizer_text(unet, args.diffusion_unet_train["lr"])
        else:
            optimizer = create_optimizer(unet, args.diffusion_unet_train["lr"])
    else:
        optimizer = create_optimizer(unet, args.diffusion_unet_train["lr"])

    if args.resume or args.existing_ckpt_filepath is not None:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("Warning: Resuming but optimizer state not found in checkpoint")


    total_steps = (args.diffusion_unet_train["n_epochs"] * len(train_loader.dataset)) / args.diffusion_unet_train[
        "batch_size"
    ]
    lr_scheduler = create_lr_scheduler(optimizer, total_steps)

    if args.resume or args.existing_ckpt_filepath is not None:
        if 'scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            print("Warning: Resuming but scheduler state not found in checkpoint")

    loss_pt = torch.nn.L1Loss()
    scaler = GradScaler("cuda")

    torch.set_float32_matmul_precision("highest")
    logger.info("torch.set_float32_matmul_precision -> highest.")
    
    if args.resume or args.existing_ckpt_filepath is not None:
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0

    for epoch in range(start_epoch,args.diffusion_unet_train["n_epochs"]):
        loss_torch = train_one_epoch(
            epoch,
            unet,
            train_loader,
            optimizer,
            lr_scheduler,
            loss_pt,
            scaler,
            scale_factor,
            noise_scheduler,
            args.diffusion_unet_train["batch_size"],
            args.noise_scheduler["num_train_timesteps"],
            device,
            logger,
            local_rank,
            args.diffusion_unet_train['imkey'],
            args.base_maisi,
            args.text_maisi,
            tensorboard_writer if local_rank==0 else None,
            tfevents_dir=tensorboard_path,
            text_class_weight = args.text_class_pred_weight if args.use_text_class_pred else None,
            text_class_loss = class_loss if args.use_text_class_pred else None,
        )

        loss_torch = loss_torch.tolist()
        if torch.cuda.device_count() == 1 or local_rank == 0:
            loss_torch_epoch = loss_torch[0] / loss_torch[1]
            logger.info(f"epoch {epoch + 1} average loss: {loss_torch_epoch:.4f}.")
            tensorboard_writer.add_scalar("train_loss_epoch", loss_torch_epoch,total_step)
            total_step+=1

            save_checkpoint(
                epoch,
                unet,
                loss_torch_epoch,
                args.noise_scheduler["num_train_timesteps"],
                scale_factor,
                args.model_dir,
                args,
                optimizer,
                lr_scheduler
            )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training")
    parser.add_argument(
        "--env_config",
        type=str,
        default="./configs/environment_maisi_diff_model_train.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/config_maisi_diff_model_train.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument(
        "--model_def", type=str, default="./configs/config_maisi.json", help="Path to model definition file"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for training")
    parser.add_argument("--resume", action="store_true", help="Resume training")

    args = parser.parse_args()
    diff_model_train(args.env_config, args.model_config, args.model_def, args.num_gpus, args.resume)
