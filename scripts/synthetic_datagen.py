# Riffed off diff_model_create_training_data_multi.py
# Modified to generate a 1:1 synthetic dataset using the specified MAISI text-conditioned model

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import random

import numpy as np
import torch
import torch.distributed as dist

from monai.transforms import Compose
from monai.utils import set_determinism

from tqdm import tqdm
from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .utils import define_instance, ReconModel, ReconModelRaw

from monai.transforms.transform import MapTransform, Transform
from monai.utils.type_conversion import convert_to_tensor
from monai.data.meta_obj import get_track_meta

from monai.utils import set_determinism
from monai.data import CacheDataset, DataLoader, partition_dataset
from monai.transforms import Compose, Lambdad, LoadImaged

import SimpleITK as sitk

import pandas as pd

def resample_volume(volume, mask=False, new_spacing = [0.5,0.5, 3.0], new_size = [256,256,32]): # Note: Newsize rewritten
    if mask:
        # interpolator = sitk.sitkNearestNeighbor
        interpolator = sitk.sitkGaussian
    else:
        interpolator = sitk.sitkLanczosWindowedSinc #sitk.sitkLinear
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
    
    return sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                        volume.GetOrigin(), new_spacing, volume.GetDirection(), 0,
                        volume.GetPixelID())

def set_random_seed(seed: int) -> int:
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed.

    Returns:
        int: Set random seed.
    """
    random_seed = random.randint(0, 99999) if seed is None else seed
    set_determinism(random_seed)
    return random_seed


# Set the random seed for reproducibility
set_determinism(seed=0)


def load_models(args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> tuple:
    """
    Load the autoencoder and UNet models.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to load models on.
        logger (logging.Logger): Logger for logging information.

    Returns:
        tuple: Loaded autoencoder, UNet model, and scale factor.
    """
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    try:
        checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=True)
        autoencoder.load_state_dict(checkpoint_autoencoder)
    except Exception:
        logger.error("The trained_autoencoder_path does not exist!")

    unet = define_instance(args, "diffusion_unet_def").to(device)
    if not args.modelname_override:
        model_filepath = f"{args.model_dir}/{args.model_filename}"
    else:
        model_filepath = f"{args.model_dir}/{args.modelname_override}"
    logger.info(f"Loading from {model_filepath}.")
    checkpoint = torch.load(model_filepath, map_location=device, weights_only=False)
    unet.load_state_dict(checkpoint["unet_state_dict"], strict=True)
    logger.info(f"checkpoints {model_filepath} loaded.")

    scale_factor = checkpoint["scale_factor"]
    logger.info(f"scale_factor -> {scale_factor}.")

    return autoencoder, unet, scale_factor

def prepare_dataloader_path_eval(
    json_data_list: list | str,
    data_base_dir: list | str,
    embedding_base_dir: list | str,
    imkey: str,
    lblkey: str,
    is_val: bool,
    batch_size: int = 1,
    cache_rate: float = 0.0,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """
    Prepare dataloaders for evaluation (Includes base images)

    Args:
        json_data_list (list | str): the name of JSON files listing the data.
        data_base_dir (list | str): directory of files.
        imkey (str): image key in JSON
        lblkey (str): label key in JSON
        is_val (bool): whether to shuffle dataset (and selects number of workers)
        batch_size (int, optional): how many samples per batch to load . Defaults to 1.
        cache_rate (float, optional): percentage of cached data in total. Forces to 0.0.
        rank (int, optional): rank of the current process. Defaults to 0.
        world_size (int, optional): number of processes participating in the job. Defaults to 1.

    Returns:
        Dataloader:  The dataloader
    """
    cache_rate=0.0
    use_ddp = world_size > 1
    if isinstance(json_data_list, list):
        assert isinstance(data_base_dir, list)
        list_train = []
        for data_list, data_root, embedding_root in zip(json_data_list, data_base_dir, embedding_base_dir):
            json_data = load_filenames(data_list)
            list_train += json_data
    else:
        json_data = load_filenames(json_data_list)
        list_train = json_data

    # Don't need images just the other details
    common_transform = [
        Lambdad(keys="pirads", func=lambda x: torch.FloatTensor(x)),
        Lambdad(keys="spacing", func=lambda x: torch.FloatTensor(x)),
        Lambdad(keys=["pirads", "spacing"], func=lambda x: x * 1e2),
    ]

    train_transforms = Compose(common_transform)

    train_loader = None

    even_divisible = False
    dl_shuffle = False
    dl_workers = 8
    dl_pinmem = True

    if use_ddp:
        list_train = partition_dataset(
            data=list_train,
            shuffle=True,
            num_partitions=world_size,
            even_divisible=even_divisible,
        )[rank]
    train_ds = CacheDataset(data=list_train, transform=train_transforms, cache_rate=cache_rate, num_workers=8)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=dl_shuffle, num_workers=dl_workers, pin_memory=dl_pinmem)
    return train_loader

def prepare_dataloader_text_eval(
    json_data_list: list | str,
    data_base_dir: list | str,
    embedding_base_dir: list | str,
    imkey: str,
    lblkey: str,
    is_val: bool,
    batch_size: int = 1,
    cache_rate: float = 0.0,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """
    Prepare dataloaders for evaluation (Includes base images)

    Args:
        json_data_list (list | str): the name of JSON files listing the data.
        data_base_dir (list | str): directory of files.
        imkey (str): image key in JSON
        lblkey (str): label key in JSON
        is_val (bool): whether to shuffle dataset (and selects number of workers)
        batch_size (int, optional): how many samples per batch to load . Defaults to 1.
        cache_rate (float, optional): percentage of cached data in total. Forces to 0.0.
        rank (int, optional): rank of the current process. Defaults to 0.
        world_size (int, optional): number of processes participating in the job. Defaults to 1.

    Returns:
        Dataloader:  The dataloader
    """
    cache_rate=0.0
    use_ddp = world_size > 1
    if isinstance(json_data_list, list):
        assert isinstance(data_base_dir, list)
        list_train = []
        for data_list, data_root, embedding_root in zip(json_data_list, data_base_dir, embedding_base_dir):
            json_data = load_filenames(data_list)
            list_train += json_data
    else:
        json_data = load_filenames(json_data_list)
        list_train = json_data


    common_transform = [
        LoadImaged(keys=['text']),
        Lambdad(keys="pirads", func=lambda x: torch.FloatTensor(x)),
        Lambdad(keys="spacing", func=lambda x: torch.FloatTensor(x)),
        Lambdad(keys=["pirads", "spacing"], func=lambda x: x * 1e2),
    ]

    train_transforms = Compose(common_transform)

    train_loader = None

    
    even_divisible = False
    dl_shuffle = False
    dl_workers = 8
    dl_pinmem = True

    if use_ddp:
        list_train = partition_dataset(
            data=list_train,
            shuffle=True,
            num_partitions=world_size,
            even_divisible=even_divisible,
        )[rank]
    train_ds = CacheDataset(data=list_train, transform=train_transforms, cache_rate=cache_rate, num_workers=8)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=dl_shuffle, num_workers=dl_workers, pin_memory=dl_pinmem)
    return train_loader

def prepare_dataloader_base_eval(
    json_data_list: list | str,
    data_base_dir: list | str,
    embedding_base_dir: list | str,
    imkey: str,
    lblkey: str,
    is_val: bool,
    batch_size: int = 1,
    cache_rate: float = 0.0,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """
    Prepare dataloader for base MAISI DM eval (Includes base images)

    Args:
        json_data_list (list | str): the name of JSON files listing the data.
        data_base_dir (list | str): directory of files.
        imkey (str): image key in JSON
        lblkey (str): label key in JSON
        is_val (bool): whether to shuffle dataset (and selects number of workers)
        batch_size (int, optional): how many samples per batch to load . Defaults to 1.
        cache_rate (float, optional): percentage of cached data in total. Forces to 0.0.
        rank (int, optional): rank of the current process. Defaults to 0.
        world_size (int, optional): number of processes participating in the job. Defaults to 1.

    Returns:
        Dataloader:  The dataloader
    """
    cache_rate=0.0
    use_ddp = world_size > 1
    if isinstance(json_data_list, list):
        assert isinstance(data_base_dir, list)
        list_train = []
        for data_list, data_root, embedding_root in zip(json_data_list, data_base_dir, embedding_base_dir):
            json_data = load_filenames(data_list)
            list_train += json_data
    else:
        json_data = load_filenames(json_data_list)
        list_train = json_data


    common_transform = [
        Lambdad(keys="top_region_index", func=lambda x: torch.FloatTensor(x)),
        Lambdad(keys="bottom_region_index", func=lambda x: torch.FloatTensor(x)),
        Lambdad(keys="spacing", func=lambda x: torch.FloatTensor(x)),
        Lambdad(keys=["top_region_index", "bottom_region_index", "spacing"], func=lambda x: x * 1e2),
    ]

    train_transforms = Compose(common_transform)

    train_loader = None

    
    even_divisible = False
    dl_shuffle = False
    dl_workers = 8
    dl_pinmem = True

    if use_ddp:
        list_train = partition_dataset(
            data=list_train,
            shuffle=True,
            num_partitions=world_size,
            even_divisible=even_divisible,
        )[rank]
    train_ds = CacheDataset(data=list_train, transform=train_transforms, cache_rate=cache_rate, num_workers=8)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=dl_shuffle, num_workers=dl_workers, pin_memory=dl_pinmem)
    return train_loader

def prepare_dataloader_path_eval_userstudy(
    json_data_list: list,
    data_base_dir: list | str,
    embedding_base_dir: list | str,
    imkey: str,
    lblkey: str,
    is_val: bool,
    batch_size: int = 1,
    cache_rate: float = 0.0,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """
    Prepare dataloaders for evaluation (Includes base images)

    Args:
        json_data_list (list | str): the name of JSON files listing the data.
        data_base_dir (list | str): directory of files.
        imkey (str): image key in JSON
        lblkey (str): label key in JSON
        is_val (bool): whether to shuffle dataset (and selects number of workers)
        batch_size (int, optional): how many samples per batch to load . Defaults to 1.
        cache_rate (float, optional): percentage of cached data in total. Forces to 0.0.
        rank (int, optional): rank of the current process. Defaults to 0.
        world_size (int, optional): number of processes participating in the job. Defaults to 1.

    Returns:
        Dataloader:  The dataloader
    """
    cache_rate=0.0
    use_ddp = world_size > 1
    list_train = json_data_list

    # Don't need images just the other details
    common_transform = [
        Lambdad(keys="pirads", func=lambda x: torch.FloatTensor(x)),
        Lambdad(keys="spacing", func=lambda x: torch.FloatTensor(x)),
        Lambdad(keys=["pirads", "spacing"], func=lambda x: x * 1e2),
    ]

    train_transforms = Compose(common_transform)

    train_loader = None

    even_divisible = False
    dl_shuffle = False
    dl_workers = 8
    dl_pinmem = True

    if use_ddp:
        list_train = partition_dataset(
            data=list_train,
            shuffle=True,
            num_partitions=world_size,
            even_divisible=even_divisible,
        )[rank]
    train_ds = CacheDataset(data=list_train, transform=train_transforms, cache_rate=cache_rate, num_workers=8)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=dl_shuffle, num_workers=dl_workers, pin_memory=dl_pinmem)
    return train_loader

def prepare_dataloader_text_eval_userstudy(
    json_data_list: list,
    data_base_dir: list | str,
    embedding_base_dir: list | str,
    imkey: str,
    lblkey: str,
    is_val: bool,
    batch_size: int = 1,
    cache_rate: float = 0.0,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """
    Prepare dataloaders for evaluation (Includes base images)

    Args:
        json_data_list (list | str): the name of JSON files listing the data.
        data_base_dir (list | str): directory of files.
        imkey (str): image key in JSON
        lblkey (str): label key in JSON
        is_val (bool): whether to shuffle dataset (and selects number of workers)
        batch_size (int, optional): how many samples per batch to load . Defaults to 1.
        cache_rate (float, optional): percentage of cached data in total. Forces to 0.0.
        rank (int, optional): rank of the current process. Defaults to 0.
        world_size (int, optional): number of processes participating in the job. Defaults to 1.

    Returns:
        Dataloader:  The dataloader
    """
    cache_rate=0.0
    use_ddp = world_size > 1
    list_train = json_data_list


    common_transform = [
        LoadImaged(keys=['text']),
        Lambdad(keys="pirads", func=lambda x: torch.FloatTensor(x)),
        Lambdad(keys="spacing", func=lambda x: torch.FloatTensor(x)),
        Lambdad(keys=["pirads", "spacing"], func=lambda x: x * 1e2),
    ]

    train_transforms = Compose(common_transform)

    train_loader = None

    
    even_divisible = False
    dl_shuffle = False
    dl_workers = 8
    dl_pinmem = True

    if use_ddp:
        list_train = partition_dataset(
            data=list_train,
            shuffle=True,
            num_partitions=world_size,
            even_divisible=even_divisible,
        )[rank]
    train_ds = CacheDataset(data=list_train, transform=train_transforms, cache_rate=cache_rate, num_workers=8)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=dl_shuffle, num_workers=dl_workers, pin_memory=dl_pinmem)
    return train_loader

def prepare_dataloader_base_eval_userstudy(
    json_data_list: list,
    data_base_dir: list | str,
    embedding_base_dir: list | str,
    imkey: str,
    lblkey: str,
    is_val: bool,
    batch_size: int = 1,
    cache_rate: float = 0.0,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """
    Prepare dataloader for base MAISI DM eval (Includes base images)

    Args:
        json_data_list (list | str): the name of JSON files listing the data.
        data_base_dir (list | str): directory of files.
        imkey (str): image key in JSON
        lblkey (str): label key in JSON
        is_val (bool): whether to shuffle dataset (and selects number of workers)
        batch_size (int, optional): how many samples per batch to load . Defaults to 1.
        cache_rate (float, optional): percentage of cached data in total. Forces to 0.0.
        rank (int, optional): rank of the current process. Defaults to 0.
        world_size (int, optional): number of processes participating in the job. Defaults to 1.

    Returns:
        Dataloader:  The dataloader
    """
    cache_rate=0.0
    use_ddp = world_size > 1
    list_train = json_data_list


    common_transform = [
        Lambdad(keys="top_region_index", func=lambda x: torch.FloatTensor(x)),
        Lambdad(keys="bottom_region_index", func=lambda x: torch.FloatTensor(x)),
        Lambdad(keys="spacing", func=lambda x: torch.FloatTensor(x)),
        Lambdad(keys=["top_region_index", "bottom_region_index", "spacing"], func=lambda x: x * 1e2),
    ]

    train_transforms = Compose(common_transform)

    train_loader = None

    
    even_divisible = False
    dl_shuffle = False
    dl_workers = 8
    dl_pinmem = True

    if use_ddp:
        list_train = partition_dataset(
            data=list_train,
            shuffle=True,
            num_partitions=world_size,
            even_divisible=even_divisible,
        )[rank]
    train_ds = CacheDataset(data=list_train, transform=train_transforms, cache_rate=cache_rate, num_workers=8)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=dl_shuffle, num_workers=dl_workers, pin_memory=dl_pinmem)
    return train_loader

def run_inference_base(
    args: argparse.Namespace,
    device: torch.device,
    unet: torch.nn.Module,
    top_region_index_tensor: torch.Tensor,
    bottom_region_index_tensor: torch.Tensor,
    spacing_tensor: torch.Tensor,
    output_size: tuple,
    divisor: int,
    logger: logging.Logger,
) -> torch.Tensor:
    """
    Run the inference to generate synthetic images.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to run inference on.
        autoencoder (torch.nn.Module): Autoencoder model.
        unet (torch.nn.Module): UNet model.
        scale_factor (float): Scale factor for the model.
        top_region_index_tensor (torch.Tensor): Top region index tensor.
        bottom_region_index_tensor (torch.Tensor): Bottom region index tensor.
        spacing_tensor (torch.Tensor): Spacing tensor.
        output_size (tuple): Output size of the synthetic image.
        divisor (int): Divisor for downsample level.
        logger (logging.Logger): Logger for logging information.

    Returns:
        np.ndarray: Generated synthetic image data.
    """
    if args.diffusion_unet_inference['imkey'] == "multi":
        raise NotImplementedError("Multi will not work in base MAISI!")
    noise = torch.randn(
        (
            spacing_tensor.shape[0],
            args.latent_channels,
            output_size[0] // divisor,
            output_size[1] // divisor,
            output_size[2] // divisor,
        ),
        device=device,
    )
    logger.info(f"noise: {noise.device}, {noise.dtype}, {type(noise)}, {noise.shape}")

    image = noise
    noise_scheduler = define_instance(args, "noise_scheduler")
    noise_scheduler.set_timesteps(num_inference_steps=args.diffusion_unet_inference["num_inference_steps"])

    unet.eval()

    with torch.amp.autocast("cuda", enabled=True):
        # for t in tqdm(noise_scheduler.timesteps, ncols=110):
        for t in noise_scheduler.timesteps:
            timestep_preshape = [t for _ in range(spacing_tensor.shape[0])] #(t,)
            # print(f"Timestep preshape {timestep_preshape}")
            model_output = unet(
                x=image,
                timesteps=torch.Tensor(timestep_preshape).to(device), # torch.Tensor((t,)).to(device)
                top_region_index_tensor=top_region_index_tensor,
                bottom_region_index_tensor=bottom_region_index_tensor,
                spacing_tensor=spacing_tensor,
            )
            image, _ = noise_scheduler.step(model_output, t, image)

        return image

def run_inference_path(
    args: argparse.Namespace,
    device: torch.device,
    unet: torch.nn.Module,
    pirads_tensor: torch.Tensor,
    spacing_tensor: torch.Tensor | None,
    output_size: tuple,
    divisor: int,
    logger: logging.Logger,
) -> torch.Tensor:
    """
    Run the inference to generate synthetic images.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to run inference on.
        autoencoder (torch.nn.Module): Autoencoder model.
        unet (torch.nn.Module): UNet model.
        scale_factor (float): Scale factor for the model.
        pirads_tensor (torch.Tensor): PIRADS tensor.
        spacing_tensor (torch.Tensor): Spacing tensor.
        output_size (tuple): Output size of the synthetic image.
        divisor (int): Divisor for downsample level.
        logger (logging.Logger): Logger for logging information.

    Returns:
        np.ndarray: Generated synthetic image data.
    """
    # if args.diffusion_unet_inference['imkey'] == "multi":
    #     raise NotImplementedError("Multi not yet implemented!") # Todo: implement this
    noise = torch.randn(
        (
            pirads_tensor.shape[0],
            args.latent_channels,
            output_size[0] // divisor,
            output_size[1] // divisor,
            output_size[2] // divisor,
        ),
        device=device,
    )
    logger.info(f"noise: {noise.device}, {noise.dtype}, {type(noise)}, {noise.shape}")

    image = noise
    noise_scheduler = define_instance(args, "noise_scheduler")
    noise_scheduler.set_timesteps(num_inference_steps=args.diffusion_unet_inference["num_inference_steps"])

    unet.eval()

    with torch.amp.autocast("cuda", enabled=True):
        # for t in tqdm(noise_scheduler.timesteps, ncols=110):
        for t in noise_scheduler.timesteps:
            timestep_preshape = [t for _ in range(pirads_tensor.shape[0])]
            model_output = unet(
                x=image,
                timesteps=torch.Tensor(timestep_preshape).to(device),
                pirads=pirads_tensor,
                spacing_tensor=spacing_tensor,
            )
            image, _ = noise_scheduler.step(model_output, t, image)

        if args.diffusion_unet_inference['imkey'] == "multi":
            # axt2 will be first 4 channels, then b1600, then adc
            axt2_latent = image[:,:4,:,:,:] # 0-3
            highb_latent = image[:,4:8,:,:,:] # 4-7
            adc_latent = image[:,8:,:,:,:] # 8-11

            return axt2_latent, highb_latent, adc_latent
        else:
            return image

def run_inference_text(
    args: argparse.Namespace,
    device: torch.device,
    unet: torch.nn.Module,
    pirads_tensor: torch.Tensor,
    spacing_tensor: torch.Tensor | None,
    text_tensor: torch.Tensor,
    output_size: tuple,
    divisor: int,
    logger: logging.Logger,
) -> torch.Tensor:
    """
    Run the inference to generate synthetic images.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to run inference on.
        autoencoder (torch.nn.Module): Autoencoder model.
        unet (torch.nn.Module): UNet model.
        scale_factor (float): Scale factor for the model.
        pirads_tensor (torch.Tensor): PIRADS tensor.
        spacing_tensor (torch.Tensor): Spacing tensor.
        text_tensor (torch.Tensor): Text tensor.
        output_size (tuple): Output size of the synthetic image.
        divisor (int): Divisor for downsample level.
        logger (logging.Logger): Logger for logging information.

    Returns:
        np.ndarray: Generated synthetic image data.
    """
    # if args.diffusion_unet_inference['imkey'] == "multi":
    #     raise NotImplementedError("Multi not yet implemented!") # Todo: implement this
    noise = torch.randn(
        (
            pirads_tensor.shape[0],
            args.latent_channels,
            output_size[0] // divisor,
            output_size[1] // divisor,
            output_size[2] // divisor,
        ),
        device=device,
    )
    logger.info(f"noise: {noise.device}, {noise.dtype}, {type(noise)}, {noise.shape}")

    image = noise
    noise_scheduler = define_instance(args, "noise_scheduler")
    noise_scheduler.set_timesteps(num_inference_steps=args.diffusion_unet_inference["num_inference_steps"])

    unet.eval()

    with torch.amp.autocast("cuda", enabled=True):
        # for t in tqdm(noise_scheduler.timesteps, ncols=110):
        for t in noise_scheduler.timesteps:
            timestep_preshape = [t for _ in range(pirads_tensor.shape[0])] #(t,)
            # print(f"Timestep preshape {timestep_preshape}")
            model_output = unet(
                x=image,
                timesteps=torch.Tensor(timestep_preshape).to(device), # torch.Tensor((t,)).to(device)
                pirads=pirads_tensor,
                spacing_tensor=spacing_tensor,
                text_encoding=text_tensor,
            )
            image, _ = noise_scheduler.step(model_output, t, image)

        if args.diffusion_unet_inference['imkey'] == "multi":
            # axt2 will be first 4 channels, then b1600, then adc
            axt2_latent = image[:,:4,:,:,:] # 0-3
            highb_latent = image[:,4:8,:,:,:] # 4-7
            adc_latent = image[:,8:,:,:,:] # 8-11

            return axt2_latent, highb_latent, adc_latent
        else:
            return image
   
def run_inference_textclass(
    args: argparse.Namespace,
    device: torch.device,
    unet: torch.nn.Module,
    spacing_tensor: torch.Tensor | None,
    text_tensor: torch.Tensor,
    output_size: tuple,
    divisor: int,
    logger: logging.Logger,
) -> torch.Tensor:
    """
    Run the inference to generate synthetic images.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to run inference on.
        autoencoder (torch.nn.Module): Autoencoder model.
        unet (torch.nn.Module): UNet model.
        scale_factor (float): Scale factor for the model.
        spacing_tensor (torch.Tensor): Spacing tensor.
        text_tensor (torch.Tensor): Text tensor.
        output_size (tuple): Output size of the synthetic image.
        divisor (int): Divisor for downsample level.
        logger (logging.Logger): Logger for logging information.

    Returns:
        np.ndarray: Generated synthetic image data.
    """
    # if args.diffusion_unet_inference['imkey'] == "multi":
    #     raise NotImplementedError("Multi not yet implemented!") # Todo: implement this
    noise = torch.randn(
        (
            spacing_tensor.shape[0],
            args.latent_channels,
            output_size[0] // divisor,
            output_size[1] // divisor,
            output_size[2] // divisor,
        ),
        device=device,
    )
    logger.info(f"noise: {noise.device}, {noise.dtype}, {type(noise)}, {noise.shape}")

    image = noise
    noise_scheduler = define_instance(args, "noise_scheduler")
    noise_scheduler.set_timesteps(num_inference_steps=args.diffusion_unet_inference["num_inference_steps"])

    unet.eval()

    with torch.amp.autocast("cuda", enabled=True):
        # for t in tqdm(noise_scheduler.timesteps, ncols=110):
        for t in noise_scheduler.timesteps:
            timestep_preshape = [t for _ in range(spacing_tensor.shape[0])] #(t,)
            # print(f"Timestep preshape {timestep_preshape}")
            model_output, class_pred = unet(
                x=image,
                timesteps=torch.Tensor(timestep_preshape).to(device), # torch.Tensor((t,)).to(device)
                spacing_tensor=spacing_tensor,
                text_encoding=text_tensor,
            )
            image, _ = noise_scheduler.step(model_output, t, image)

        if args.diffusion_unet_inference['imkey'] == "multi":
            # axt2 will be first 4 channels, then b1600, then adc
            axt2_latent = image[:,:4,:,:,:] # 0-3
            highb_latent = image[:,4:8,:,:,:] # 4-7
            adc_latent = image[:,8:,:,:,:] # 8-11

            return axt2_latent, highb_latent, adc_latent, class_pred
        else:
            return image, class_pred
        

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

def make_img_from_latents(image,autoencoder,scale_factor,device, multi=False):
    autoencoder.eval()
    if multi: #args.diffusion_unet_inference['imkey'] == "multi":
        recon_model = ReconModelRaw(autoencoder=autoencoder).to(device)
        axt2_latent, highb_latent, adc_latent = image
        with torch.amp.autocast("cuda", enabled=True):
            axt2 = recon_model(axt2_latent, scale_factor=scale_factor[0])
            hb = recon_model(highb_latent, scale_factor=scale_factor[1])
            adc = recon_model(adc_latent, scale_factor=scale_factor[2])

        return axt2, hb, adc
    else:
        recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
        with torch.amp.autocast("cuda", enabled=True):
            image = recon_model(image)

        return image


def main_flstudy(args,tvt, output_img_dir, spreadsheet_path):
    logger = logging.getLogger("maisi.synthetic_datagen")
    local_rank, world_size, device = initialize_distributed(num_gpus=args.num_gpus)

    with open(args.environment_file, "r") as env_file:
        env_dict = json.load(env_file)
    with open(args.config_file, "r") as config_file:
        config_dict = json.load(config_file)
    with open(args.training_config, "r") as training_config_file:
        training_config_dict = json.load(training_config_file)

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)
    for k, v in training_config_dict.items():
        setattr(args, k, v)
        
    imkey=args.diffusion_unet_inference["imkey"]
    random_seed = set_random_seed(
        args.diffusion_unet_inference["random_seed"] + local_rank
        if args.diffusion_unet_inference["random_seed"]
        else None
    )
    logger.info(f"Using {device} of {world_size} with random seed: {random_seed}")

    synth_outfolder = output_img_dir
    if imkey == 'multi':
        synth_outfolder += "_multi"

    if not os.path.exists(synth_outfolder):
        os.makedirs(synth_outfolder)

    # Load the Excel file
    df = pd.read_excel(spreadsheet_path)

    if tvt == 'train':
        starter_json = args.train_json
    elif tvt == 'test':
        starter_json = args.test_json

    loaded_json = load_filenames(starter_json)

    cur_json = []

    for d in loaded_json:
        if d['dataset'] == "Institute1": # Only use the Institute1 cases
            cur_json.append(d)

    if local_rank==0:
        print(f"Dataset size: {len(cur_json)}")

    if 'use_text_conditioning' in args and args.use_text_conditioning:
        val_loader = prepare_dataloader_text_eval_userstudy(
            json_data_list=cur_json,
            data_base_dir=args.data_base_dir,
            embedding_base_dir=args.embedding_base_dir,
            imkey=imkey,
            lblkey='label', # won't actually use this but good to have in case / can't hurt for now
            is_val=True,
            rank=local_rank,
            world_size=world_size,
            batch_size=args.batch_size,
            cache_rate=0.0,
        )
        if 'use_text_class_pred' in args and args.use_text_class_pred:
            class_preds = []
            class_reals = []
    else:
        if args.base_maisi:
            prepare_maisi_diffusion_json_dataloader = prepare_dataloader_base_eval_userstudy
        else:
            prepare_maisi_diffusion_json_dataloader = prepare_dataloader_path_eval_userstudy

        val_loader = prepare_maisi_diffusion_json_dataloader(
            json_data_list=cur_json,
            data_base_dir=args.data_base_dir,
            embedding_base_dir=args.embedding_base_dir,
            imkey=imkey,
            lblkey='label', # won't actually use this but good to have in case / can't hurt for now
            is_val=True,
            rank=local_rank,
            world_size=world_size,
            batch_size=args.batch_size,
            cache_rate=0.0,
        )

    autoencoder, unet, scale_factor = load_models(args, device, logger)

    num_downsample_level = max(
        1,
        (
            len(args.diffusion_unet_def["num_channels"])
            if isinstance(args.diffusion_unet_def["num_channels"], list)
            else len(args.diffusion_unet_def["attention_levels"])
        ),
    )
    divisor = 2 ** (num_downsample_level - 2)
    logger.info(f"num_downsample_level -> {num_downsample_level}, divisor -> {divisor}.")

    output_size = tuple(args.diffusion_unet_inference["dim"])

    df_dict = {"folder":[],"PIRADS_real":[],"PIRADS_synth":[],"Label":[]} # One entry per pair of real/synthetic images

    for idx, batch in tqdm(enumerate(val_loader)):
        if args.base_maisi:
            top_region_index_tensor = batch["top_region_index"].to(device)
            bottom_region_index_tensor = batch["bottom_region_index"].to(device)
            spacing_tensor = batch["spacing"].to(device)
        elif 'use_text_conditioning' in args and args.use_text_conditioning:
            pirads_tensor = batch["pirads"].to(device)
            spacing_tensor = batch["spacing"].to(device)
            text_tensor = batch["text"].to(device)
        else:
            pirads_tensor = batch["pirads"].to(device)
            spacing_tensor = batch["spacing"].to(device)


        if args.base_maisi:
            synthetic_images = run_inference_base(
                args,
                device,
                unet,
                top_region_index_tensor,
                bottom_region_index_tensor,
                spacing_tensor,
                output_size,
                divisor,
                logger,
            )
        elif 'use_text_conditioning' in args and args.use_text_conditioning:
            if 'use_text_class_pred' in args and args.use_text_class_pred:
                synth_gen = run_inference_textclass(
                    args,
                    device,
                    unet,
                    spacing_tensor,
                    text_tensor,
                    output_size,
                    divisor,
                    logger,
                )
                if len(synth_gen) == 2:
                    synthetic_images, class_pred = synth_gen
                elif len(synth_gen) == 4:
                    synthetic_images, class_pred = synth_gen[:3], synth_gen[-1]
                else:
                    raise NotImplementedError(f"This should not happen! Got synthgen length of {len(synth_gen)}")

                class_pred_torch = torch.argmax(class_pred, dim=1).cpu().detach().numpy()
                class_real_torch = torch.argmax(pirads_tensor, dim=1).cpu().detach().numpy()
                for case_i in range(class_pred.shape[0]):
                    class_preds.append(class_pred_torch[case_i])
                    class_reals.append(class_real_torch[case_i])
            else:
                synthetic_images = run_inference_text(
                    args,
                    device,
                    unet,
                    pirads_tensor,
                    spacing_tensor, # Spacing tensor
                    text_tensor,
                    output_size,
                    divisor,
                    logger,
                )
        else:
            synthetic_images = run_inference_path(
                args,
                device,
                unet,
                pirads_tensor,
                spacing_tensor, # Spacing tensor
                output_size,
                divisor,
                logger,
            )

        if imkey == 'multi':
            axt2_s_im, highb_s_im, adc_s_im = synthetic_images
            axt2_s_chunks = torch.chunk(axt2_s_im, args.autoencoder_batch_size, dim=0)
            highb_s_chunks = torch.chunk(highb_s_im, args.autoencoder_batch_size, dim=0)
            adc_s_chunks = torch.chunk(adc_s_im, args.autoencoder_batch_size, dim=0)
            imchunks=zip(axt2_s_chunks,highb_s_chunks,adc_s_chunks)
        else:
            imchunks = torch.chunk(synthetic_images, args.autoencoder_batch_size, dim=0)

        if imkey == 'multi':
            i=0

            for imchunk in imchunks:
                minibatch_size = imchunk[0].shape[0]

                axt2, hb, adc = make_img_from_latents(imchunk,autoencoder,scale_factor,device,multi=True)

                for im_i in range(axt2.shape[0]): # Save synthetic images and put info in dataframe
                    cur_folder = batch["folder"][im_i+i]
                    df_dict["folder"].append(cur_folder)
                    # Find the real PIRADS label from dataframe where folder is match
                    real_pirads = df.loc[df['folder'] == cur_folder]['PosPIRADS'].values[0]
                    # if value is NA then use PosISUP
                    if pd.isna(real_pirads):
                        real_pirads = df.loc[df['folder'] == cur_folder]['PosISUP'].values[0]
                    real_label = df.loc[df['folder'] == cur_folder]['Label'].values[0]
                    if 'use_text_conditioning' in args and args.use_text_conditioning and 'use_text_class_pred' in args and args.use_text_class_pred:
                        df_dict["PIRADS_real"].append(real_pirads)
                        df_dict["PIRADS_synth"].append(class_pred_torch[im_i+i])
                    else:
                        df_dict["PIRADS_real"].append(real_pirads)
                        df_dict['PIRADS_synth'].append('')
                    df_dict["Label"].append(real_label)

                    outfolder_cur = os.sep.join([synth_outfolder,cur_folder])
                    if not os.path.exists(outfolder_cur):
                        os.makedirs(outfolder_cur)

                    for filename, im in [('axt2.nii.gz',axt2),('b1600.nii.gz',hb),('adc.nii.gz',adc)]:

                        im_spacing = spacing_tensor[im_i+i*args.autoencoder_batch_size].cpu().detach().numpy()
                        im_spacing/=100.0
                        im_spacing = im_spacing.tolist()

                        im_to_resize = im[im_i].squeeze().cpu().detach().numpy()
                        im_to_resize = im_to_resize.astype(np.float32)
                        
                        im_to_resize = np.transpose(im_to_resize,(2,1,0))
                        im_to_resize = np.flip(im_to_resize,axis=1)
                        im_to_resize = np.flip(im_to_resize,axis=2)

                        outpath = os.sep.join([outfolder_cur,filename])

                        im_mean = np.mean(im_to_resize)
                        im_std = np.std(im_to_resize)
                        if im_std == 0.0:
                            im_std = 1.0 # Prevent division by 0
                        out_im = (im_to_resize - im_mean)/im_std

                        out_im = sitk.GetImageFromArray(out_im)
                        out_im.SetSpacing(im_spacing)
                        
                        out_im = resample_volume(out_im) # Image size should already be appropriate + zero-padded

                        sitk.WriteImage(out_im,outpath)

                i += minibatch_size
        else:
            i=0

            for imchunk in imchunks:
                minibatch_size = imchunk.shape[0]

                minibatch_images = make_img_from_latents(imchunk,autoencoder,scale_factor,device,multi=False)

                for im_i in range(minibatch_images.shape[0]): # Save synthetic images and put info in dataframe
                    cur_folder = batch["folder"][im_i+i]
                    df_dict["folder"].append(cur_folder)
                    # Find the real PIRADS label from dataframe where folder is match
                    real_pirads = df.loc[df['folder'] == cur_folder]['PosPIRADS'].values[0]
                    # if value is NA then use PosISUP
                    if pd.isna(real_pirads):
                        real_pirads = df.loc[df['folder'] == cur_folder]['PosISUP'].values[0]
                    real_label = df.loc[df['folder'] == cur_folder]['Label'].values[0]
                    if 'use_text_conditioning' in args and args.use_text_conditioning and 'use_text_class_pred' in args and args.use_text_class_pred:
                        df_dict["PIRADS_real"].append(real_pirads)
                        df_dict["PIRADS_synth"].append(class_pred_torch[im_i+i])
                    else:
                        df_dict["PIRADS_real"].append(real_pirads)
                        df_dict['PIRADS_synth'].append('')
                    df_dict["Label"].append(real_label)

                    # Resample and save the synthetic image to the synthetic directory
                    outfolder_cur = os.sep.join([synth_outfolder,cur_folder])
                    if not os.path.exists(outfolder_cur):
                        os.makedirs(outfolder_cur)

                    filename = f"{imkey}.nii.gz"

                    im_spacing = spacing_tensor[im_i+i*args.autoencoder_batch_size].cpu().detach().numpy()
                    im_spacing/=100.0
                    im_spacing = im_spacing.tolist()

                    im_to_resize = minibatch_images[im_i].squeeze().cpu().detach().numpy()
                    im_to_resize = im_to_resize.astype(np.float32)
                    
                    im_to_resize = np.transpose(im_to_resize,(2,1,0))
                    im_to_resize = np.flip(im_to_resize,axis=1)
                    im_to_resize = np.flip(im_to_resize,axis=2)

                    outpath = os.sep.join([outfolder_cur,filename])
                    # z-score normalize

                    im_mean = np.mean(im_to_resize)
                    im_std = np.std(im_to_resize)
                    if im_std == 0.0:
                        im_std = 1.0 # Prevent division by 0
                    out_im = (im_to_resize - im_mean)/im_std

                    out_im = sitk.GetImageFromArray(out_im)
                    out_im.SetSpacing(im_spacing)
                    
                    out_im = resample_volume(out_im) # Image size should already be appropriate + zero-padded

                    sitk.WriteImage(out_im,outpath)

                i += minibatch_size

    df_out = pd.DataFrame(df_dict)
    df_outpath = f"{synth_outfolder}/synthsheet_tfproxy"
    if imkey == 'multi':
        df_outpath += "_multi"
    df_outpath += f"_{local_rank}_{tvt}.xlsx"
    df_out.to_excel(df_outpath)

    if args.num_gpus>1:
        dist.destroy_process_group()

if __name__ == "__main__":
    # raise ValueError("We shouldn't be using this much")
    parser = argparse.ArgumentParser(description="Diffusion Model Synthetic Training Data Creation")
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser.add_argument(
        "-e",
        "--environment-file",
        "--env_config",
        default="./configs/environment_maisi_controlnet_train.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        "--model_def",
        default="./configs/config_maisi.json",
        help="config json file that stores network hyper-parameters",
    )
    parser.add_argument(
        "-t",
        "--training-config",
        "--model_config",
        default="./configs/config_maisi_controlnet_train.json",
        help="config json file that stores training hyper-parameters",
    )
    parser.add_argument(
        "-r",
        "--modelname_override",
        default="",
        help="Override the model name",
    )
    parser.add_argument("-g", "--num_gpus", default=2, type=int, help="number of gpus per node")
    parser.add_argument("-b", "--batch_size", default=2, type=int, help="LDM batch size")
    parser.add_argument("-a", "--autoencoder_batch_size", default=1, type=int, help="autoencoder batch size")
    parser.add_argument("-o", "--output_dir", default="./synthdata", type=str, help="Output directory for synthetic images")
    parser.add_argument("-s", "--spreadsheet_path", default="./synthdata/synthsheet.xlsx", type=str, help="Path to the spreadsheet for additional data")
    args = parser.parse_args()


    if args.autoencoder_batch_size > 1:
        raise ValueError("Autoencoder not programmed to scale images properly with batch size greater than 1")

    output_img_dir = args.output_dir
    spreadsheet_path = args.spreadsheet_path
    with torch.no_grad():
        for tvt in ['train','test']:
            main_flstudy(args,tvt, output_img_dir, spreadsheet_path)
