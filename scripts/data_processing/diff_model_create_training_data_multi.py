# Modified from MONAI's diffusion model data processing script

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist

import monai
from monai.transforms import Compose
from monai.utils import set_determinism

from ..diff_model_setting import initialize_distributed, load_config, setup_logging
from ..utils import define_instance


from monai.transforms.transform import MapTransform, Transform
from monai.utils.type_conversion import convert_to_tensor
from monai.data.meta_obj import get_track_meta
from tqdm import tqdm

# Set the random seed for reproducibility
set_determinism(seed=0)


def create_transforms() -> Compose:
    """
    Create a set of MONAI transforms for preprocessing.

    Args:
        dim (tuple, optional): New dimensions for resizing. Defaults to None.

    Returns:
        Compose: Composed MONAI transforms.
    """
    imkeys = ['axt2','b1600','adc']
    non_CT_keys = ['axt2','b1600']
    CT_keys = ['adc']
    return Compose(
        [
            monai.transforms.LoadImaged(keys=imkeys),
            monai.transforms.EnsureChannelFirstd(keys=imkeys),
            monai.transforms.Orientationd(keys=imkeys, axcodes="RAS"),
            monai.transforms.EnsureTyped(keys=imkeys, dtype=torch.float32),
            monai.transforms.ScaleIntensityRangePercentilesd(keys=non_CT_keys, lower=0, upper=99.5, b_min=0, b_max=1, clip=True), # Justification: MAISI uses this for MRI data
            monai.transforms.ScaleIntensityRanged(keys=CT_keys,a_min=0,a_max=3000.0, b_min=0, b_max=1, clip=True),
        ]
    )



def round_number(number: int, base_number: int = 128) -> int:
    """
    Round the number to the nearest multiple of the base number, with a minimum value of the base number.

    Args:
        number (int): Number to be rounded.
        base_number (int): Number to be common divisor.

    Returns:
        int: Rounded number.
    """
    new_number = max(round(float(number) / float(base_number)), 1.0) * float(base_number)
    return int(new_number)


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


def process_file(
    filepath: str,
    args: argparse.Namespace,
    autoencoder: torch.nn.Module,
    device: torch.device,
    transforms: Compose,
    logger: logging.Logger,
) -> None:
    """
    Process a single file to create training data.

    Args:
        filepath (str): Path to the file to be processed.
        args (argparse.Namespace): Configuration arguments.
        autoencoder (torch.nn.Module): Autoencoder model.
        device (torch.device): Device to process the file on.
        transforms (Compose): Transforms for images
        logger (logging.Logger): Logger for logging information.
    """
    axt2_out_filename = os.path.join(args.embedding_base_dir, filepath['axt2'].removesuffix('.nii.gz')+"_emb.nii.gz")
    highb_out_filename = os.path.join(args.embedding_base_dir, filepath['b1600'].removesuffix('.nii.gz')+"_emb.nii.gz")
    adc_out_filename = os.path.join(args.embedding_base_dir, filepath['adc'].removesuffix('.nii.gz')+"_emb.nii.gz")
    multi_out_filename = os.path.join(args.embedding_base_dir, filepath['axt2'].removesuffix('axt2.nii.gz')+"multi_emb.nii.gz")

    test_data = {"axt2": os.path.join(args.data_base_dir, filepath['axt2']),\
                    "b1600": os.path.join(args.data_base_dir, filepath['b1600']),\
                    "adc": os.path.join(args.data_base_dir, filepath['adc']),}

    transformed_data = transforms(test_data)
    axt2 = transformed_data['axt2'].to(device)
    highb = transformed_data['b1600'].to(device)
    adc = transformed_data['adc'].to(device)

    new_affine = axt2.meta["affine"].numpy()
    
    logger.info(f"axt2_shape: {axt2.shape}, highb_shape: {highb.shape}, adc_shape: {adc.shape}")

    try:
        out_path = Path(axt2_out_filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"axt2_out_filename: {axt2_out_filename}")

        with torch.amp.autocast("cuda"):
            pt_axt2 = axt2.unsqueeze(0)
            pt_highb = highb.unsqueeze(0)
            pt_adc = adc.unsqueeze(0)
            z_axt2 = autoencoder.encode_stage_2_inputs(pt_axt2)
            z_highb = autoencoder.encode_stage_2_inputs(pt_highb)
            z_adc = autoencoder.encode_stage_2_inputs(pt_adc)
            logger.info(f"z_axt2: {z_axt2.size()}, {z_axt2.dtype}")
            z_multi = torch.concat([z_axt2,z_highb,z_adc],dim=1) # multi embedding
            logger.info(f"z_multi: {z_multi.size()}, {z_multi.dtype}")

            out_axt2 = z_axt2.squeeze().cpu().detach().numpy().transpose(1, 2, 3, 0)
            out_highb = z_highb.squeeze().cpu().detach().numpy().transpose(1, 2, 3, 0)
            out_adc = z_adc.squeeze().cpu().detach().numpy().transpose(1, 2, 3, 0)
            out_multi = z_multi.squeeze().cpu().detach().numpy().transpose(1, 2, 3, 0)
            out_axt2_img = nib.Nifti1Image(np.float32(out_axt2), affine=new_affine)
            out_highb_img = nib.Nifti1Image(np.float32(out_highb), affine=new_affine)
            out_adc_img = nib.Nifti1Image(np.float32(out_adc), affine=new_affine)
            out_multi_img = nib.Nifti1Image(np.float32(out_multi), affine=new_affine)
            nib.save(out_axt2_img, axt2_out_filename)
            nib.save(out_highb_img, highb_out_filename)
            nib.save(out_adc_img, adc_out_filename)
            nib.save(out_multi_img, multi_out_filename)
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")


@torch.inference_mode()
def diff_model_create_training_data(
    env_config_path: str, model_config_path: str, model_def_path: str, num_gpus: int
) -> None:
    """
    Create training data for the diffusion model.

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
        model_def_path (str): Path to the model definition file.
    """
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, world_size, device = initialize_distributed(num_gpus=num_gpus)
    logger = setup_logging("creating training data")
    logger.info(f"Using device {device}")

    autoencoder = define_instance(args, "autoencoder_def").to(device)
    try:
        checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=True)
        autoencoder.load_state_dict(checkpoint_autoencoder)
    except Exception:
        logger.error("The trained_autoencoder_path does not exist!")

    Path(args.embedding_base_dir).mkdir(parents=True, exist_ok=True)

    filenames_raw = []
    filenames_raw.extend(load_filenames(args.train_json))
    filenames_raw.extend(load_filenames(args.test_json))

    filenames_all = []

    # Prune anything except Institute1
    for i in range(len(filenames_raw)):
        file_data = filenames_raw[i]
        if file_data["dataset"] == "Institute1":
            filenames_all.append(file_data)


    # logger.info(f"filenames_raw: {filenames_raw}")
    logger.info(f"Lengths filenames_all: {len(filenames_all)}")

    transforms = create_transforms()

    for _iter in tqdm(range(len(filenames_all))):
        if _iter % world_size != local_rank:
            continue

        data_dict = filenames_all[_iter]

        process_file(data_dict, args, autoencoder, device, transforms, logger)

    if dist.is_initialized():
        dist.destroy_process_group()

    print("Complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training Data Creation")
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
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for distributed training")

    args = parser.parse_args()
    diff_model_create_training_data(args.env_config, args.model_config, args.model_def, args.num_gpus)
