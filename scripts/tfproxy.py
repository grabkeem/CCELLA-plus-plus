# 2025-MAY-05
# Written using the MONAI workflows for code efficiency
# Majority of code borrowed from flproxy.py based on flproxy_resnet.py
# Restricted to single GPU unfortunately
# Ref: https://github.com/Project-MONAI/tutorials/blob/main/modules/cross_validation_models_ensemble.ipynb

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from ignite.handlers.tensorboard_logger import * #TensorboardLogger
from ignite.engine import Events
from tqdm import tqdm

from ignite.metrics import Accuracy, Precision, Recall, AveragePrecision, Loss

from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

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
from monai.networks.nets import DenseNet121, EfficientNetBN

from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .utils import define_instance
import numpy as np
import pandas as pd

import pandas as pd

from glob import glob

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import monai
from monai.data import DataLoader, Dataset, DistributedSampler, CacheDataset, ThreadDataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRangePercentilesd,
    RandFlipd,
    RandRotated,
    RandAffined,
    RandZoomd,
    Activations,
    AsDiscrete,
    CropForegroundd,
    SpatialPadd,
    CenterSpatialCropd,
    EnsureTyped,
    Activationsd,
    AsDiscreted,
    Lambdad,
    CopyItemsd,
    ConcatItemsd,
    DeleteItemsd,
    ClipIntensityPercentilesd,
    NormalizeIntensityd,
)
from torch.utils.tensorboard import SummaryWriter

from monai.apps import CrossValidation
from monai.config import print_config
from monai.engines import EnsembleEvaluator, SupervisedEvaluator, SupervisedTrainer
from monai.handlers import ROCAUC, StatsHandler, ValidationHandler, from_engine, EarlyStopHandler, TensorBoardHandler, ConfusionMatrix, TensorBoardStatsHandler, CheckpointSaver, CheckpointLoader, LrScheduleHandler, MetricsSaver, ClassificationSaver
from monai.inferers import SimpleInferer
from monai.utils import set_determinism


def lower_loss_better(current_metric: float, prev_best: float) -> bool:
    """
    The default function to compare metric values between current metric and previous best metric.

    Args:
        current_metric: metric value of current round computation.
        prev_best: the best metric value of previous rounds to compare with.

    """
    return current_metric < prev_best

def prep(args,train_flag=False,val_flag=False):
    if not train_flag and not val_flag:
        raise ValueError("Must specify train or val")
    args.exp_name = f"tfproxy_en"+args.model_type #_r{int(args.real_data)}_s{int(args.synth_data)}_l{int(args.generated_label)}
    if args.inst1_data:
        args.exp_name += "_u"
        if args.real_data:
            args.exp_name += "r"
        elif args.synth_data:
            args.exp_name += "s"
            if args.generated_label:
                args.exp_name += "_l1"
            else:
                args.exp_name += "_l0"
    if args.inst2_data:
        if args.finetune:
            args.pretrain_name = args.exp_name
            args.exp_name += "_pf"
        else:
            args.exp_name += "_p"

    if args.imkey == "multi":
        args.exp_name += "_multi"
        if args.finetune:
            args.pretrain_name += "_multi"

    if args.add_to_modelname:
        args.exp_name += f"_{args.add_to_modelname}"

    if args.count != 0:
        args.exp_name += f"_{args.count}"
        if args.finetune:
            args.pretrain_name += f"_{args.count}" # Finetune should build off the previous count

    exp_path_starter = './models/exp3'

    if not os.path.exists(exp_path_starter):
        os.makedirs(exp_path_starter)
    print(f"Experiment name: {args.exp_name}")

    set_determinism(seed=args.seed+args.count)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    device = torch.device("cuda:0")

    df_path = f"{args.output_dir}/synthsheet_tfproxy"
    if args.imkey == "multi":
        df_path += "_multi"
    df_path += ".xlsx"
    df = pd.read_excel(df_path)

    train_df = df[df["Label"] == "train"]
    val_df = df[df["Label"] == "val"]
    test_df = df[df["Label"] == "test"]

    real_folder = args.realdata_path
    synth_folder = args.synthdata_path
    if args.imkey == "multi":
        synth_folder += "_multi"

    train_files = []

    if args.imkey == "multi": # Files, dataloader, and input channels transforms will be different
        for idx, row in train_df.iterrows():
            folder = row["folder"]
            is_inst1 = row["dataset"] == "Institute1"
            if args.inst2_data and not is_inst1: # If inst2 flag set and data is inst2
                axt2_path = os.path.join(real_folder, folder,'axt2.nii.gz')
                highb_path = os.path.join(real_folder, folder,'b1600.nii.gz')
                adc_path = os.path.join(real_folder, folder,'adc.nii.gz')
                train_files.append({"axt2": axt2_path,
                                    "highb": highb_path,
                                    "adc": adc_path,
                                    "label": row["ISUP_real"]})
            elif args.inst1_data and is_inst1 and (not args.inst2_data and not args.finetune): # If data is Institute1 and Institute1 flag is set (and not because we're fine-tuning on inst2)
                if args.real_data:
                    axt2_path = os.path.join(real_folder, folder,'axt2.nii.gz')
                    highb_path = os.path.join(real_folder, folder,'b1600.nii.gz')
                    adc_path = os.path.join(real_folder, folder,'adc.nii.gz')
                    train_files.append({"axt2": axt2_path,
                                        "highb": highb_path,
                                        "adc": adc_path,
                                        "label": row["ISUP_real"]})
                elif args.synth_data:
                    axt2_path = os.path.join(synth_folder, folder,'axt2.nii.gz')
                    highb_path = os.path.join(synth_folder, folder,'b1600.nii.gz')
                    adc_path = os.path.join(synth_folder, folder,'adc.nii.gz')
                    if args.generated_label:
                        train_files.append({"axt2": axt2_path,
                                            "highb": highb_path,
                                            "adc": adc_path,
                                            "label": row["PIRADS_synth"]}) # Synthetic data with synthetic label
                    else:
                        train_files.append({"axt2": axt2_path,
                                            "highb": highb_path,
                                            "adc": adc_path,
                                            "label": row["ISUP_real"]})
            else: # inst2 flag with Institute1 data or we're not Institute1 flag/data/not-finetune then skip
                continue
    else:
        for idx, row in train_df.iterrows():
            folder = row["folder"]
            is_inst1 = row["dataset"] == "Institute1"
            if args.inst2_data and not is_inst1: # If inst2 flag set and data is inst2
                img_path = os.path.join(real_folder, folder,f'{args.imkey}.nii.gz')
                train_files.append({"image": img_path, "label": row["ISUP_real"]})
            elif args.inst1_data and is_inst1 and (not args.inst2_data and not args.finetune): # If data is Institute1 and Institute1 flag is set (and not because we're fine-tuning on inst2)
                if args.real_data:
                    img_path = os.path.join(real_folder, folder,f'{args.imkey}.nii.gz')
                    train_files.append({"image": img_path, "label": row["ISUP_real"]})
                elif args.synth_data:
                    img_path = os.path.join(synth_folder, folder,f'{args.imkey}.nii.gz')
                    if args.generated_label:
                        train_files.append({"image": img_path, "label": row["PIRADS_synth"]}) # Synthetic data with synthetic label
                    else:
                        train_files.append({"image": img_path, "label": row["ISUP_real"]})
            else: # inst2 flag with Institute1 data or we're not Institute1 flag/data/not-finetune then skip
                continue

    val_files = []

    # No longer using PIRADS_real but ISUP_real unless generated in which case PIRADS_synth
    if args.imkey == "multi": # Files, dataloader, and input channels transforms will be different
        for idx, row in val_df.iterrows():
            folder = row["folder"]
            is_inst1 = row["dataset"] == "Institute1"
            if args.inst2_data and not is_inst1: # If inst2 flag set and data is inst2
                axt2_path = os.path.join(real_folder, folder,'axt2.nii.gz')
                highb_path = os.path.join(real_folder, folder,'b1600.nii.gz')
                adc_path = os.path.join(real_folder, folder,'adc.nii.gz')
                val_files.append({"axt2": axt2_path,
                                    "highb": highb_path,
                                    "adc": adc_path,
                                    "label": row["ISUP_real"]})
            elif args.inst1_data and is_inst1 and (not args.inst2_data and not args.finetune): # If data is Institute1 and Institute1 flag is set (and not because we're fine-tuning on inst2)
                if args.real_data:
                    axt2_path = os.path.join(real_folder, folder,'axt2.nii.gz')
                    highb_path = os.path.join(real_folder, folder,'b1600.nii.gz')
                    adc_path = os.path.join(real_folder, folder,'adc.nii.gz')
                    val_files.append({"axt2": axt2_path,
                                        "highb": highb_path,
                                        "adc": adc_path,
                                        "label": row["ISUP_real"]})
                elif args.synth_data:
                    axt2_path = os.path.join(synth_folder, folder,'axt2.nii.gz')
                    highb_path = os.path.join(synth_folder, folder,'b1600.nii.gz')
                    adc_path = os.path.join(synth_folder, folder,'adc.nii.gz')
                    if args.generated_label:
                        val_files.append({"axt2": axt2_path,
                                            "highb": highb_path,
                                            "adc": adc_path,
                                            "label": row["PIRADS_synth"]}) # Synthetic data with synthetic label
                    else:
                        val_files.append({"axt2": axt2_path,
                                            "highb": highb_path,
                                            "adc": adc_path,
                                            "label": row["ISUP_real"]})
            else: # inst2 flag with Institute1 data or we're not Institute1 flag/data/not-finetune then skip
                continue
    else:
        for idx, row in val_df.iterrows():
            folder = row["folder"]
            is_inst1 = row["dataset"] == "Institute1"
            if args.inst2_data and not is_inst1:
                img_path = os.path.join(real_folder, folder,f'{args.imkey}.nii.gz')
                val_files.append({"image": img_path, "label": row["ISUP_real"]})
            elif args.inst1_data and is_inst1 and (not args.inst2_data and not args.finetune): # If data is Institute1 and Institute1 flag is set (and not because we're fine-tuning on inst2)
                if args.real_data:
                    img_path = os.path.join(real_folder, folder,f'{args.imkey}.nii.gz')
                    val_files.append({"image": img_path, "label": row["ISUP_real"]})
                elif args.synth_data:
                    img_path = os.path.join(synth_folder, folder,f'{args.imkey}.nii.gz')
                    if args.generated_label:
                        val_files.append({"image": img_path, "label": row["PIRADS_synth"]}) # Synthetic data with synthetic label
                    else:
                        val_files.append({"image": img_path, "label": row["ISUP_real"]})
            else:
                continue
        
    test_files_all = []
    test_files_inst1 = []
    test_files_inst2 = []
    if args.imkey == "multi": # Files, dataloader, and input channels transforms will be different
        for idx, row in test_df.iterrows():
            folder = row["folder"]
            is_inst1 = row["dataset"] == "Institute1"
            axt2_path = os.path.join(real_folder, folder,'axt2.nii.gz')
            highb_path = os.path.join(real_folder, folder,'b1600.nii.gz')
            adc_path = os.path.join(real_folder, folder,'adc.nii.gz')
            test_files_all.append({"axt2": axt2_path,
                                    "highb": highb_path,
                                    "adc": adc_path,
                                    "label": row["ISUP_real"]})
            if is_inst1:
                test_files_inst1.append({"axt2": axt2_path,
                                        "highb": highb_path,
                                        "adc": adc_path,
                                        "label": row["ISUP_real"]})
            else:
                test_files_inst2.append({"axt2": axt2_path,
                                        "highb": highb_path,
                                        "adc": adc_path,
                                        "label": row["ISUP_real"]})
    else:
        for idx, row in test_df.iterrows():
            folder = row["folder"]
            is_inst1 = row["dataset"] == "Institute1"
            img_path = os.path.join(real_folder, folder,f'{args.imkey}.nii.gz')
            test_files_all.append({"image": img_path, "label": row["ISUP_real"]})
            if is_inst1:
                test_files_inst1.append({"image": img_path, "label": row["ISUP_real"]})
            else:
                test_files_inst2.append({"image": img_path, "label": row["ISUP_real"]})

    prob=0.5

    print(f"Train set size: {len(train_files)}, Val set size: {len(val_files)}, Test set size: {len(test_files_all)}")
    if args.imkey == "multi":
        train_transforms = [
                    LoadImaged(keys=["axt2","highb","adc"], ensure_channel_first=True, image_only=True),
                    CropForegroundd(keys=["axt2","highb","adc"], source_key="axt2"),
                    ClipIntensityPercentilesd(keys=["axt2","highb","adc"],lower=2.0, upper=98.0, sharpness_factor=10.0),
                    NormalizeIntensityd(keys=["axt2","highb","adc"]),

                    ConcatItemsd(keys=["axt2", "highb", "adc"], name="image"),
                    DeleteItemsd(keys=["axt2", "highb", "adc"]), # If we don't do this torch will try to collate them; doubles our cache size and will error
                ]
    else:
        train_transforms = [
                    LoadImaged(keys=["image"], ensure_channel_first=True, image_only=True),
                    CropForegroundd(keys=["image"], source_key="image"),
                    ClipIntensityPercentilesd(keys=["image"],lower=2.0, upper=98.0, sharpness_factor=10.0),
                    NormalizeIntensityd(keys=["image"]),
                ]
        
    train_transforms += [
                SpatialPadd(keys=["image"], spatial_size=(256,256,32), mode="constant", constant_values=[0]),
                CenterSpatialCropd(keys=["image"], roi_size=(256,256,32)),
                RandFlipd(keys=["image"], prob=prob, spatial_axis=0), # 0:l/r, 1: ant/post, 2: sup/inf
                RandRotated(keys=["image"], range_z=15*3.14/180, prob=prob, mode=('bilinear')),
                RandAffined(keys=["image"], prob=prob, translate_range=(25,25,2), padding_mode='zeros', mode=('bilinear')), # 10% of 256x256x32
                RandZoomd(keys=["image"], prob=prob, min_zoom=0.9, max_zoom=1.1, mode=('bilinear'), padding_mode='constant',constant_values=[0]),
            ]
    
    train_transforms = Compose(train_transforms)

    if args.imkey == "multi":
        val_transforms = [
                    LoadImaged(keys=["axt2","highb","adc"], ensure_channel_first=True, image_only=True),
                    CropForegroundd(keys=["axt2","highb","adc"], source_key="axt2"),
                    ClipIntensityPercentilesd(keys=["axt2","highb","adc"],lower=2.0, upper=98.0, sharpness_factor=10.0),
                    NormalizeIntensityd(keys=["axt2","highb","adc"]),

                    ConcatItemsd(keys=["axt2", "highb", "adc"], name="image"),
                    DeleteItemsd(keys=["axt2", "highb", "adc"]), # If we don't do this torch will try to collate them; doubles our cache size and will error
                ]
    else:
        val_transforms = [
            LoadImaged(keys=["image"], ensure_channel_first=True, image_only=True),
            CropForegroundd(keys=["image"], source_key="image"),
            ClipIntensityPercentilesd(keys=["image"],lower=2.0, upper=98.0, sharpness_factor=10.0),
            NormalizeIntensityd(keys=["image"]),
        ]
        
    val_transforms += [
        SpatialPadd(keys=["image"], spatial_size=(256,256,32), mode="constant", constant_values=[0]),
        CenterSpatialCropd(keys=["image"], roi_size=(256,256,32)),
    ]

    val_transforms = Compose(val_transforms)

    if train_flag:
        print("Beginning training")
        if args.imkey == "multi":
            cache_rate = 0.33
        else:
            cache_rate = 1.0
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=cache_rate, num_workers=7)
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=cache_rate, num_workers=1)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, shuffle=False)
        
        train(args,device,train_loader,val_loader,exp_path_starter)
    if val_flag: # Testing
        print("Beginning testing")
        test_ds_all = Dataset(data=test_files_all, transform=val_transforms)
        test_loader_all = DataLoader(test_ds_all, batch_size=1, num_workers=2)
        test_ds_inst2 = Dataset(data=test_files_inst2, transform=val_transforms)
        test_loader_inst2 = DataLoader(test_ds_inst2, batch_size=1, num_workers=2)
        test_ds_inst1 = Dataset(data=test_files_inst1, transform=val_transforms)
        test_loader_inst1 = DataLoader(test_ds_inst1, batch_size=1, num_workers=2)

        model = EfficientNetBN(
            model_name="efficientnet-"+args.model_type,
            pretrained=False,
            spatial_dims=3,
            in_channels=args.inchannels,
            num_classes=2
        ).to(device)
        model_options = glob(f"{exp_path_starter}/{args.exp_name}_checkpoint_key_metric=*.pt")
        # If multiple pick the highest
        if len(model_options) == 0:
            raise ValueError(f"No model found for {args.exp_name}")
        elif len(model_options) > 1:
            model_options = sorted(model_options, key=lambda x: int(x.split("=")[-1].split(".")[0]))
            model_path = model_options[-1]
        else:
            model_path = model_options[0]
        # Load the model
        model.load_state_dict(torch.load(model_path)["net"],strict=True)
        model.eval()
        try:
            evaluate(args,model,device,test_loader_all,test_loader_inst2,test_loader_inst1)
        except:
            pass

def evaluate(args,model,device,test_loader_all,test_loader_inst2,test_loader_inst1):
    post_transforms = Compose( # Voting
        [EnsureTyped(keys=["pred","label"],device=device),
         Activationsd(keys=["pred"], softmax=True),
         CopyItemsd(keys="pred",times=2,names=["pred_binary", "pred_single"]),
         AsDiscreted(keys="pred_binary", argmax=True, keepdim=True), # threshold=0.5
         CopyItemsd(keys="label",times=1,names=["label_binary"]),
         AsDiscreted(keys="label", to_onehot=2),
         Lambdad(keys="pred_single", func=lambda x: x[1]), # take the second channel
         EnsureTyped(keys=["pred","label","pred_binary"],device=device, data_type="tensor"),
         ] 
    )

    print("Institute1 metrics")
    evaluator_inst1 = SupervisedEvaluator(
        device=device,
        non_blocking=True,
        val_data_loader=test_loader_inst1,
        network=model,
        inferer=SimpleInferer(),
        postprocessing=post_transforms,
        key_val_metric={
            "test_ap": AveragePrecision(output_transform=from_engine(["pred_single", "label_binary"]), device=device), # pred, label without lambda in train
        },
        additional_metrics={
            "test_auc": ROCAUC(average="macro", output_transform=from_engine(["pred", "label"])), # pred, label without lambda in train
            "test_acc": ConfusionMatrix(include_background=False,metric_name='accuracy', output_transform=from_engine(["pred_binary", "label"])),
            "test_sens": ConfusionMatrix(include_background=False,metric_name='sensitivity', output_transform=from_engine(["pred_binary", "label"])),
            "test_spec": ConfusionMatrix(include_background=False,metric_name='specificity', output_transform=from_engine(["pred_binary", "label"])),
            "test_PPV": ConfusionMatrix(include_background=False,metric_name='precision', output_transform=from_engine(["pred_binary", "label"])),
            "test_NPV": ConfusionMatrix(include_background=False,metric_name='negative predictive value', output_transform=from_engine(["pred_binary", "label"])),
        },
        val_handlers=[
            MetricsSaver(save_dir=f"./outputs/tfproxy/{args.exp_name}_inst1",metrics=["test_ap", "test_auc", "test_acc", "test_sens", "test_spec", "test_PPV", "test_NPV"],
                         metric_details=["accuracy", "sensitivity", "specificity", "precision", "negative predictive value"]),
            ]
    )
    evaluator_inst1.run()

    print("Institute2 metrics")
    evaluator_inst2 = SupervisedEvaluator(
        device=device,
        non_blocking=True,
        val_data_loader=test_loader_inst2,
        network=model,
        inferer=SimpleInferer(),
        postprocessing=post_transforms,
        key_val_metric={
            "test_ap": AveragePrecision(output_transform=from_engine(["pred_single", "label_binary"]), device=device), # pred, label without lambda in train
        },
        additional_metrics={
            "test_auc": ROCAUC(average="macro", output_transform=from_engine(["pred", "label"])), # pred, label without lambda in train
            "test_acc": ConfusionMatrix(include_background=False,metric_name='accuracy', output_transform=from_engine(["pred_binary", "label"])),
            "test_sens": ConfusionMatrix(include_background=False,metric_name='sensitivity', output_transform=from_engine(["pred_binary", "label"])),
            "test_spec": ConfusionMatrix(include_background=False,metric_name='specificity', output_transform=from_engine(["pred_binary", "label"])),
            "test_PPV": ConfusionMatrix(include_background=False,metric_name='precision', output_transform=from_engine(["pred_binary", "label"])),
            "test_NPV": ConfusionMatrix(include_background=False,metric_name='negative predictive value', output_transform=from_engine(["pred_binary", "label"])),
        },
        val_handlers=[
            MetricsSaver(save_dir=f"./outputs/tfproxy/{args.exp_name}_inst2",metrics=["test_ap", "test_auc", "test_acc", "test_sens", "test_spec", "test_PPV", "test_NPV"],
                         metric_details=["accuracy", "sensitivity", "specificity", "precision", "negative predictive value"]),
            ]
    )
    evaluator_inst2.run()

    print("All metrics")
    evaluator_all = SupervisedEvaluator(
        device=device,
        non_blocking=True,
        val_data_loader=test_loader_all,
        network=model,
        inferer=SimpleInferer(),
        postprocessing=post_transforms,
        key_val_metric={
            "test_ap": AveragePrecision(output_transform=from_engine(["pred_single", "label_binary"]), device=device), # pred, label without lambda in train
        },
        additional_metrics={
            "test_auc": ROCAUC(average="macro", output_transform=from_engine(["pred", "label"])), # pred, label without lambda in train
            "test_acc": ConfusionMatrix(include_background=False,metric_name='accuracy', output_transform=from_engine(["pred_binary", "label"])),
            "test_sens": ConfusionMatrix(include_background=False,metric_name='sensitivity', output_transform=from_engine(["pred_binary", "label"])),
            "test_spec": ConfusionMatrix(include_background=False,metric_name='specificity', output_transform=from_engine(["pred_binary", "label"])),
            "test_PPV": ConfusionMatrix(include_background=False,metric_name='precision', output_transform=from_engine(["pred_binary", "label"])),
            "test_NPV": ConfusionMatrix(include_background=False,metric_name='negative predictive value', output_transform=from_engine(["pred_binary", "label"])),
        },
        val_handlers=[
            MetricsSaver(save_dir=f"./outputs/tfproxy/{args.exp_name}_all",metrics=["test_ap", "test_auc", "test_acc", "test_sens", "test_spec", "test_PPV", "test_NPV"],
                         metric_details=["accuracy", "sensitivity", "specificity", "precision", "negative predictive value"]),
            ]
    )
    evaluator_all.run()
    


def train(args,device,train_loader,val_loader,model_path):
    tensorboard_path = os.path.join("./outputs/tfproxy", f"{args.exp_name}")
    Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
    tensorboard_writer = SummaryWriter(tensorboard_path)
    net = EfficientNetBN(
        model_name="efficientnet-"+args.model_type,
        pretrained=False,
        spatial_dims=3,
        in_channels=args.inchannels,
        num_classes=2
    ).to(device)

    if args.finetune and not args.resume: # Resume takes prio over finetune in case we're resuming a finetune
        model_options=glob(f"{model_path}/{args.pretrain_name}_checkpoint_key_metric=*.pt") 
        if len(model_options) == 0:
            raise ValueError(f"No model found for {args.pretrain_name}")
        elif len(model_options) > 1:
            model_options = sorted(model_options, key=lambda x: int(x.split("=")[-1].split(".")[0]))
            finetune_model_path = model_options[-1]
        else:
            finetune_model_path = model_options[0]
        print(f"Loading pretrain checkpoint from {finetune_model_path}")
        # Load the model
        net.load_state_dict(torch.load(finetune_model_path)["net"],strict=True)
        args.lr = args.lr_finetune # Change to finetune LR

    loss = FocalLoss(
        to_onehot_y=True,
        use_softmax=True,
        gamma=2.0, 
        weight=None, 
        reduction="mean"
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), args.lr)


    total_steps = (args.num_epochs * len(train_loader.dataset)) / (args.batch_size)
    if args.add_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(opt, total_iters=total_steps, power=2.0)


    # Want probabilities, so use softmax then take the second channel
    val_post_transforms = Compose(
        [EnsureTyped(keys=["pred","label"],device=device),
         CopyItemsd(keys="pred",times=1,names=["pred_raw"]),
         CopyItemsd(keys="label",times=2,names=["label_raw","label_binary"]),
         Activationsd(keys="pred", softmax=True),
         CopyItemsd(keys="pred",times=2,names=["pred_binary", "pred_single"]),
         AsDiscreted(keys="pred_binary", argmax=True, keepdim=True),
         Lambdad(keys="pred_single", func=lambda x: x[1]),
        AsDiscreted(keys="label", to_onehot=2)
         ] 
    )


    val_handlers = [
        TensorBoardStatsHandler(summary_writer=tensorboard_writer, output_transform=lambda x:None),
    ]

    evaluator = SupervisedEvaluator(
        device=device,
        non_blocking=True,
        val_data_loader=val_loader,
        network=net,
        inferer=SimpleInferer(),
        postprocessing=val_post_transforms,
        key_val_metric={
            "val_ap": AveragePrecision(output_transform=from_engine(["pred_single", "label_binary"]), device=device),
        },
        additional_metrics={
            "val_auc": ROCAUC(average="macro", output_transform=from_engine(["pred", "label"])), 
            "val_acc": ConfusionMatrix(include_background=False,metric_name='accuracy', output_transform=from_engine(["pred_binary", "label"])),
            "val_sens": ConfusionMatrix(include_background=False,metric_name='sensitivity', output_transform=from_engine(["pred_binary", "label"])),
            "val_spec": ConfusionMatrix(include_background=False,metric_name='specificity', output_transform=from_engine(["pred_binary", "label"])),
            "val_PPV": ConfusionMatrix(include_background=False,metric_name='precision', output_transform=from_engine(["pred_binary", "label"])),
            "val_NPV": ConfusionMatrix(include_background=False,metric_name='negative predictive value', output_transform=from_engine(["pred_binary", "label"])),
            "val_loss": Loss(loss_fn=loss, output_transform=from_engine(["pred_raw", "label_raw"]), device=device),
        },
        val_handlers=val_handlers,
        amp=args.autocast,
    ) 
    train_handlers = [
        ValidationHandler(validator=evaluator, interval=5, epoch_level=True),
        TensorBoardStatsHandler(summary_writer=tensorboard_writer, tag_name='train_loss', output_transform=from_engine(["loss"], first=True)),
    ]
    if args.add_scheduler:
        train_handlers += [
            LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=False, epoch_level=False),
        ]

    trainer = SupervisedTrainer(
        device=device,
        non_blocking=True,
        max_epochs=args.num_epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=loss,
        inferer=SimpleInferer(),
        amp=args.autocast,
        train_handlers=train_handlers,
    )
    if args.add_scheduler:
        save_dict = {
            "trainer":trainer,
            "net": net,
            "opt": opt,
            "lr": lr_scheduler,
        }
    else:
        save_dict = {
            "trainer":trainer,
            "net": net,
            "opt": opt,
        }
    ckpt_saver=CheckpointSaver(save_dir=model_path,file_prefix=f"{args.exp_name}", save_dict=save_dict, save_key_metric=True,
                                key_metric_greater_or_equal=True, key_metric_name="val_ap", save_final=True, epoch_level=True,
                                key_metric_save_state=False)

    ckpt_saver.attach(evaluator)

    if args.resume:
        model_ckpt_paths=glob(f"{model_path}/{args.exp_name}_checkpoint_final_iteration=*.pt")
        if len(model_ckpt_paths)==0:
            raise ValueError(f"No checkpoint found for {args.exp_name}")
        elif len(model_ckpt_paths) > 1:
            model_ckpt_paths = sorted(model_ckpt_paths, key=lambda x: int(x.split("=")[-1].split(".")[0]))
            model_ckpt_path = model_ckpt_paths[-1]
        else:
            model_ckpt_path = model_ckpt_paths[0]
        print(f"Loading checkpoint from {model_ckpt_path}")
        ckpt_loader = CheckpointLoader(load_path=model_ckpt_path, load_dict=save_dict,map_location=device,strict=True)
        ckpt_loader(trainer)
        args.resume = False # Prevent loading again
    
    trainer.run()
    return net

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--imkey", type=str, default='axt2', help="Image key")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_finetune", type=float, default=1e-5, help="Learning rate for finetuning (will override args.lr if --finetune active)")
    parser.add_argument("-b","--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-n","--num_workers", type=int, default=7, help="Num workers")
    parser.add_argument("--num_epochs_inst1", type=int, default=155, help="Num epochs for Institute1")
    parser.add_argument("--num_epochs_inst2", type=int, default=385, help="Num epochs for Institute2")
    parser.add_argument("--num_epochs_finetune", type=int, default=385, help="Num epochs for Fine tuning")

    # Flags for data, should have inst2 yes/no, Institute1 yes/no, real or synthetic for Institute1, generated or true labels for synthetic Institute1
    parser.add_argument("-p","--inst2_data", action="store_true", help="Use Institute2 data (real only)")
    parser.add_argument("-u","--inst1_data", action="store_true", help="Use Institute1 data, requires real or synthetic flag")
    parser.add_argument("-r","--real_data", action="store_true", help="Use real data for Institute1")
    parser.add_argument("-s","--synth_data", action="store_true", help="Use synthetic data for Institute1")
    parser.add_argument("-l","--generated_label", type=int, help="Use generated if true, real if false labels for synthetic data")
    parser.add_argument("-f","--finetune", action="store_true", help="Fine tune an existing classifier")

    parser.add_argument("-a","--add_to_modelname", type=str, default='', help="Add to model name")
    parser.add_argument("-c","--count", type=int, default=0, help="What iterator are we running")
    parser.add_argument("-v","--validate", action="store_true", help="Validate instead of train")
    parser.add_argument("--autocast", action="store_true", help="Use Torch AMP")
    
    parser.add_argument("--resume", action="store_true", help="Resume training from latest")
    parser.add_argument("--seed", type=int, default=12345, help="Set seed for reproducibility")
    parser.add_argument("--add_scheduler", action="store_true", help="Add cosine annealing LR scheduler")
    parser.add_argument("-m","--model_type", type=str, default='b0', help="Type of efficientnet to use")
    parser.add_argument("-o", "--output_dir", default="./synthdata", type=str, help="Output directory for synthetic images")
    parser.add_argument("--spreadsheet_path", default="./synthdata/synthsheet.xlsx", type=str, help="Path to the spreadsheet for additional data")
    parser.add_argument("--realdata_path", default="./realdata", type=str, help="Path to the real data directory")
    parser.add_argument("--synthdata_path", default="./synthdata", type=str, help="Path to the synthetic data directory")

    args = parser.parse_args()

    if args.synth_data and args.generated_label is None:
        args.generated_label = 1 # Default to using the synthetic label

    if args.imkey == "multi":
        args.inchannels = 3
    else:
        args.inchannels = 1

    # Validate args:
    if not args.inst2_data and not args.inst1_data:
        raise ValueError("Must specify inst2 or Institute1 data")
    if args.inst1_data and (not args.real_data and not args.synth_data):
        raise ValueError("Must specify one of real or synthetic data for Institute1")
    if args.synth_data and args.generated_label is None:
        raise ValueError("Must specify generated or true labels for synthetic data")
    if args.real_data and args.synth_data:
        raise ValueError("Cannot specify both real and synthetic data")
    if args.inst2_data and args.inst1_data and not args.finetune:
        raise ValueError("Institute2 and Institute1 flags active require fine-tune to be active also")
    if args.finetune and (not args.inst2_data or not args.inst1_data):
        raise ValueError("This experiment should only fine-tune if Institute2 and Institute1 flags are active")
    
    if args.finetune:
        print(f"This is a finetune, using {args.num_epochs_finetune} for epochs")
        args.num_epochs = args.num_epochs_finetune
    elif args.inst2_data:
        print(f"This is a train on Institute2, using {args.num_epochs_inst2} for epochs")
        args.num_epochs = args.num_epochs_inst2
    elif args.inst1_data:
        print(f"This is a train on Institute1, using {args.num_epochs_inst1} for epochs")
        args.num_epochs = args.num_epochs_inst1
    else:
        raise ValueError("Unknown training paradigm")

    prep(args,train_flag=not args.validate,val_flag=True)


if __name__ == "__main__":
    main()