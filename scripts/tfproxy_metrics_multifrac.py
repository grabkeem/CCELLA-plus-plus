import argparse
import os
import sys
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from monai.transforms import Compose, LoadImaged, CropForegroundd, ScaleIntensityRangePercentilesd, SpatialPadd, CenterSpatialCropd, ConcatItemsd, DeleteItemsd, ClipIntensityPercentilesd, NormalizeIntensityd
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism
from monai.networks.nets import EfficientNetBN
import torch

def evaluate_model(model_path, test_loader, device, model_type="b0", multi=False):
    model = EfficientNetBN(
        model_name="efficientnet-"+model_type,
        pretrained=False,
        spatial_dims=3,
        in_channels=3 if multi else 1,
        num_classes=2
    ).to(device)
    model.load_state_dict(torch.load(model_path,weights_only=True)["net"], strict=True)
    model.eval()

    results_array = []
    for batch in tqdm(test_loader, desc="Evaluating"):
        with torch.no_grad():
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            folder = batch["folder"]

            pred_raw = model(image)
            pred = torch.softmax(pred_raw, dim=1)
            preds_np = pred.cpu().detach().numpy()

            for case_i in range(label.shape[0]):
                case_raw = preds_np[case_i]
                case_pred = case_raw.argmax()
                results_array.append([
                    folder[case_i].item(),
                    case_raw[1],
                    case_pred,
                    label[case_i].item()
                ])
    return np.array(results_array)

def calculate_metrics(results_array):
    raws = results_array[:, 1]
    preds = results_array[:, 2]
    labels = results_array[:, 3]

    auc = roc_auc_score(labels, raws)
    ap = average_precision_score(labels, raws)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) != 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) != 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) != 0 else np.nan

    return {
        "AUC": auc,
        "AP": ap,
        "Accuracy": acc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": ppv,
        "NPV": npv
    }

def validate(args):
    set_determinism(seed=12345)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    device = torch.device("cuda:0")

    df_path = f"{args.output_dir}/synthsheet_tfproxy"
    if args.imkey == "multi":
        df_path += "_multi"
    df_path += ".xlsx"
    df = pd.read_excel(df_path)
    test_df = df[df["Label"] == "test"]

    real_folder = args.realdata_path
    synth_folder = args.synthdata_path
    if args.imkey == "multi":
        synth_folder += "_multi"

    inst1_files = []
    non_inst1_files = []

    for _, row in test_df.iterrows():
        folder = row["folder"]
        # folder_uid is a number-only variant to folder, should be unique despite lack of non-numeric chars
        folder_uid = int(''.join(filter(str.isdigit, folder)))
        is_inst1 = folder.startswith("AIPR_2")
        if args.imkey == "multi":
            axt2_path = os.path.join(real_folder, folder,'axt2.nii.gz')
            highb_path = os.path.join(real_folder, folder,'b1600.nii.gz')
            adc_path = os.path.join(real_folder, folder,'adc.nii.gz')
            if is_inst1:
                inst1_files.append({"axt2": axt2_path,
                                        "highb": highb_path,
                                        "adc": adc_path,
                                        "label": row["ISUP_real"], "folder":folder_uid})
            else:
                non_inst1_files.append({"axt2": axt2_path,
                                        "highb": highb_path,
                                        "adc": adc_path,
                                        "label": row["ISUP_real"],"folder":folder_uid})
        else:
            img_path = os.path.join(real_folder, folder,f'{args.imkey}.nii.gz')
            if is_inst1:
                inst1_files.append({"image": img_path, "label": row["ISUP_real"], "folder":folder_uid})
            else:
                non_inst1_files.append({"image": img_path, "label": row["ISUP_real"],"folder":folder_uid})


    if args.imkey == "multi":
        val_transforms = [
                    LoadImaged(keys=["axt2","highb","adc"], ensure_channel_first=True, image_only=True),
                    # Crop foreground then pad to 256x256x32
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

    test_files = inst1_files + non_inst1_files
    inst1_test_ds = Dataset(data=inst1_files, transform=val_transforms)
    inst1_test_loader = DataLoader(inst1_test_ds, batch_size=1, num_workers=4, pin_memory=True)

    non_inst1_test_ds = Dataset(data=non_inst1_files, transform=val_transforms)
    non_inst1_test_loader = DataLoader(non_inst1_test_ds, batch_size=1, num_workers=4, pin_memory=True)
    test_ds = Dataset(data=test_files, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, pin_memory=True)

    all_metrics = []
    for p in tqdm([0,6.25,12.5,25,50,100]):
        addname = ""
        if args.imkey == "multi":
            addname += "_multi"
        if args.exp_name.endswith('_p'):
            if p == 0: # Ignore the 0 case on P 
                continue # Skip the first model if it ends with '_p'
            elif p == 100:
                model_name = f"{args.exp_name}{addname}"
            else:
                model_name = f"{args.exp_name}{addname}_p{p}"
        else:
            if p == 0:
                model_name = f"{args.exp_name}{addname}"
            else:
                model_name = f"{args.exp_name}_pf{addname}"
                if p != 100:
                    model_name += f"_p{p}"
        if args.count_start == 0: # and count_end
            model_path = glob(f"./models/exp3/{model_name}_checkpoint_key_metric=*.pt")
            if not model_path:
                print(f"No model found for {model_name}, looked for {f'./models/exp3/{model_name}_checkpoint_key_metric=*.pt'}")
                continue
            model_path = sorted(model_path, key=lambda x: int(x.split("=")[-1].split(".")[0]))[-1]

            print(f"Evaluating model: {model_name}")

            # Evaluate on combined test set
            results_array = evaluate_model(model_path, test_loader, device, args.model_type)
            metrics = calculate_metrics(results_array)
            metrics["Model"] = model_name
            metrics["Dataset"] = "Combined"
            all_metrics.append(metrics)

            # Evaluate on Institute1 test set
            results_array_inst1 = evaluate_model(model_path, inst1_test_loader, device, args.model_type)
            metrics_inst1 = calculate_metrics(results_array_inst1)
            metrics_inst1["Model"] = model_name
            metrics_inst1["Dataset"] = "Institute1"
            all_metrics.append(metrics_inst1)

            # Evaluate on Non-Institute1 test set
            results_array_non_inst1 = evaluate_model(model_path, non_inst1_test_loader, device, args.model_type)
            metrics_non_inst1 = calculate_metrics(results_array_non_inst1)
            metrics_non_inst1["Model"] = model_name
            metrics_non_inst1["Dataset"] = "Non-Institute1"
            all_metrics.append(metrics_non_inst1)
        else: # Run a loop for all counts
            initial_modelname = model_name
            for count in range(args.count_start,args.count_end+1):
                model_name = initial_modelname+f"_{count}"
                model_path = glob(f"./models/exp3/{model_name}_checkpoint_key_metric=*.pt")
                if not model_path:
                    print(f"No model found for {model_name}, looked for {f'./models/exp3/{model_name}_checkpoint_key_metric=*.pt'}")
                    continue
                model_path = sorted(model_path, key=lambda x: int(x.split("=")[-1].split(".")[0]))[-1]

                print(f"Evaluating model: {model_name}")

                # Evaluate on combined test set
                results_array = evaluate_model(model_path, test_loader, device, args.model_type, multi=(args.imkey == "multi"))
                metrics = calculate_metrics(results_array)
                metrics["Model"] = model_name
                metrics["Dataset"] = "Combined"
                all_metrics.append(metrics)

                # Evaluate on Institute1 test set
                results_array_inst1 = evaluate_model(model_path, inst1_test_loader, device, args.model_type, multi=(args.imkey == "multi"))
                metrics_inst1 = calculate_metrics(results_array_inst1)
                metrics_inst1["Model"] = model_name
                metrics_inst1["Dataset"] = "Institute1"
                all_metrics.append(metrics_inst1)

                # Evaluate on Non-Institute1 test set
                results_array_non_inst1 = evaluate_model(model_path, non_inst1_test_loader, device, args.model_type, multi=(args.imkey == "multi"))
                metrics_non_inst1 = calculate_metrics(results_array_non_inst1)
                metrics_non_inst1["Model"] = model_name
                metrics_non_inst1["Dataset"] = "Non-Institute1"
                all_metrics.append(metrics_non_inst1)


    metrics_df = pd.DataFrame(all_metrics)
    excel_filestarter = f"./outputs/tfproxy/{args.exp_name}"
    if args.imkey == "multi":
        excel_filestarter += "_multi"
    excel_out_filename = f"{excel_filestarter}_all_metrics.xlsx"
    metrics_df.to_excel(excel_out_filename, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--imkey", type=str, default='axt2', help="Image key")
    parser.add_argument("-m", "--model_type", type=str, default='b0', help="Type of efficientnet to use")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--count_start", type=int, default=0, help="Count start (inclusive)")
    parser.add_argument("--count_end", type=int, default=0, help="Count end (inclusive)")
    parser.add_argument("-o", "--output_dir", default="./synthdata", type=str, help="Output directory for synthetic images")
    parser.add_argument("--spreadsheet_path", default="./synthdata/synthsheet.xlsx", type=str, help="Path to the spreadsheet for additional data")
    parser.add_argument("--realdata_path", default="./realdata", type=str, help="Path to the real data directory")
    parser.add_argument("--synthdata_path", default="./synthdata", type=str, help="Path to the synthetic data directory")
    args = parser.parse_args()

    if (args.count_start != 0 and args.count_end==0) or (args.count_start == 0 and args.count_end!=0):
        raise ValueError("Both count_start and count_end must be specified")

    if args.imkey == "multi":
        args.inchannels = 3
    else:
        args.inchannels = 1

    validate(args)

if __name__ == "__main__":
    main()
