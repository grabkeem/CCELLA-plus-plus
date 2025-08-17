import os
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import argparse
from monai.metrics.fid import FIDMetric


def spatial_average_2d(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3], keepdim=keepdim)

def get_features_2d(image, radnet):
    # If input has just 1 channel, repeat channel to have 3 channels
    if image.shape[1]:
        image = image.repeat(1, 3, 1, 1)

    # Change order from 'RGB' to 'BGR'
    image = image[:, [2, 1, 0], ...]


    # Get model outputs
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True):
            feature_image = radnet.forward(image)
            # flattens the image spatially
            feature_image = spatial_average_2d(feature_image, keepdim=False)

    return feature_image

def spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3, 4], keepdim=keepdim)

def get_features(image, radnet):

    # Get model outputs
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=True):
            feature_image = radnet.forward(image)
            # flattens the image spatially
            feature_image = spatial_average(feature_image, keepdim=False)

    return feature_image

def main(multi, device, real_dir, synth_dir, df_path):
    device = torch.device(f"cuda:{device}" if (device != -1 and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    radnet_2d = torch.hub.load("Warvito/radimagenet-models", model="radimagenet_resnet50", verbose=True, trust_repo=True, ) # 2D
    radnet_2d.to(device)
    radnet_2d.eval()

    radnet = torch.hub.load("Warvito/MedicalNet-models", model="medicalnet_resnet50_23datasets", verbose=True, trust_repo=True, )
    radnet.to(device)
    radnet.eval()

    # Directories
    real_dir = real_dir
    synth_dir = synth_dir
    out_dir = "."
    df_path = df_path

    add_to_filename = ""

    if multi:
        synth_dir += "_multi"
        df_path += "_multi"
        add_to_filename += "_multi"
    df_path += ".xlsx"

    full_df = pd.read_excel(df_path)

    test_df = full_df[full_df["Label"] == "test"]

    print(f"Test set size: {len(test_df)}")

    tvt = "test" # Only assessing metrics on test set

    df = test_df

    df = df[df["dataset"] == "Institute1"]
    print(f"Number of subfolders in {tvt} set: {len(df)}")

    # Output Excel file
    output_file = f"{out_dir}/ldm_metrics_test{add_to_filename}.xlsx"

    keys = ['axt2', 'b1600', 'adc'] if multi else ['axt2']

    synth_features = {key: [] for key in keys}
    real_features = {key: [] for key in keys}
    synth_features_2d_x = {key: [] for key in keys}
    synth_features_2d_y = {key: [] for key in keys}
    synth_features_2d_z = {key: [] for key in keys}
    real_features_2d_x = {key: [] for key in keys}
    real_features_2d_y = {key: [] for key in keys}
    real_features_2d_z = {key: [] for key in keys}

    with torch.no_grad():
        for idx, row in tqdm(df.iterrows()):
            subfolder = row["folder"]
            if not row["dataset"] == "Institute1":
                raise ValueError(f"Subfolder {subfolder} does not belong to 'Institute1'") # Smoketest; should be dropped by this point
            real_subfolder = os.path.join(real_dir, subfolder)
            synth_subfolder = os.path.join(synth_dir, subfolder)

            if os.path.isdir(real_subfolder) and os.path.isdir(synth_subfolder):
                for key in keys:
                    real_file = os.path.join(real_subfolder, f"{key}.nii.gz")
                    synth_file = os.path.join(synth_subfolder, f"{key}.nii.gz")

                    if os.path.exists(real_file) and os.path.exists(synth_file):
                        # Load images
                        real_img = nib.load(real_file).get_fdata()
                        synth_img = nib.load(synth_file).get_fdata()

                        if real_img.shape != synth_img.shape:
                            print(f"Shape mismatch in {subfolder}: {real_img.shape} vs {synth_img.shape}")

                            # Force the shapes to match by center cropping the larger image
                            min_shape = np.minimum(real_img.shape, synth_img.shape)
                            real_img = real_img[
                                (real_img.shape[0] - min_shape[0]) // 2 : (real_img.shape[0] + min_shape[0]) // 2,
                                (real_img.shape[1] - min_shape[1]) // 2 : (real_img.shape[1] + min_shape[1]) // 2,
                                (real_img.shape[2] - min_shape[2]) // 2 : (real_img.shape[2] + min_shape[2]) // 2
                            ]
                            synth_img = synth_img[
                                (synth_img.shape[0] - min_shape[0]) // 2 : (synth_img.shape[0] + min_shape[0]) // 2,
                                (synth_img.shape[1] - min_shape[1]) // 2 : (synth_img.shape[1] + min_shape[1]) // 2,
                                (synth_img.shape[2] - min_shape[2]) // 2 : (synth_img.shape[2] + min_shape[2]) // 2
                            ]
                        # Check if the images are still the same shape after cropping
                            if real_img.shape != synth_img.shape:
                                raise ValueError(f"Shape mismatch after cropping in {subfolder}: {real_img.shape} vs {synth_img.shape}")
                                continue
                            else:
                                pass

                        real_img = real_img.astype(np.float32)
                        synth_img = synth_img.astype(np.float32)

                        # Normalize images
                        eps = 1e-8
                        real_img_clip = (real_img - np.min(real_img)) / ((np.max(real_img) - np.min(real_img)) + eps)
                        real_img_z = (real_img - np.mean(real_img)) / (np.std(real_img) + eps)
                        synth_img_clip = (synth_img - np.min(synth_img)) / ((np.max(synth_img) - np.min(synth_img)) + eps)

                        real_img = real_img_z

                        real_tensor = torch.from_numpy(real_img).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimensions
                        synth_tensor = torch.from_numpy(synth_img).unsqueeze(0).unsqueeze(0).to(device)
                        
                        real_tensor_clip_torch = torch.from_numpy(real_img_clip).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimensions
                        synth_tensor_clip_torch = torch.from_numpy(synth_img_clip).unsqueeze(0).unsqueeze(0).to(device)

                        # Compute features
                        real_features[key].append(get_features(real_tensor, radnet))
                        synth_features[key].append(get_features(synth_tensor, radnet))

                        # Loop over slices to get 2D features
                        for i in range(real_img.shape[0]):
                            real_slice = real_tensor_clip_torch[:, :, i, :, :]
                            synth_slice = synth_tensor_clip_torch[:, :, i, :, :]

                            real_features_2d_x[key].append(get_features_2d(real_slice, radnet_2d))
                            synth_features_2d_x[key].append(get_features_2d(synth_slice, radnet_2d))
                        
                        for i in range(real_img.shape[1]):
                            real_slice = real_tensor_clip_torch[:, :, :, i, :]
                            synth_slice = synth_tensor_clip_torch[:, :, :, i, :]

                            real_features_2d_y[key].append(get_features_2d(real_slice, radnet_2d))
                            synth_features_2d_y[key].append(get_features_2d(synth_slice, radnet_2d))

                        for i in range(real_img.shape[2]):
                            real_slice = real_tensor_clip_torch[:, :, :, :, i]
                            synth_slice = synth_tensor_clip_torch[:, :, :, :, i]

                            real_features_2d_z[key].append(get_features_2d(real_slice, radnet_2d))
                            synth_features_2d_z[key].append(get_features_2d(synth_slice, radnet_2d))

                    else:
                        raise ValueError(f"Files not found in {subfolder}: {real_file}, {synth_file}")
            else:
                raise ValueError(f"Subfolder not found: {subfolder}")

    # Compute FID metrics
    fid_results = {key: [] for key in keys}
    fid_results_2d_x = {key: [] for key in keys}
    fid_results_2d_y = {key: [] for key in keys}
    fid_results_2d_z = {key: [] for key in keys}
    fid = FIDMetric()

    df_aggregate = pd.DataFrame(columns=["Metric"]+ keys)

    for key in keys:
        # Compute FID for 3D features
        fid_results[key] = fid(torch.vstack(synth_features[key]), torch.vstack(real_features[key])).item()

        # Compute FID for 2D features
        fid_results_2d_x[key] = fid(torch.vstack(synth_features_2d_x[key]), torch.vstack(real_features_2d_x[key])).item()
        fid_results_2d_y[key] = fid(torch.vstack(synth_features_2d_y[key]), torch.vstack(real_features_2d_y[key])).item()
        fid_results_2d_z[key] = fid(torch.vstack(synth_features_2d_z[key]), torch.vstack(real_features_2d_z[key])).item()

        print(f"FID for {key}: {fid_results[key]}, 2D X: {fid_results_2d_x[key]}, 2D Y: {fid_results_2d_y[key]}, 2D Z: {fid_results_2d_z[key]}")

    fid_row = ["FID"] + [fid_results[key] for key in keys]
    fid_row_2d_x = ["FID_2D_X"] + [fid_results_2d_x[key] for key in keys]
    fid_row_2d_y = ["FID_2D_Y"] + [fid_results_2d_y[key] for key in keys]
    fid_row_2d_z = ["FID_2D_Z"] + [fid_results_2d_z[key] for key in keys]
    df_aggregate.loc[len(df_aggregate)] = fid_row
    df_aggregate.loc[len(df_aggregate)] = fid_row_2d_x
    df_aggregate.loc[len(df_aggregate)] = fid_row_2d_y
    df_aggregate.loc[len(df_aggregate)] = fid_row_2d_z

    # Save to Excel
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_aggregate.to_excel(writer, sheet_name='Metrics', index=False)

    print(f"Metrics saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--multi", action="store_true", help="Whether to assess multi-sequence data")
    parser.add_argument("--device", type=int, default=0, help="Device to use for computation (default: 0 for GPU. CPU if -1 or not available)")
    parser.add_argument("-r", "--real_dir", default="./realdata", type=str, help="Directory for real images")
    parser.add_argument("-s", "--synth_dir", default="./synthdata", type=str, help="Directory for synthetic images")
    parser.add_argument("--spreadsheet_path", default="./synthdata/synthsheet.xlsx", type=str, help="Path to the spreadsheet for additional data")
    args = parser.parse_args()

    main(args.multi, args.device, args.real_dir, args.synth_dir, args.spreadsheet_path)

