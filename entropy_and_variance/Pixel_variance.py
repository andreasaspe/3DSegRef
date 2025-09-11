import numpy as np
import SimpleITK as sitk
import os
from tqdm import tqdm
import time

path_variance = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/predictions/man_preds_variance"
path_img = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/imagesTs" # For copying image information
outputpath = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/predictions/man_preds_variance_nifti" # Output path

os.makedirs(outputpath, exist_ok=True)

all_subjects = [x.split('_')[0] for x in os.listdir(path_variance) if x.endswith('.npz')]

all_subjects = sorted(all_subjects, key=lambda x: int(x))

# def entropy(probs):
#     # probs: (C,H,W,D)
#     eps = 1e-10  # to avoid log(0)
#     H = -np.sum(probs * np.log2(probs + eps), axis=0)  # sum over channel dimension
#     return H  # shape: (H,W,D)

for subject in tqdm(all_subjects):
    try:
        variance = np.load(os.path.join(path_variance, subject + "_0000_pred_var.nii.gz.npz"))['probabilities'][1] # Get only foreground class
        
        print(f"Loaded variance for {subject}, shape: {variance.shape}, min: {np.min(variance)}, max: {np.max(variance)}")
        
        time.sleep(1)  # Pause for 1 second to ensure the print statement is readable
        
        vmin, vmax = np.min(variance), np.max(variance)
        if vmax > vmin:
            variance = (variance - vmin) / (vmax - vmin) * 100 #SCALING!
        else:
            # variance map is flat → no contrast
            variance = np.zeros_like(variance)                
            
        img_sitk = sitk.ReadImage(os.path.join(path_img, subject + "_0000.nii.gz")) # Only for copying information

        # print(subject)
        # print(variance.shape)
        # print(img_sitk.GetSize()[::-1])

        # Create a new SimpleITK image from the variance array
        variance_sitk = sitk.GetImageFromArray(variance)
        variance_sitk.CopyInformation(img_sitk)
        sitk.WriteImage(variance_sitk, os.path.join(outputpath, subject + "_variance.nii.gz"))
        print(f"Saved variance for {subject}")
    except:
        print(f"Could not load {subject}")
        continue

    # variance_sitk = sitk.GetImageFromArray(variance)
    # variance_sitk.CopyInformation(img_sitk)
    # sitk.WriteImage(variance_sitk, os.path.join(outputpath, subject + "_variance.nii.gz"))
