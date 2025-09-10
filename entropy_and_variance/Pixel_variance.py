import numpy as np
import SimpleITK as sitk
import os
from tqdm import tqdm


path_variance = "/home/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/imagesTs/variances_preds"
path_img = "/home/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/imagesTs"
outputpath = "/home/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/imagesTs/variances_nifti" # Output path

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
        
        variance = (variance - np.min(variance)) / (np.max(variance) - np.min(variance)) * 100
                
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
