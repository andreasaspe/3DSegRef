import numpy as np
import SimpleITK as sitk
import os
from tqdm import tqdm
import time

path_deterministic = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/predictions/man_preds_deterministic"
path_stochastic_PPT = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/predictions/man_preds_stochastic_PPT"
path_stochastic_multibasis = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/predictions/man_preds_stochastic_multibasis"
path_img = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/imagesTs" # For copying image information
outputpath = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/predictions/probs" # Output path

os.makedirs(outputpath, exist_ok=True)

all_subjects = [x.split('_')[0] for x in os.listdir(path_img) if x.endswith('.nii.gz')]

all_subjects = sorted(all_subjects, key=lambda x: int(x))


for subject in tqdm(all_subjects):
    probs_deterministic = np.load(os.path.join(path_deterministic, subject + "_0000_pred.nii.gz.npz"))['probabilities'][1] # Get only foreground class
    probs_multibasis = np.load(os.path.join(path_stochastic_multibasis, subject + "_0000_pred.nii.gz.npz"))['probabilities'][1] # Get only foreground class
    probs_PPT = np.load(os.path.join(path_stochastic_PPT, subject + "_0000_pred.nii.gz.npz"))['probabilities'][1] # Get only foreground class

    img_sitk = sitk.ReadImage(os.path.join(path_img, subject + "_0000.nii.gz")) # Only for copying information
    
    print(f"Loaded probabilities for {subject}, shape: {probs_deterministic.shape}, min: {np.min(probs_deterministic)}, max: {np.max(probs_deterministic)}, mean: {np.mean(probs_deterministic)}")
    print(f"Loaded probabilities for {subject}, shape: {probs_multibasis.shape}, min: {np.min(probs_multibasis)}, max: {np.max(probs_multibasis)}, mean: {np.mean(probs_multibasis)}")
    print(f"Loaded probabilities for {subject}, shape: {probs_PPT.shape}, min: {np.min(probs_PPT)}, max: {np.max(probs_PPT)}, mean: {np.mean(probs_PPT)}")

    # Calculate probs_difference
    probs_difference_multibasis = probs_deterministic - probs_multibasis
    probs_difference_PPT = probs_deterministic - probs_PPT
    
    # Save probabilities
    probs_deterministic_sitk = sitk.GetImageFromArray(probs_deterministic)
    probs_deterministic_sitk.CopyInformation(img_sitk)
    sitk.WriteImage(probs_deterministic_sitk, os.path.join(outputpath, subject + "_probs_deterministic.nii.gz"))
    
    probs_multibasis_sitk = sitk.GetImageFromArray(probs_multibasis)
    probs_multibasis_sitk.CopyInformation(img_sitk)
    sitk.WriteImage(probs_multibasis_sitk, os.path.join(outputpath, subject + "_probs_multibasis.nii.gz"))
    
    probs_PPT_sitk = sitk.GetImageFromArray(probs_PPT)
    probs_PPT_sitk.CopyInformation(img_sitk)
    sitk.WriteImage(probs_PPT_sitk, os.path.join(outputpath, subject + "_probs_PPT.nii.gz"))
    
    probs_difference_multibasis_sitk = sitk.GetImageFromArray(probs_difference_multibasis)
    probs_difference_multibasis_sitk.CopyInformation(img_sitk)
    sitk.WriteImage(probs_difference_multibasis_sitk, os.path.join(outputpath, subject + "_probs_difference_multibasis.nii.gz"))
    
    probs_difference_PPT_sitk = sitk.GetImageFromArray(probs_difference_PPT)
    probs_difference_PPT_sitk.CopyInformation(img_sitk)
    sitk.WriteImage(probs_difference_PPT_sitk, os.path.join(outputpath, subject + "_probs_difference_PPT.nii.gz"))




















        
    #     # print(f"Loaded variance for {subject}, shape: {variance.shape}, min: {np.min(variance)}, max: {np.max(variance)}")
    #     # time.sleep(1)  # Pause for 1 second to ensure the print statement is readable
        
    #     vmin, vmax = np.min(variance), np.max(variance)
    #     if vmax > vmin:
    #         variance = (variance - vmin) / (vmax - vmin) * 100 #SCALING!
    #     else:
    #         # variance map is flat → no contrast
    #         variance = np.zeros_like(variance)                
            
    #     img_sitk = sitk.ReadImage(os.path.join(path_img, subject + "_0000.nii.gz")) # Only for copying information

    #     # print(subject)
    #     # print(variance.shape)
    #     # print(img_sitk.GetSize()[::-1])

    #     # Create a new SimpleITK image from the variance array
    #     variance_sitk = sitk.GetImageFromArray(variance)
    #     variance_sitk.CopyInformation(img_sitk)
    #     sitk.WriteImage(variance_sitk, os.path.join(outputpath, subject + "_variance.nii.gz"))
    #     print(f"Saved variance for {subject}")
    # except:
    #     print(f"Could not load {subject}")
    #     continue
