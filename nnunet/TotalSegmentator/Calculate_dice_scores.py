import os
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk

def dice_score(y_true, y_pred, smooth=1e-6):
    """
    Compute the Dice score between two binary masks.

    Args:
        y_true (numpy.ndarray): Ground truth binary mask.
        y_pred (numpy.ndarray): Predicted binary mask.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: Dice score.
    """
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

label_gt_path = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/labelsTs"
label_pred_path_deterministic = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/imagesTs/man_preds_deterministic"
label_pred_path = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/imagesTs/man_preds"

label_gt_file = [x for x in os.listdir(label_gt_path) if x.endswith(".nii.gz")]
label_pred_file = [x for x in os.listdir(label_pred_path) if x.endswith(".nii.gz")]

all_subjects = [x.split('_')[0] for x in label_pred_file]

#Make sure the files are sorted in the same order
label_gt_file = sorted(label_gt_file, key=lambda x: x.split('_')[0])
label_pred_file = sorted(label_pred_file, key=lambda x: x.split('_')[1])

# if len(label_gt_file) != len(label_pred_file):
#     raise ValueError("Number of ground truth and prediction files do not match")

dice_list = []
n = len(label_gt_file)

for subject in all_subjects:
    try:
        mask_gt = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(label_gt_path, f"{subject}.nii.gz")))
        pred = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(label_pred_path, f"{subject}_0000_pred.nii.gz")))
        pred_deterministic = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(label_pred_path_deterministic, f"{subject}_0000_pred.nii.gz")))
        
        # print(os.path.join(label_gt_path,label_gt_file[i]))
        # print(os.path.join(label_pred_path,label_pred_file[i]))
        # tools.plot_central_slice_img_mask(mask_gt, pred, spacing=None)
        

        dice = dice_score(mask_gt, pred)
        dice_deterministic = dice_score(mask_gt, pred_deterministic)

        dice_diff = dice - dice_deterministic
        
        
        
        dice_list.append(dice_diff)
        

        print(f"Dice diff for subject {subject} is: {dice_diff}")
    except:
        print(f"Error processing subject {subject}. Skipping...")

print(f"Mean dice score: {round(np.mean(dice_list), 3)}")
print(f"Median dice score: {round(np.median(dice_list), 3)}")
print(f"Max dice score: {round(np.max(dice_list), 3)}")
print(f"Min dice score: {round(np.min(dice_list), 3)}")
