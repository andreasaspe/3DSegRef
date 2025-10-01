import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from matplotlib.patches import Patch
from tqdm import tqdm

def get_direction_code(img_sitk):
    """ Get the direction of a SimpleITK image.
    Args:
        img_sitk (SimpleITK.Image): Input image.
    Returns:
        str: Direction code of the image.
    """
    direction_code = sitk.DICOMOrientImageFilter().GetOrientationFromDirectionCosines(img_sitk.GetDirection())

    return direction_code

image_path = "/scratch/awias/data/Pancreas/nnUNet_dataset/nnUNet_raw/Dataset001_Pancreas/imagesTs"
label_gt_path = "/scratch/awias/data/Pancreas/nnUNet_dataset/nnUNet_raw/Dataset001_Pancreas/labelsTs"
label_pred_path = "/scratch/awias/data/Pancreas/nnUNet_dataset/nnUNet_raw/Dataset001_Pancreas/imagesTs/man_preds"

outputfolder = "/scratch/awias/data/Pancreas/test2"

os.makedirs(outputfolder, exist_ok=True)

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

for subject in tqdm(all_subjects):
        mask_gt_sitk = sitk.ReadImage(os.path.join(label_gt_path, f"{subject}.nii.gz"))
        pred_sitk = sitk.ReadImage(os.path.join(label_pred_path, f"{subject}_0000_pred.nii.gz"))
        img_sitk = sitk.ReadImage(os.path.join(image_path, f"{subject}_0000.nii.gz"))
        
        mask_gt = sitk.GetArrayFromImage(mask_gt_sitk)
        pred = sitk.GetArrayFromImage(pred_sitk)
        img = sitk.GetArrayFromImage(img_sitk)

        # # Check orientation
        # print(f"Orientation of mask_gt: {get_direction_code(mask_gt_sitk)}")
        # print(f"Orientation of pred: {get_direction_code(pred_sitk)}")
        # print(f"Orientation of img: {get_direction_code(img_sitk)}")

        # Check dice score
        intersection = np.sum((mask_gt > 0) & (pred > 0))
        dice = 2 * intersection / (np.sum(mask_gt > 0) + np.sum(pred > 0))
        
        print(f"Dice score for subject {subject} is: {dice}")
        
        dice_list.append(dice)

        mask_gt_sitk2 = sitk.GetImageFromArray(mask_gt)
        pred_sitk2 = sitk.GetImageFromArray(pred)
        img_sitk2 = sitk.GetImageFromArray(img)

        # Give default metadata
        mask_gt_sitk2.SetOrigin((0, 0, 0))
        mask_gt_sitk2.SetSpacing((1, 1, 1))
        mask_gt_sitk2.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
        pred_sitk2.SetOrigin((0, 0, 0))
        pred_sitk2.SetSpacing((1, 1, 1))
        pred_sitk2.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
        img_sitk2.SetOrigin((0, 0, 0))
        img_sitk2.SetSpacing((1, 1, 1))
        img_sitk2.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

        mask_gt_savepath = os.path.join(outputfolder, f"{subject}_label.nii.gz")
        pred_savepath = os.path.join(outputfolder, f"{subject}_pred.nii.gz")
        img_savepath = os.path.join(outputfolder, f"{subject}_img.nii.gz")

        sitk.WriteImage(mask_gt_sitk2, mask_gt_savepath)
        sitk.WriteImage(pred_sitk2, pred_savepath)
        sitk.WriteImage(img_sitk2, img_savepath)


print(f"Mean dice score: {sum(dice_list) / len(dice_list)}")
print(f"Median dice score: {sorted(dice_list)[len(dice_list) // 2]}")
print(f"Max dice score: {max(dice_list)}")
print(f"Min dice score: {min(dice_list)}")