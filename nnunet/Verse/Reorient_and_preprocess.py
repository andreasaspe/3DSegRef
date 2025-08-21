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

def reorient_sitk(img_sitk, new_direction):
    """ Reorient a SimpleITK image to a new direction.
    Args:
        img_sitk (SimpleITK.Image): Input image to be reoriented.
        new_direction (str): New direction code (e.g., 'LPS', 'RAS').
    Returns:
        SimpleITK.Image: Reoriented image.
    """

    img_sitk_reoriented = sitk.DICOMOrient(img_sitk, new_direction)
    
    return img_sitk_reoriented


image_tr_path = "/home/awias/data/nnUNet/nnUNet_raw/Dataset001_Verse20/imagesTr"
label_tr_path = "/home/awias/data/nnUNet/nnUNet_raw/Dataset001_Verse20/labelsTr"
image_ts_path = "/home/awias/data/nnUNet/nnUNet_raw/Dataset001_Verse20/imagesTs"
label_ts_path = "/home/awias/data/nnUNet/nnUNet_raw/Dataset001_Verse20/labelsTs"

all_subjects_tr = [x.split('_')[0] for x in os.listdir(image_tr_path) if x.endswith(".nii.gz")]
all_subjects_ts = [x.split('_')[0] for x in os.listdir(image_ts_path) if x.endswith(".nii.gz")]

# Training
for subject in tqdm(all_subjects_tr):
    img_path = os.path.join(image_tr_path, f"{subject}_0000.nii.gz")
    mask_path = os.path.join(label_tr_path, f"{subject}.nii.gz")

    image_sitk = sitk.ReadImage(img_path)
    mask_sitk = sitk.ReadImage(mask_path)
    
    direction_code = get_direction_code(image_sitk)
    new_orientation_code = 'LAS'

    if not direction_code == new_orientation_code:
        image_sitk_reoriented = reorient_sitk(image_sitk, new_orientation_code)
        mask_sitk_reoriented = reorient_sitk(mask_sitk, new_orientation_code)
    else:
        image_sitk_reoriented = image_sitk
        mask_sitk_reoriented = mask_sitk

    mask = sitk.GetArrayFromImage(mask_sitk_reoriented)
    mask[mask > 0] = 1  # Convert all non-zero labels to 1

    mask_sitk_reoriented_and_binary = sitk.GetImageFromArray(mask)
    mask_sitk_reoriented_and_binary.CopyInformation(mask_sitk_reoriented)

    # Save image and mask in same folder reoriented
    sitk.WriteImage(image_sitk_reoriented, img_path)
    sitk.WriteImage(mask_sitk_reoriented_and_binary, mask_path)
    
    
# Test
for subject in tqdm(all_subjects_ts):
    img_path = os.path.join(image_ts_path, f"{subject}_0000.nii.gz")
    mask_path = os.path.join(label_ts_path, f"{subject}.nii.gz")

    image_sitk = sitk.ReadImage(img_path)
    mask_sitk = sitk.ReadImage(mask_path)

    direction_code = get_direction_code(image_sitk)
    new_orientation_code = 'LAS'

    if not direction_code == new_orientation_code:
        image_sitk_reoriented = reorient_sitk(image_sitk, new_orientation_code)
        mask_sitk_reoriented = reorient_sitk(mask_sitk, new_orientation_code)
    else:
        image_sitk_reoriented = image_sitk
        mask_sitk_reoriented = mask_sitk

    mask = sitk.GetArrayFromImage(mask_sitk_reoriented)
    mask[mask > 0] = 1  # Convert all non-zero labels to 1

    mask_sitk_reoriented_and_binary = sitk.GetImageFromArray(mask)
    mask_sitk_reoriented_and_binary.CopyInformation(mask_sitk_reoriented)

    # Save image and mask in same folder reoriented
    sitk.WriteImage(image_sitk_reoriented, img_path)
    sitk.WriteImage(mask_sitk_reoriented_and_binary, mask_path)
