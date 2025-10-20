import os
import SimpleITK as sitk
import numpy as np
import pickle
import tools
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

root_img = "/home/awias/data/nnUNet/nnUNet_raw/Dataset013_TotalSegmentator_4organs/imagesTs"
root_label = "/home/awias/data/nnUNet/nnUNet_raw/Dataset013_TotalSegmentator_4organs/labelsTs"
output_folder_img = "/home/awias/data/SwinUNETR/imagesTs"
output_folder_label = "/home/awias/data/SwinUNETR/labelsTs"

target_size = (512, 480, 848)
HU_range_cutoff = [-200, 500]
HU_range_normalize = [-1, 1]
pad_value = HU_range_normalize[0]
standard_dev = 0.2  # for gaussian smoothing [mm]
new_spacing = 1.5

window_level = 40
window_width = 400
lower = window_level - window_width / 2
upper = window_level + window_width / 2

os.makedirs(output_folder_img, exist_ok=True)
os.makedirs(output_folder_label, exist_ok=True)

all_subjects = [x.split("_")[0] for x in os.listdir(root_img) if x.endswith(".nii.gz")]

# Check spacing and orientation of all subjects if it is not done yet
for idx, subject in tqdm(enumerate(all_subjects)):
    img_filename = subject + "_0000.nii.gz"
    label_filename = subject + ".nii.gz"
    
    img_path = os.path.join(root_img, img_filename)
    label_path = os.path.join(root_label, label_filename)
    
    img_sitk = sitk.ReadImage(img_path)
    label_sitk = sitk.ReadImage(label_path)
    
    spacing = img_sitk.GetSpacing()

    # print("Resampling to isotropic spacing")
    img_sitk = tools.resample_to_isotropic_spacing(img_sitk, new_spacing, 'linear')
    label_sitk = tools.resample_to_isotropic_spacing(label_sitk, new_spacing, 'nearest')
    
    # print("Gaussian smoothing")
    # smoothing_filter = sitk.SmoothingRecursiveGaussianImageFilter()
    # smoothing_filter.SetSigma(standard_dev)
    # img_sitk = smoothing_filter.Execute(img_sitk)
        
    # print("Clamping")
    img_sitk = sitk.Clamp(img_sitk, lowerBound=HU_range_cutoff[0], upperBound=HU_range_cutoff[1])

    # print(f"Clamping to HU range: {lower:.1f} to {upper:.1f}")
    # img_sitk = sitk.Clamp(img_sitk, lowerBound=lower, upperBound=upper)

    # print("Normalizing")
    img_sitk = sitk.RescaleIntensity(
        img_sitk,
        outputMinimum=HU_range_normalize[0],
        outputMaximum=HU_range_normalize[1]
    )

    # print("Padding")
    img_sitk = tools.pad_to_shape(img_sitk, target_size, pad_value=pad_value)
    label_sitk = tools.pad_to_shape(label_sitk, target_size, pad_value=0)
    
    output_path_img = os.path.join(output_folder_img, img_filename)
    sitk.WriteImage(img_sitk, output_path_img)
    
    output_path_label = os.path.join(output_folder_label, label_filename)
    sitk.WriteImage(label_sitk, output_path_label)