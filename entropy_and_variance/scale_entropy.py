import os
import nibabel as nib
import numpy as np
from tqdm import tqdm

path = "/home/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/predictions/entropy"

output_dir = os.path.join(os.path.dirname(path), "entropy_scaled")
os.makedirs(output_dir, exist_ok=True)

for filename in tqdm(os.listdir(path)):
    if filename.startswith("1264"):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path) and filename.endswith(".nii") or filename.endswith(".nii.gz"):
            img = nib.load(file_path)
            data = img.get_fdata()
            scaled_data = data * 1000
        scaled_img = nib.Nifti1Image(scaled_data, img.affine, img.header)
        output_path = os.path.join(output_dir, filename)
        nib.save(scaled_img, output_path)