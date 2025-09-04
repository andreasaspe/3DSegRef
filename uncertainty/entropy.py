import nibabel as nib
import numpy as np
import os

path_root = "/home/awias/data/nnUNet/nnUNet_results/Dataset004_TotalSegmentatorPancreas/pred/"
path_basis = os.path.join(path_root, "basis_model_only")
path_stochastic = os.path.join(path_root, "variance")

filename_basis = [x for x in os.listdir(path_basis) if 'entropy' in x][0]
filename_stochastic = [x for x in os.listdir(path_stochastic) if 'entropy' in x][0]

img_basis = nib.load(os.path.join(path_basis, filename_basis))
img_stochastic = nib.load(os.path.join(path_stochastic, filename_stochastic))

data_basis = img_basis.get_fdata()
data_stochastic = img_stochastic.get_fdata()

data_diff = data_stochastic - data_basis

img_diff = nib.Nifti1Image(data_diff, img_basis.affine, img_basis.header)
output_path = os.path.join(path_root, "entropy_diff.nii.gz")
nib.save(img_diff, output_path)