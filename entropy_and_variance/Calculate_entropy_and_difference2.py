import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

path_img = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/imagesTs"
path_entropy = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/imagesTs/entropy"

all_subjects = [x.split('_')[0] for x in os.listdir(path_img) if x.endswith('.nii.gz')]
all_subjects = sorted(all_subjects, key=lambda x: int(x))

# Check if all entropy-diff files exist
all_done = True
missing_subjects = []
for subject in all_subjects:
    out_file = os.path.join(path_entropy, subject + "_entropy-diff.nii.gz")
    if not os.path.isfile(out_file):
        all_done = False
        missing_subjects.append(subject)

if not all_done:
    print("Missing files for subjects:", missing_subjects)
else:
    print("All files found. Calculating mean entropy difference...")
    mean_entropies = []
    for subject in tqdm(all_subjects):
        file_path = os.path.join(path_entropy, subject + "_entropy-diff.nii.gz")
        img = sitk.ReadImage(file_path)
        arr = sitk.GetArrayFromImage(img)
        mean_entropies.append(np.mean(arr))
        print("Min entropy:", np.min(arr), "Max entropy:", np.max(arr), "Mean entropy:", np.mean(arr))
    overall_mean = np.mean(mean_entropies)
    print("Mean entropy difference per subject:", mean_entropies)
    print("Overall mean entropy difference:", overall_mean)
