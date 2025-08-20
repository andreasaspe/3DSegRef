# TO ORGANIZE THE FOLDERS FOR NNUNETV2 
# import necessary libraries
import os
import shutil
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# original folders:
data_dir_train = "/scratch/awias/data/Verse/Data/Verse20/Verse20_training_unpacked"  # images
data_dir_val = "/scratch/awias/data/Verse/Data/Verse20/Verse20_validation_unpacked"  # images
data_dir_test = "/scratch/awias/data/Verse/Data/Verse20/Verse20_test_unpacked"  # images

output_dir = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset001_Verse20"  # new folder with the desired subfolders

images_train = [x for x in os.listdir(data_dir_train) if x.endswith("img.nii.gz") and not x.startswith('.')]
images_val = [x for x in os.listdir(data_dir_val) if x.endswith("img.nii.gz") and not x.startswith('.')]
images_test = [x for x in os.listdir(data_dir_test) if x.endswith("img.nii.gz") and not x.startswith('.')]

# Copy files into right folders
for file in tqdm(images_train):
    image_src = os.path.join(data_dir_train, }_img.nii.gz")
    label_src = os.path.join(labels_dir, f"label{int(subject):04d}.nii.gz")

    if os.path.exists(image_src) and os.path.exists(label_src):
        shutil.copy(image_src, os.path.join(images_tr_dir, f"{image_id}_0000.nii.gz"))
        shutil.copy(label_src, os.path.join(labels_tr_dir, f"{image_id}.nii.gz"))

for image_id in tqdm(test_ids):
    image_src = os.path.join(images_dir, f"PANCREAS_{int(image_id):04d}.nii.gz")
    label_src = os.path.join(labels_dir, f"label{int(image_id):04d}.nii.gz")

    if os.path.exists(image_src) and os.path.exists(label_src):
        shutil.copy(image_src, os.path.join(images_ts_dir, f"{image_id}_0000.nii.gz"))
        shutil.copy(label_src, os.path.join(labels_ts_dir, f"{image_id}.nii.gz"))

print("Successfully organized.")
