# TO ORGANIZE THE FOLDERS FOR NNUNETV2 
# import necessary libraries
import os
import shutil
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# original folders:
images_dir = "/scratch/awias/data/Pancreas/images"  # images
labels_dir = "/scratch/awias/data/Pancreas/labels"  # labels
output_dir = "/scratch/awias/data/Pancreas/nnUNet_dataset/nnUNet_raw/Dataset001_Pancreas"  # new folder with the desired subfolders

# new subfolders
images_tr_dir = os.path.join(output_dir, "imagesTr")
images_ts_dir = os.path.join(output_dir, "imagesTs")
labels_tr_dir = os.path.join(output_dir, "labelsTr")
labels_ts_dir = os.path.join(output_dir, "labelsTs")
os.makedirs(images_tr_dir, exist_ok=True)
os.makedirs(images_ts_dir, exist_ok=True)
os.makedirs(labels_tr_dir, exist_ok=True)
os.makedirs(labels_ts_dir, exist_ok=True)

# obtain IDs
label_files = [f for f in os.listdir(labels_dir) if f.endswith(".nii.gz")]
label_ids = {str(int(f.split(".")[0].replace("label", ""))) for f in label_files}
image_files = [f for f in os.listdir(images_dir) if f.endswith("nii.gz")]
image_ids = {str(int(f.split(".")[0].replace("PANCREAS_", ""))) for f in image_files}

# divide into train and test
random.seed(42) 
image_ids = sorted(image_ids)
train_ids, test_ids = train_test_split(image_ids, test_size=0.2, random_state=42)

# rename labels and images
# for image_id in tqdm(train_ids):
#     image_src = os.path.join(images_dir, f"PANCREAS_{int(image_id):04d}.nii.gz")
#     label_src = os.path.join(labels_dir, f"label{int(image_id):04d}.nii.gz")

#     if os.path.exists(image_src) and os.path.exists(label_src):
#         shutil.copy(image_src, os.path.join(images_tr_dir, f"{image_id}_0000.nii.gz"))
#         shutil.copy(label_src, os.path.join(labels_tr_dir, f"{image_id}.nii.gz"))

for image_id in tqdm(test_ids):
    image_src = os.path.join(images_dir, f"PANCREAS_{int(image_id):04d}.nii.gz")
    label_src = os.path.join(labels_dir, f"label{int(image_id):04d}.nii.gz")

    if os.path.exists(image_src) and os.path.exists(label_src):
        shutil.copy(image_src, os.path.join(images_ts_dir, f"{image_id}_0000.nii.gz"))
        shutil.copy(label_src, os.path.join(labels_ts_dir, f"{image_id}.nii.gz"))

print("Successfully organized.")
