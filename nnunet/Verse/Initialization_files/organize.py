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
# new subfolders
images_tr_dir = os.path.join(output_dir, "imagesTr")
images_ts_dir = os.path.join(output_dir, "imagesTs")
labels_tr_dir = os.path.join(output_dir, "labelsTr")
labels_ts_dir = os.path.join(output_dir, "labelsTs")
os.makedirs(images_tr_dir, exist_ok=True)
os.makedirs(images_ts_dir, exist_ok=True)
os.makedirs(labels_tr_dir, exist_ok=True)
os.makedirs(labels_ts_dir, exist_ok=True)


images_train = [x for x in os.listdir(data_dir_train) if x.endswith("img.nii.gz") and not x.startswith('.')]
images_val = [x for x in os.listdir(data_dir_val) if x.endswith("img.nii.gz") and not x.startswith('.')]
images_test = [x for x in os.listdir(data_dir_test) if x.endswith("img.nii.gz") and not x.startswith('.')]

mask_train = [x for x in os.listdir(data_dir_train) if x.endswith("msk.nii.gz") and not x.startswith('.')]
mask_val = [x for x in os.listdir(data_dir_val) if x.endswith("msk.nii.gz") and not x.startswith('.')]
mask_test = [x for x in os.listdir(data_dir_test) if x.endswith("msk.nii.gz") and not x.startswith('.')]

images_train_sorted = sorted(images_train, key=lambda x: x.split('_')[0])
mask_train_sorted = sorted(mask_train, key=lambda x: x.split('_')[0])
images_val_sorted = sorted(images_val, key=lambda x: x.split('_')[0])
mask_val_sorted = sorted(mask_val, key=lambda x: x.split('_')[0])
images_test_sorted = sorted(images_test, key=lambda x: x.split('_')[0])
mask_test_sorted = sorted(mask_test, key=lambda x: x.split('_')[0])

idx = 1

# Copy files into right folders
for i in tqdm(range(len(images_train_sorted))):

    subject_image = images_train_sorted[i].split('_')[0]
    subject_mask = mask_train_sorted[i].split('_')[0]

    assert subject_image == subject_mask, "Image and mask IDs do not match"

    image_src = os.path.join(data_dir_train, images_train_sorted[i])
    label_src = os.path.join(data_dir_train, mask_train_sorted[i])

    if os.path.exists(image_src) and os.path.exists(label_src):
        shutil.copy(image_src, os.path.join(images_tr_dir, f"{idx}_0000.nii.gz"))
        shutil.copy(label_src, os.path.join(labels_tr_dir, f"{idx}.nii.gz"))

        idx += 1

for i in tqdm(range(len(images_val_sorted))):

    subject_image = images_val_sorted[i].split('_')[0]
    subject_mask = mask_val_sorted[i].split('_')[0]

    assert subject_image == subject_mask, "Image and mask IDs do not match"

    image_src = os.path.join(data_dir_val, images_val_sorted[i])
    label_src = os.path.join(data_dir_val, mask_val_sorted[i])

    if os.path.exists(image_src) and os.path.exists(label_src):
        shutil.copy(image_src, os.path.join(images_tr_dir, f"{idx}_0000.nii.gz"))
        shutil.copy(label_src, os.path.join(labels_tr_dir, f"{idx}.nii.gz"))

        idx += 1

for i in tqdm(range(len(images_test_sorted))):

    subject_image = images_test_sorted[i].split('_')[0]
    subject_mask = mask_test_sorted[i].split('_')[0]

    assert subject_image == subject_mask, "Image and mask IDs do not match"

    image_src = os.path.join(data_dir_test, images_test_sorted[i])
    label_src = os.path.join(data_dir_test, mask_test_sorted[i])

    if os.path.exists(image_src) and os.path.exists(label_src):
        shutil.copy(image_src, os.path.join(images_ts_dir, f"{idx}_0000.nii.gz"))
        shutil.copy(label_src, os.path.join(labels_ts_dir, f"{idx}.nii.gz"))

        idx += 1

print("Successfully organized.")
