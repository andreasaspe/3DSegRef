import os
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

data_folder = "/scratch/awias/data/Totalsegmentator_dataset_v201_filtered_liver"

dataset_name_and_id = "Dataset003_TotalSegmentatorLiver"

output_image_tr_path = f"/scratch/awias/data/nnUNet/nnUNet_raw/{dataset_name_and_id}/imagesTr"
output_label_tr_path = f"/scratch/awias/data/nnUNet/nnUNet_raw/{dataset_name_and_id}/labelsTr"
output_image_ts_path = f"/scratch/awias/data/nnUNet/nnUNet_raw/{dataset_name_and_id}/imagesTs"
output_label_ts_path = f"/scratch/awias/data/nnUNet/nnUNet_raw/{dataset_name_and_id}/labelsTs"

all_subjects = [x.split('_')[0] for x in os.listdir(data_folder) if x.endswith("img.nii.gz")]

# all_subjects_numbers = [str(int(s[1:])) for s in all_subjects]

train_subjects, test_subjects = train_test_split(all_subjects, test_size=0.2, random_state=42)

print(f"Number of training subjects: {len(train_subjects)}")
print(f"Number of test subjects: {len(test_subjects)}")

os.makedirs(output_image_tr_path, exist_ok=True)
os.makedirs(output_label_tr_path, exist_ok=True)
os.makedirs(output_image_ts_path, exist_ok=True)
os.makedirs(output_label_ts_path, exist_ok=True)

def copy_files(subjects, image_dest, label_dest):
    for subject in tqdm(subjects):
        image_src = os.path.join(data_folder, f"{subject}_img.nii.gz")
        label_src = os.path.join(data_folder, f"{subject}_liver.nii.gz")
        subject_number = str(int(subject[1:]))
        image_dst = os.path.join(image_dest, f"{subject_number}_0000.nii.gz")
        label_dst = os.path.join(label_dest, f"{subject_number}.nii.gz")
        if os.path.exists(image_src) and os.path.exists(label_src):
            shutil.copy2(image_src, image_dst)
            shutil.copy2(label_src, label_dst)
        else:
            print(f"Missing files for subject {subject}")

copy_files(train_subjects, output_image_tr_path, output_label_tr_path)
copy_files(test_subjects, output_image_ts_path, output_label_ts_path)