import os
import numpy as np
import nibabel as nib
import nibabel.orientations as nio
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil
from multiprocessing import Pool, cpu_count
from functools import partial

#############################
# CONFIGURATION
#############################
datasetname = "TotalSegmentator_4organs" # target organ NOT CAPITALIZED!!!
dataset_id = 13 # integer ID for dataset
dataset_name_and_id = f"Dataset{dataset_id:03d}_{datasetname}" # Format: DatasetXXX_datasetname.

# Base directories
base_dir = f"/scratch/awias/data/Totalsegmentator_dataset_v201"
savepath_root = f"/scratch/awias/data/{dataset_name_and_id}_filtered"
nnunet_root = f"/scratch/awias/data/nnUNet/nnUNet_raw/{dataset_name_and_id}"

list_of_organs = {'pancreas': 1, 'gallbladder': 2, 'duodenum': 3, 'adrenal_gland_left': 4}

# Train/test split ratio
test_size = 0.2
random_state = 42

#############################
# HELPERS
#############################
def reorient_to(img, axcodes_to=("L", "A", "S"), verb=False):
    aff = img.affine
    ornt_fr = nio.io_orientation(aff)
    axcodes_fr = nio.ornt2axcodes(ornt_fr)
    if axcodes_to == axcodes_fr:
        return img
    ornt_to = nio.axcodes2ornt(axcodes_to)
    arr = np.asanyarray(img.dataobj, dtype=img.dataobj.dtype)
    ornt_trans = nio.ornt_transform(ornt_fr, ornt_to)
    arr = nio.apply_orientation(arr, ornt_trans)
    aff_trans = nio.inv_ornt_aff(ornt_trans, arr.shape)
    newaff = np.matmul(aff, aff_trans)
    newimg = nib.Nifti1Image(arr, newaff)
    if verb:
        print(f"[*] Image reoriented from {axcodes_fr} to {axcodes_to}")
    return newimg

def process_subject(subj, base_dir, savepath_root, new_orientation=("L","A","S")):
    try:
        ct_path = os.path.join(base_dir, subj, "ct.nii.gz")
        if not os.path.exists(ct_path):
            return f"Skipped {subj}: Missing files"
        
        combined_seg_arr = None

        for organ, organ_id in list_of_organs.items():
                seg_path = os.path.join(base_dir, subj, "segmentations", f"{organ}.nii.gz")
                seg_nib = nib.load(seg_path)
                seg_arr = seg_nib.get_fdata(dtype=np.float32)
                if np.max(seg_arr) == 1:
                    if combined_seg_arr is None:
                        combined_seg_arr = np.zeros_like(seg_arr)
                    combined_seg_arr[seg_arr == 1] = organ_id
                    
        if combined_seg_arr is None:
            return f"Skipped {subj}: No organ segmentation found"

        img_nib = nib.load(ct_path)
        arr = img_nib.get_fdata(dtype=np.float32)
        zooms = img_nib.header.get_zooms()[:3]

        # Clean affine
        new_affine = np.diag(list(zooms) + [1])
        new_affine[:3, :3] = np.eye(3) * np.array(zooms)
        new_affine[:3, 3] = 0

        new_img = nib.Nifti1Image(arr, affine=new_affine)
        new_seg = nib.Nifti1Image(combined_seg_arr, affine=new_affine)

        img_nib_reoriented = reorient_to(new_img, axcodes_to=new_orientation)
        seg_nib_reoriented = reorient_to(new_seg, axcodes_to=new_orientation)

        savepath_img = os.path.join(savepath_root, f"{subj}_img.nii.gz")
        savepath_seg = os.path.join(savepath_root, f"{subj}_msk.nii.gz")

        nib.save(img_nib_reoriented, savepath_img)
        nib.save(seg_nib_reoriented, savepath_seg)

        return f"Processed {subj}: Success"
    except Exception as e:
        return f"Error processing {subj}: {str(e)}"

#############################
# MAIN PIPELINE
#############################
def main():
    os.makedirs(savepath_root, exist_ok=True)

    subjects = sorted([d for d in os.listdir(base_dir) if d.startswith("s")])
    num_processes = max(1, cpu_count() - 1)
    # num_processes = 1 # For debugging, set to 1
    print(f"Step 1/2: Processing {len(subjects)} subjects using {num_processes} processes...")

    process_func = partial(process_subject, base_dir=base_dir, savepath_root=savepath_root)
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_func, subjects), total=len(subjects), desc="Processing subjects"))

    success_count = sum(1 for r in results if "Success" in r)
    skip_count = sum(1 for r in results if "Skipped" in r)
    error_count = sum(1 for r in results if "Error" in r)
    print(f"Finished preprocessing. Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}")

    print("\nStep 2/2: Creating nnU-Net folder structure and splitting train/test...")

    output_image_tr_path = os.path.join(nnunet_root, "imagesTr")
    output_label_tr_path = os.path.join(nnunet_root, "labelsTr")
    output_image_ts_path = os.path.join(nnunet_root, "imagesTs")
    output_label_ts_path = os.path.join(nnunet_root, "labelsTs")

    os.makedirs(output_image_tr_path, exist_ok=True)
    os.makedirs(output_label_tr_path, exist_ok=True)
    os.makedirs(output_image_ts_path, exist_ok=True)
    os.makedirs(output_label_ts_path, exist_ok=True)

    all_subjects = [x.split('_')[0] for x in os.listdir(savepath_root) if x.endswith("img.nii.gz")]
    train_subjects, test_subjects = train_test_split(all_subjects, test_size=test_size, random_state=random_state)

    print(f"Number of training subjects: {len(train_subjects)}")
    print(f"Number of test subjects: {len(test_subjects)}")

    def copy_files(subjects, image_dest, label_dest):
        for subject in tqdm(subjects):
            image_src = os.path.join(savepath_root, f"{subject}_img.nii.gz")
            label_src = os.path.join(savepath_root, f"{subject}_msk.nii.gz")
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
 
    # Optionally, remove the temporary filtered dataset folder
    shutil.rmtree(savepath_root)

    print("\nAll done! Dataset prepared for nnU-Net.")

if __name__ == "__main__":
    main()
