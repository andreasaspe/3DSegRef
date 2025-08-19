import os
import dicom2nifti
 
# Example usage
dicom_directory = "/scratch/awias/data/manifest-1599750808610/Pancreas-CT"
output_dir_root = "/scratch/awias/data/Pancreas"

os.makedirs(output_dir_root, exist_ok=True)

# dicom2nifti.dicom_series_to_nifti(dicom_directory, output_nifti_file, reorient_nifti=True)

idx = 0
for instance in os.listdir(dicom_directory):
    dir_path = os.path.join(dicom_directory, instance)
    if not os.path.isdir(dir_path):
        continue
    parent_folder_name = instance
    for (root,dirs,files) in os.walk(dir_path):
        if files:
            print(f"idx: {idx}")
            input_dir = root
            output_dir = os.path.join(output_dir_root, parent_folder_name+'.nii.gz')
            
            if os.path.exists(output_dir):
                idx += 1
                continue
            
            print(f"Converting {input_dir} to {output_dir}")
            dicom2nifti.dicom_series_to_nifti(root, output_dir, reorient_nifti=True)
            idx += 1