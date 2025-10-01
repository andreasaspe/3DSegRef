import os
import SimpleITK as sitk

path =  "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset001_Verse20/labelsTr"

for root, dirs, files in os.walk(path):
    for filename in files:
        if filename.endswith(".nii.gz"):
            file_path = os.path.join(root, filename)
                        
            image = sitk.ReadImage(file_path)
            array = sitk.GetArrayFromImage(image)
            unique_values = set(array.flatten())
            print(f"{filename}: {unique_values}")