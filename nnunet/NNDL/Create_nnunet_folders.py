import os

root = "/scratch/awias/data/nnUNet"
dataset_name = "Dataset001_Verse20"

os.makedirs(os.path.join(root, 'nnUNet_raw', dataset_name), exist_ok=True)
os.makedirs(os.path.join(root, 'nnUNet_preprocessed', dataset_name), exist_ok=True)
os.makedirs(os.path.join(root, 'nnUNet_results', dataset_name), exist_ok=True)

# Create subfolders in nnUNet_raw
os.makedirs(os.path.join(root, 'nnUNet_raw', dataset_name, 'imagesTr'), exist_ok=True)
os.makedirs(os.path.join(root, 'nnUNet_raw', dataset_name, 'imagesTs'), exist_ok=True)
os.makedirs(os.path.join(root, 'nnUNet_raw', dataset_name, 'labelsTr'), exist_ok=True)
os.makedirs(os.path.join(root, 'nnUNet_raw', dataset_name, 'labelsTs'), exist_ok=True)