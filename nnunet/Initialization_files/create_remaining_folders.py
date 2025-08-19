import os

root = "/scratch/awias/data/Pancreas/nnUNet_dataset"
dataset_name = "Dataset001_Pancreas"

os.makedirs(os.path.join(root, "nnUNet_preprocessed", dataset_name), exist_ok=True)
os.makedirs(os.path.join(root, "nnUNet_results", dataset_name), exist_ok=True)
