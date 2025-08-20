import pickle
import os

pkl_path = "/scratch/awias/data/Pancreas/nnUNet_dataset/nnUNet_preprocessed/Dataset001_Pancreas/nnUNetPlans_3d_fullres"

filename = os.path.join(pkl_path, "2.pkl")

with open(filename, "rb") as f:
    data = pickle.load(f)

print(data)