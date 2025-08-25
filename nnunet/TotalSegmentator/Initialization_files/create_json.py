# CREATE .JSON FILE:
# import libraries
import os
import json

# folders containing dataset
# root = "/scratch/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas"
root = "/home/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas"

images_tr_dir = os.path.join(root,"imagesTr")
images_ts_dir = os.path.join(root,"imagesTs")
labels_tr_dir = os.path.join(root,"labelsTr")

# lists with the corresponding files
train_images = sorted([f for f in os.listdir(images_tr_dir) if f.endswith("_0000.nii.gz")])
test_images = sorted([f for f in os.listdir(images_ts_dir) if f.endswith("_0000.nii.gz")])
train_labels = sorted([f for f in os.listdir(labels_tr_dir) if f.endswith(".nii.gz")])

# structure of the dataset.json
dataset = {
    "name": "Dataset004_TotalSegmentatorPancreas",  # change name
    "description": "Dataset for pancreas segmentation from TotalSegmentator dataset", #change description
    "reference": "",
    "licence": "",
    "release": "1.0",
    "tensorImageSize": "3D",
    "modality": {
        "0": "CT"  # change modality if needed
    },
    "channel_names": {"0":"CT"},
    "file_ending":".nii.gz",
    "labels": {
        "background": "0", #change labels if needed
        "pancreas": "1"
    },
    "numTraining": len(train_images),
    "numTest": len(test_images),
    "training": [
        {
            "image": f"./imagesTr/{img}",
            "label": f"./labelsTr/{img.split('_')[0]}.nii.gz"
        }
        for img in train_images
    ],
    "test": [
        f"./imagesTs/{img}"
        for img in test_images
    ]
}

#save dataset.json
output_file = "dataset.json"
with open(os.path.join(root,output_file), "w") as f:
    json.dump(dataset, f, indent=4)

print(f"'dataset.json' created and saved in {output_file}")

