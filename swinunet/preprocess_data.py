import os
from glob import glob
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRanged, CropForegroundd, SaveImaged, EnsureTyped
)
from monai.data import Dataset, DataLoader
from tqdm import tqdm

def preprocess(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    images_dir = os.path.join(data_dir, "imagesTr")
    labels_dir = os.path.join(data_dir, "labelsTr")

    image_files = sorted(glob(os.path.join(images_dir, "*_0000.nii.gz")))
    data_dicts = []
    for img_path in image_files:
        case_id = os.path.basename(img_path).replace("_0000.nii.gz", "")
        label_path = os.path.join(labels_dir, f"{case_id}.nii.gz")
        if os.path.exists(label_path):
            data_dicts.append({"image": img_path, "label": label_path})

    preproc = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        # Spacingd(
        #     keys=["image", "label"],
        #     pixdim=(1.5, 1.5, 1.5),
        #     mode=("bilinear", "nearest")
        # ),
        # ScaleIntensityRanged(
        #     keys=["image"],
        #     a_min=-160.0,  # -175.0,
        #     a_max=240.0,  # 250.0,
        #     b_min=-1.0,
        #     b_max=1.0,
        #     clip=True
        # ),
        # CropForegroundd(keys=["image", "label"], source_key="image"),
        # EnsureTyped(keys=["image", "label"]),
        SaveImaged(keys=["image", "label"],
                   output_dir=output_dir,
                   output_postfix="preproc",
                   output_ext=".nii.gz",
                   separate_folder=True)
    ])

    ds = Dataset(data=data_dicts, transform=preproc)
    loader = DataLoader(ds, batch_size=1, num_workers=1)

    for batch in tqdm(loader, desc="Preprocessing"):
        _ = batch  # The SaveImaged transform already saves files to disk

if __name__ == "__main__":
    preprocess("/home/awias/data/SwinUNETR2", "/home/awias/data/SwinUNETR2_preprocessed")