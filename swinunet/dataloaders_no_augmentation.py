import os
from glob import glob
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import SimpleITK as sitk
from monai.data import Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd


class NiftiDataset(Dataset):
    def __init__(self, file_paths):
        """
        Args:
            file_paths (list[str]): List of paths to NIfTI files.
        """
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        item = self.file_paths[idx]
        
        image_path = item['image']
        label_path = item['label']

        # Load image
        print("Reading image")
        img_sitk= sitk.ReadImage(image_path)
        print("Getting image array")
        img = sitk.GetArrayFromImage(img_sitk)  # (z, y, x)

        # Load label
        print("Reading label")
        label_sitk = sitk.ReadImage(label_path)
        print("Getting label array")
        label = sitk.GetArrayFromImage(label_sitk)
        
        # Convert to float32 tensor and add channel dimension
        print("To tensor image")
        img_tensor = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)  # (1, z, y, x)
        print("To tensor label")
        label_tensor = torch.from_numpy(label.astype(np.float32)).unsqueeze(0)  # (1, z, y, x)

        return img_tensor, label_tensor


def get_nnunet_datalist(data_dir, split="train"):
    """
    Parse nnUNet data format.
    
    Expected structure:
    data_dir/
        imagesTr/
            case_0000_0000.nii.gz
            case_0001_0000.nii.gz
            ...
        labelsTr/
            case_0000.nii.gz
            case_0001.nii.gz
            ...
        imagesTs/  (optional, for test set)
            ...
    
    Args:
        data_dir: Path to nnUNet dataset directory
        split: "train" or "test"
    
    Returns:
        List of dictionaries with 'image' and 'label' keys
    """
    if split == "train":
        images_dir = os.path.join(data_dir, "imagesTr")
        labels_dir = os.path.join(data_dir, "labelsTr")
    else:
        images_dir = os.path.join(data_dir, "imagesTs")
        labels_dir = os.path.join(data_dir, "labelsTs")
    
    # Get all image files (nnUNet format: casename_0000.nii.gz for single channel)
    image_files = sorted(glob(os.path.join(images_dir, "*_0000.nii.gz")))
    
    data_dicts = []
    for img_path in image_files:
        # Extract case identifier (e.g., "case_0000" from "case_0000_0000.nii.gz")
        basename = os.path.basename(img_path)
        case_id = basename.replace("_0000.nii.gz", "")
        
        # Corresponding label file
        label_path = os.path.join(labels_dir, f"{case_id}.nii.gz")
        
        if os.path.exists(label_path):
            data_dicts.append({
                "image": img_path,
                "label": label_path
            })
        else:
            print(f"Warning: Label not found for {img_path}")
    
    print(f"Found {len(data_dicts)} {split} samples")
    return data_dicts


def create_dataloaders(
    data_dir,
    batch_size=2,
    num_workers=4,
    train_val_split=0.8
):
    
    """
    Create dataloaders using PersistentDataset - preprocesses once, saves to disk.
    RECOMMENDED: For large datasets that don't fit in RAM.
    
    The first run will preprocess and save to cache_dir.
    Subsequent runs will load from cache_dir (much faster).
    
    Args:
        data_dir: Path to nnUNet dataset directory
        cache_dir: Directory to store preprocessed data
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        pixdim: Target spacing (x, y, z)
        train_val_split: Fraction of data to use for training
    
    Returns:
        train_loader, val_loader
    """
    # Get all training data
    data_dicts = get_nnunet_datalist(data_dir, split="train")
    
    # Split into train and validation
    train_size = int(len(data_dicts) * train_val_split)
    train_files = data_dicts[:train_size]
    val_files = data_dicts[train_size:]
    
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    
    # Minimal transforms - just load the files
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
    ])
    
    # Create dataset
    # train_ds = Dataset(data=train_files, transform=transforms)
    # val_ds = Dataset(data=val_files, transform=transforms)
    
    train_ds = NiftiDataset(train_files)
    val_ds = NiftiDataset(val_files)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


# Example usage:
if __name__ == "__main__":
    # Path to your nnUNet dataset
    data_dir = "/home/awias/data/SwinUNETR"
        
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=1,
        num_workers=1, #4
        train_val_split=0.8
    )
    
    # Test loading a batch
    for image, label in tqdm(train_loader):
        
        # images = batch["image"]
        # print("Here2")
        # labels = batch["label"]
        # print("Here3")
        # print(f"Image batch shape: {images.shape}")
        # print(f"Label batch shape: {labels.shape}")