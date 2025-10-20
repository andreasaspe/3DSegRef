import os
from glob import glob
import torch
import numpy as np
from torch.utils.data import DataLoader
from monai.data import Dataset, CacheDataset, PersistentDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    SaveImaged,
    ResizeWithPadOrCropd
)


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


def get_preprocessing_transforms(pixdim=(1.5, 1.5, 1,5)):
    """
    Preprocessing transforms - applied ONCE and can be cached or saved.
    These are deterministic operations that don't change between epochs.
    
    Args:
        pixdim: Target spacing (x, y, z)
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=pixdim,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(128, 128, 128))
    ])


def get_augmentation_transforms():
    """
    Augmentation transforms - applied every epoch during training.
    These are stochastic operations that create variations.
    """
    return Compose([
        # Spatial augmentations
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
        RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.1, 0.1, 0.1),
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),
        # Intensity augmentations (image only)
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
        RandGaussianSmoothd(
            keys=["image"],
            prob=0.2,
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0)
        ),
    ])


def create_dataloaders_with_persistent_cache(
    data_dir,
    cache_dir,
    batch_size=2,
    num_workers=4,
    pixdim=(1.5, 1.5, 1.5),
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
    
    # Preprocessing transforms (saved to disk)
    preprocess_transforms = get_preprocessing_transforms(pixdim=pixdim)
    
    # Augmentation transforms (applied after loading from cache)
    augmentation_transforms = get_augmentation_transforms()
    
    # Create cache directories
    train_cache_dir = os.path.join(cache_dir, "train")
    val_cache_dir = os.path.join(cache_dir, "val")
    os.makedirs(train_cache_dir, exist_ok=True)
    os.makedirs(val_cache_dir, exist_ok=True)
    
    # Create datasets with persistent caching
    # Create datasets with persistent caching
    print("Loading/preprocessing training data...")
    train_ds = PersistentDataset(
        data=train_files,
        transform=preprocess_transforms,
        cache_dir=train_cache_dir
    )
    
    # Wrap with augmentation
    train_ds_with_aug = Dataset(data=train_ds, transform=augmentation_transforms)
    
    print("Loading/preprocessing validation data...")
    val_ds = PersistentDataset(
        data=val_files,
        transform=preprocess_transforms,
        cache_dir=val_cache_dir
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds_with_aug,
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
    data_dir = "/home/awias/data/nnUNet/nnUNet_raw/Dataset013_TotalSegmentator_4organs"
    
    
    # OPTION 2: Use PersistentDataset (for large datasets)
    # Uncomment to use this option instead:
    print("=" * 50)
    print("Option 2: PersistentDataset (saves to disk)")
    print("=" * 50)
    cache_dir = "/home/awias/data/SwinUNETR"
    train_loader, val_loader = create_dataloaders_with_persistent_cache(
        data_dir=data_dir,
        cache_dir=cache_dir,
        batch_size=2,
        num_workers=4,
        pixdim=(1.5, 1.5, 1.5),
        train_val_split=0.8
    )
    
    # Test loading a batch
    for batch in train_loader:
        images = batch["image"]
        labels = batch["label"]
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        break