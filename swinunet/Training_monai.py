import os
from glob import glob
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from monai.data import Dataset, CacheDataset, PersistentDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRanged, CropForegroundd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    RandShiftIntensityd, EnsureTyped, AsDiscrete
)
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference


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


def get_train_transforms(roi_size=(96, 96, 96), num_samples=4):
    """
    Training transforms with patch-based sampling.
    
    Args:
        roi_size: Size of patches to extract (D, H, W)
        num_samples: Number of patches to sample per image
    
    Returns:
        MONAI Compose transform
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 1.5),  # Adjust to your dataset
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175.0,  # Adjust based on your modality (CT/MRI)
            a_max=250.0,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # KEY TRANSFORM: Random patch sampling
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            pos=1,  # 50% patches centered on foreground
            neg=1,  # 50% patches from background
            num_samples=num_samples,  # Number of patches per image
            image_key="image",
            image_threshold=0
        ),
        # Data augmentation
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        EnsureTyped(keys=["image", "label"])
    ])


def get_val_transforms():
    """
    Validation transforms - no augmentation, no patch sampling.
    Full images will be processed with sliding window during inference.
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175.0,
            a_max=250.0,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"])
    ])


def create_dataloaders(
    data_dir,
    batch_size=1,
    num_workers=0,
    train_val_split=0.8,
    roi_size=(96, 96, 96),
    num_samples=4,
    cache_rate=1.0,
    use_persistent_cache=False,
    cache_dir=None
):
    """
    Create dataloaders with patch-based training.
    
    Args:
        data_dir: Path to nnUNet dataset directory
        batch_size: Batch size (number of patches per batch)
        num_workers: Number of workers for data loading
        train_val_split: Fraction of data to use for training
        roi_size: Size of patches (D, H, W)
        num_samples: Number of patches to sample per image
        cache_rate: Fraction of dataset to cache in RAM (0.0 to 1.0)
        use_persistent_cache: Use PersistentDataset for disk caching
        cache_dir: Directory for persistent cache
    
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
    
    # Get transforms
    train_transforms = get_train_transforms(roi_size=roi_size, num_samples=num_samples)
    val_transforms = get_val_transforms()
    
    # Create datasets with caching
    if use_persistent_cache and cache_dir:
        print("Using PersistentDataset (disk cache)")
        train_ds = PersistentDataset(
            data=train_files,
            transform=train_transforms,
            cache_dir=os.path.join(cache_dir, "train")
        )
        val_ds = PersistentDataset(
            data=val_files,
            transform=val_transforms,
            cache_dir=os.path.join(cache_dir, "val")
        )
    elif cache_rate > 0:
        print(f"Using CacheDataset (RAM cache, rate={cache_rate})")
        train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=cache_rate,
            num_workers=num_workers
        )
        val_ds = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_rate=cache_rate,
            num_workers=num_workers
        )
    else:
        print("Using standard Dataset (no caching)")
        train_ds = Dataset(data=train_files, transform=train_transforms)
        val_ds = Dataset(data=val_files, transform=val_transforms)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # Process one full image at a time
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


def train_epoch(model, loader, optimizer, loss_function, device, scaler=None):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0
    step = 0
    
    for batch_data in tqdm(loader, desc="Training"):
        image = batch_data["image"].to(device)
        label = batch_data["label"].to(device)
        
        print("HEY")
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(image)
                loss = loss_function(outputs, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(image)
            loss = loss_function(outputs, label)
            loss.backward()
            optimizer.step()
        
        epoch_loss += loss.item()
        step += 1
    
    return epoch_loss / step


def validate(model, loader, dice_metric, device, roi_size=(96, 96, 96), overlap=0.5):
    """
    Validate using sliding window inference for full images.
    
    Args:
        model: Trained model
        loader: Validation dataloader
        dice_metric: MONAI DiceMetric
        device: Device to run on
        roi_size: Patch size for sliding window
        overlap: Overlap between patches (0.0 to 1.0)
    """
    model.eval()
    
    with torch.no_grad():
        for batch_data in tqdm(loader, desc="Validation"):
            image = batch_data["image"].to(device)
            label = batch_data["label"].to(device)
            
            # Sliding window inference for full image
            outputs = sliding_window_inference(
                inputs=image,
                roi_size=roi_size,
                sw_batch_size=4,  # Number of patches to process at once
                predictor=model,
                overlap=overlap,
                mode="gaussian"  # Gaussian weighting for overlapping regions
            )
            
            # Convert to discrete predictions
            outputs = torch.argmax(outputs, dim=1, keepdim=True)
            label = torch.argmax(label, dim=1, keepdim=True) if label.shape[1] > 1 else label
            
            dice_metric(y_pred=outputs, y=label)
    
    metric = dice_metric.aggregate().item()
    dice_metric.reset()
    return metric


def main():
    # Configuration
    data_dir = "/home/awias/data/SwinUNETR"
    roi_size = (96, 96, 96)  # Patch size - adjust based on GPU memory
    batch_size = 1  # Number of patches per batch
    num_samples = 4  # Patches sampled per image
    num_epochs = 100
    val_interval = 2  # Validate every N epochs
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=4,
        train_val_split=0.8,
        roi_size=roi_size,
        num_samples=num_samples,
        cache_rate=0.5,  # Cache 50% in RAM
        use_persistent_cache=True,  # Set True for disk caching
        cache_dir="/home/awias/data/SwinUNETR/.cache"
    )
    
    # Model
    model = SwinUNETR(
        in_channels=1,
        out_channels=4,
        feature_size=48,  # 24, 48, or 96
        use_checkpoint=True,  # Gradient checkpointing to save memory
        spatial_dims=3
    ).to(device)
    
    # Loss and optimizer
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Training loop
    best_metric = -1
    best_metric_epoch = -1
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        epoch_loss = train_epoch(model, train_loader, optimizer, loss_function, device, scaler)
        print(f"Average Loss: {epoch_loss:.4f}")
        
        scheduler.step()
        
        # Validate
        if (epoch + 1) % val_interval == 0:
            dice_score = validate(model, val_loader, dice_metric, device, roi_size=roi_size)
            print(f"Validation Dice Score: {dice_score:.4f}")
            
            # Save best model
            if dice_score > best_metric:
                best_metric = dice_score
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_model.pth")
                print(f"Saved new best model (Dice: {best_metric:.4f})")
    
    print(f"\nTraining completed!")
    print(f"Best metric: {best_metric:.4f} at epoch {best_metric_epoch}")


if __name__ == "__main__":
    main()