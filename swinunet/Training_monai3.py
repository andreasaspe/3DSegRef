import os
from glob import glob
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from monai.data import Dataset, PersistentDataset, list_data_collate
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRanged, CropForegroundd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    RandShiftIntensityd, EnsureTyped
)
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference


def get_data_files(data_dir, train_val_split=0.8):
    """Get image and label file pairs."""
    images_dir = os.path.join(data_dir, "imagesTr")
    labels_dir = os.path.join(data_dir, "labelsTr")
    
    image_files = sorted(glob(os.path.join(images_dir, "*_0000.nii.gz")))
    
    data_dicts = []
    for img_path in image_files:
        basename = os.path.basename(img_path)
        case_id = basename.replace("_0000.nii.gz", "")
        label_path = os.path.join(labels_dir, f"{case_id}.nii.gz")
        
        if os.path.exists(label_path):
            data_dicts.append({"image": img_path, "label": label_path})
    
    # Split train/val
    train_size = int(len(data_dicts) * train_val_split)
    train_files = data_dicts[:train_size]
    val_files = data_dicts[train_size:]
    
    print(f"Training: {len(train_files)}, Validation: {len(val_files)}")
    return train_files, val_files


def get_base_transforms():
    """
    Base preprocessing transforms - these will be CACHED to disk.
    Include everything that's deterministic and expensive.
    """
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 1.5),
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


def get_train_augmentations(roi_size=(96, 96, 96), num_samples=4):
    """
    Training augmentations - applied ON-THE-FLY (not cached).
    These are fast random transforms.
    """
    return Compose([
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            pos=1,
            neg=1,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    ])


def train_epoch(model, loader, optimizer, loss_fn, device):
    """Train one epoch."""
    model.train()
    epoch_loss = 0
    step = 0
    
    for batch_data in tqdm(loader, desc="Training"):
        image = batch_data["image"].to(device)
        label = batch_data["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(image)
        loss = loss_fn(outputs, label)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        step += 1
    
    return epoch_loss / step


def validate(model, loader, dice_metric, device, roi_size=(96, 96, 96)):
    """Validate with sliding window inference."""
    model.eval()
    
    with torch.no_grad():
        for batch_data in tqdm(loader, desc="Validation"):
            image = batch_data["image"].to(device)
            label = batch_data["label"].to(device)
            
            outputs = sliding_window_inference(
                inputs=image,
                roi_size=roi_size,
                sw_batch_size=4,
                predictor=model,
                overlap=0.5,
                mode="gaussian"
            )
            
            outputs = torch.argmax(outputs, dim=1, keepdim=True)
            label = torch.argmax(label, dim=1, keepdim=True) if label.shape[1] > 1 else label
            
            dice_metric(y_pred=outputs, y=label)
    
    metric = dice_metric.aggregate().item()
    dice_metric.reset()
    return metric


def main():
    # Configuration
    data_dir = "/home/awias/data/SwinUNETR2"
    cache_dir = "/home/awias/data/SwinUNETR2/.cache"  # Where to store cached data
    roi_size = (96, 96, 96)
    batch_size = 2
    num_samples = 4  # Patches per image
    num_epochs = 100
    val_interval = 2
    num_workers = 4  # Use multiple workers for speed
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Get data
    train_files, val_files = get_data_files(data_dir)
    
    # Create cache directories
    os.makedirs(cache_dir, exist_ok=True)
    train_cache_dir = os.path.join(cache_dir, "train_preprocessed")
    val_cache_dir = os.path.join(cache_dir, "val_preprocessed")
    
    print("\n" + "="*60)
    print("SETTING UP DATASETS WITH PERSISTENT CACHING")
    print("="*60)
    print(f"Cache directory: {cache_dir}")
    print("\nFirst run will preprocess and cache all data (slow).")
    print("Subsequent runs will load from cache (fast!).\n")
    
    # Get transforms
    base_transforms = get_base_transforms()
    train_augmentations = get_train_augmentations(roi_size=roi_size, num_samples=num_samples)
    
    # Create datasets with PERSISTENT (disk) caching
    print("Creating training dataset...")
    train_ds = PersistentDataset(
        data=train_files,
        transform=base_transforms,  # Only cache the expensive preprocessing
        cache_dir=train_cache_dir
    )
    
    print("Creating validation dataset...")
    val_ds = PersistentDataset(
        data=val_files,
        transform=base_transforms,  # Cache validation preprocessing too
        cache_dir=val_cache_dir
    )
    
    # Wrap training dataset to apply augmentations on-the-fly
    # This is a trick: PersistentDataset caches base transforms,
    # then we apply fast augmentations on top
    class AugmentedDataset(Dataset):
        def __init__(self, cached_dataset, transform):
            self.cached_dataset = cached_dataset
            self.transform = transform
        
        def __len__(self):
            return len(self.cached_dataset)
        
        def __getitem__(self, index):
            # Get preprocessed data from cache
            item = self.cached_dataset[index]
            # Apply random augmentations
            return self.transform(item)
    
    train_ds_augmented = AugmentedDataset(train_ds, train_augmentations)
    
    print("\n" + "="*60)
    print("CREATING DATALOADERS")
    print("="*60)
    
    # Create dataloaders with multiple workers for speed
    train_loader = DataLoader(
        train_ds_augmented,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=list_data_collate,
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Model
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    
    model = SwinUNETR(
        in_channels=1,
        out_channels=5,  # Adjusted for your dataset
        feature_size=48,
        use_checkpoint=True,
        spatial_dims=3
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss, optimizer, scheduler
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    best_metric = -1
    best_metric_epoch = -1
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        # Train
        epoch_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Average Loss: {epoch_loss:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        scheduler.step()
        
        # Validate
        if (epoch + 1) % val_interval == 0:
            dice_score = validate(model, val_loader, dice_metric, device, roi_size)
            print(f"Validation Dice: {dice_score:.4f}")
            
            if dice_score > best_metric:
                best_metric = dice_score
                best_metric_epoch = epoch + 1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dice_score': dice_score,
                }, "best_model.pth")
                print(f"✓ Saved new best model! (Dice: {best_metric:.4f})")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Best Dice Score: {best_metric:.4f}")
    print(f"Best Epoch: {best_metric_epoch}")
    print(f"Model saved to: best_model.pth")


if __name__ == "__main__":
    main()