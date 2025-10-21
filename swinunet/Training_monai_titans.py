import os
from glob import glob
import warnings
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from monai.data import Dataset
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
from monai.transforms import SpatialPadd
from monai.transforms import DivisiblePadd


def get_data_files(data_dir, train_val_split=0.8):
    """Get image and label file pairs."""
    image_files = sorted(glob(os.path.join(data_dir, "*_0000_preproc.nii.gz")))

    data_dicts = []
    for img_path in image_files:
        basename = os.path.basename(img_path)
        case_id = basename.replace("_0000_preproc.nii.gz", "")
        label_path = os.path.join(data_dir, f"{case_id}_preproc.nii.gz")

        if os.path.exists(label_path):
            data_dicts.append({"image": img_path, "label": label_path})

    # Split train/val
    train_size = int(len(data_dicts) * train_val_split)
    train_files = data_dicts[:train_size]
    val_files = data_dicts[train_size:]

    print(f"Training: {len(train_files)}, Validation: {len(val_files)}")
    return train_files, val_files


def get_train_transforms(roi_size=(96, 96, 96), num_samples=2):
    """Training transforms with patch sampling."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        DivisiblePadd(keys=["image", "label"], k=32),  # Pad to multiple of 32
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        EnsureTyped(keys=["image", "label"])
    ])
    
    
def get_train_transforms_patches(roi_size=(96, 96, 96), num_samples=2):
    """Training transforms with patch sampling."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
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
        EnsureTyped(keys=["image", "label"])
    ])

def get_val_transforms():
    """Validation transforms - no augmentation."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"])
    ])


def train_epoch(model, loader, optimizer, loss_fn, device):
    """Train one epoch."""
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in tqdm(loader, desc="Training"):
            image = batch_data["image"].to(device)
            label = batch_data["label"].to(device)
            
            # print(f"Image size: {image.shape}, Label size: {label.shape}")

            optimizer.zero_grad()
            outputs = model(image)
            loss = loss_fn(outputs, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

    return epoch_loss / step


def train_epoch_patch(model, loader, optimizer, loss_fn, device):
    """Train one epoch."""
    model.train()
    epoch_loss = 0
    step = 0
    
    idx = 0

    for batch_data in tqdm(loader, desc="Training"):
            # batch_data is a list of dicts when using RandCropByPosNegLabeld with num_samples > 1
            # Each dict contains one patch
            for patch_data in batch_data:
                image = patch_data["image"].to(device)
                label = patch_data["label"].to(device)
                
                # print(f"Image size: {image.shape}, Label size: {label.shape}")

                optimizer.zero_grad()
                outputs = model(image)
                loss = loss_fn(outputs, label)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                step += 1
            
            # if idx == 15:
            #     break
            # idx += 1

    return epoch_loss / step





def validate(model, loader, dice_metric, device, roi_size=(96, 96, 96)):
    """Validate with sliding window inference."""
    model.eval()
    
    idx = 0

    with torch.no_grad():
        warnings.filterwarnings("ignore", category=UserWarning, message="Using a non-tuple sequence")
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

            if idx == 15:
                break
            idx += 1

    metric = dice_metric.aggregate().item()
    dice_metric.reset()
    return metric


def main():
    # Configuration
    data_dir = "/scratch/awias/data/SwinUNETR/Dataset013_TotalSegmentator_4organs/train"
    roi_size = (96, 96, 96)
    batch_size = 1
    num_samples = 2  # Patches per image
    num_epochs = 100
    val_interval = 2
    patching = True  # Whether to use patch-based training

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Get data
    train_files, val_files = get_data_files(data_dir)

    # Create datasets
    if patching:
        train_ds = Dataset(
            data=train_files,
            transform=get_train_transforms_patches(roi_size=roi_size, num_samples=num_samples)
        )
    else:
        train_ds = Dataset(
            data=train_files,
            transform=get_train_transforms()
        )

    val_ds = Dataset(
        data=val_files,
        transform=get_val_transforms()
    )

    # Create dataloaders (num_workers=0 to avoid multiprocessing issues)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # Use multiple workers
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,  # Use multiple workers
        pin_memory=torch.cuda.is_available()
    )

    # Model
    model = SwinUNETR(
        in_channels=1,
        out_channels=5,
        feature_size=48,
        use_checkpoint=True,
        spatial_dims=3
    ).to(device)

    # Loss, optimizer, scheduler
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Training loop
    best_metric = -1

    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")

        # Train
        if patching:
            epoch_loss = train_epoch_patch(model, train_loader, optimizer, loss_fn, device)
        else:
            epoch_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Average Loss: {epoch_loss:.4f}")

        scheduler.step()

        # Validate
        if (epoch + 1) % val_interval == 0:
            dice_score = validate(model, val_loader, dice_metric, device, roi_size)
            print(f"Validation Dice: {dice_score:.4f}")

            if dice_score > best_metric:
                # best_metric = dice_score
                # torch.save(model.state_dict(), "best_model.pth")
                print(f"✓ Saved new best model!")

    print(f"\n{'='*50}")
    print(f"Training completed! Best Dice: {best_metric:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()