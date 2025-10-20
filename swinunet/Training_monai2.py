import os
from glob import glob
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


def get_train_transforms(roi_size=(96, 96, 96), num_samples=2):
    """Training transforms with patch sampling."""
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


def train_epoch(model, loader, optimizer, loss_fn, device):
    """Train one epoch."""
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in tqdm(loader, desc="Training"):
            # batch_data is a list of dicts when using RandCropByPosNegLabeld with num_samples > 1
            # Each dict contains one patch
            for patch_data in batch_data:
                image = patch_data["image"].to(device)
                label = patch_data["label"].to(device)
                
                # # DEBUG: Check label values
                # unique_labels = torch.unique(label)
                # if torch.any(unique_labels >= 4) or torch.any(unique_labels < 0):
                #     print(f"WARNING: Invalid label values found: {unique_labels}")
                #     print(f"Label shape: {label.shape}, min: {label.min()}, max: {label.max()}")
                #     # Skip this batch to avoid crash
                #     continue

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
    roi_size = (96, 96, 96)
    batch_size = 1
    num_samples = 10  # Patches per image
    num_epochs = 100
    val_interval = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Get data
    train_files, val_files = get_data_files(data_dir)

    # Create datasets (NO CACHING, NO WORKERS)
    train_ds = Dataset(
        data=train_files,
        transform=get_train_transforms(roi_size=roi_size, num_samples=num_samples)
    )
    val_ds = Dataset(
        data=val_files,
        transform=get_val_transforms()
    )

    # Create dataloaders (num_workers=0 to avoid multiprocessing issues)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # NO WORKERS - process in main thread
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # NO WORKERS
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
        epoch_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Average Loss: {epoch_loss:.4f}")

        scheduler.step()

        # Validate
        if (epoch + 1) % val_interval == 0:
            dice_score = validate(model, val_loader, dice_metric, device, roi_size)
            print(f"Validation Dice: {dice_score:.4f}")

            if dice_score > best_metric:
                best_metric = dice_score
                torch.save(model.state_dict(), "best_model.pth")
                print(f"✓ Saved new best model!")

    print(f"\n{'='*50}")
    print(f"Training completed! Best Dice: {best_metric:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()