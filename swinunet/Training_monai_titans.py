import os
from glob import glob
import warnings
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from monai.transforms import RandAffined
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
from monai.transforms import RandScaleIntensityd, RandGaussianNoised


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
        DivisiblePadd(keys=["image", "label"], k=32),  # Pad to multiple of 32. OBS, MIGHT RESULT IN INCONSISTENT SHAPES!
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
        SpatialPadd(
            keys=["image"],
            spatial_size=roi_size,
            mode="constant",
            value=-1.0
        ),
        SpatialPadd(
            keys=["label"],
            spatial_size=roi_size,
            mode="constant",
            value=0
        ),
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
                # ADD THESE:
        RandAffined(
            keys=["image", "label"], 
            prob=0.5,
            rotate_range=(0.2, 0.2, 0.2),  # More rotation
            scale_range=(0.1, 0.1, 0.1),   # Add scaling
            mode=("bilinear", "nearest")
        ),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.15, std=0.01),
        EnsureTyped(keys=["image", "label"])
    ])
    

def get_val_transforms(roi_size=(96, 96, 96), num_samples=2):
    """Validation transforms - no augmentation."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        SpatialPadd(
            keys=["image"],
            spatial_size=roi_size,
            mode="constant",
            value=-1.0
        ),
        SpatialPadd(
            keys=["label"],
            spatial_size=roi_size,
            mode="constant",
            value=0
        ),
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
        EnsureTyped(keys=["image", "label"])
    ])

def get_val_transforms_patches():
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


def train_epoch_patches(model, loader, optimizer, loss_fn, device):
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
                
                # print(f"Image size: {image.shape}, Label size: {label.shape}")

                optimizer.zero_grad()
                outputs = model(image)
                loss = loss_fn(outputs, label)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                step += 1
            
    return epoch_loss / step





def validate(model, loader, dice_metric, loss_fn, device, roi_size=(96, 96, 96)):
    """Validate with sliding window inference."""
    model.eval()
    
    step = 0
    
    epoch_loss = 0
    
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
            
            # Calculate validation loss
            loss = loss_fn(outputs, label)
            epoch_loss += loss.item()
            step += 1
            
            outputs = torch.argmax(outputs, dim=1, keepdim=True)
            label = torch.argmax(label, dim=1, keepdim=True) if label.shape[1] > 1 else label

            dice_metric(y_pred=outputs, y=label)

    metric = dice_metric.aggregate().item()
    dice_metric.reset()
    return epoch_loss / step, metric



def validate_patches(model, loader, dice_metric, loss_fn, device, roi_size=(96, 96, 96)):
    """Validate with sliding window inference."""
    model.eval()
    
    step = 0
    val_loss = 0
    
    with torch.no_grad():
        warnings.filterwarnings("ignore", category=UserWarning, message="Using a non-tuple sequence")
        for batch_data in tqdm(loader, desc="Validation"):            
            for patch_data in batch_data:
                image = patch_data["image"].to(device)
                label = patch_data["label"].to(device)
                
                # print(f"Image size: {image.shape}, Label size: {label.shape}")

                # Forward pass
                outputs = model(image)

                # Calculate validation loss
                loss = loss_fn(outputs, label)
                val_loss += loss.item()
                step += 1
                
                outputs = torch.argmax(outputs, dim=1, keepdim=True)
                label = torch.argmax(label, dim=1, keepdim=True) if label.shape[1] > 1 else label

                dice_metric(y_pred=outputs, y=label)

    metric = dice_metric.aggregate().item()
    dice_metric.reset()
    return val_loss / step, metric


def main():
    # Configuration
    data_dir = "/scratch/awias/data/SwinUNETR/Dataset013_TotalSegmentator_4organs/train"
    checkpoint_dir = "/scratch/awias/data/SwinUNETR/Dataset013_TotalSegmentator_4organs/checkpoints"

    # REMEMBER TO SET A RUN NAME
    run_name = "Trying_new_settings"
    print(f"\nHAVE YOU SET A NEW RUN NAME? CURRENT RUN NAME: {run_name}\n")
    
    # DEFINE STUFF
    parameters_dict = {
        'run_name': run_name,
        'description': 'Little less data augmentation. No mirroring and such. Higher weight_decay and more num_samples per epoch. Also batch size is 2. Only val per 5 epochs. Voxel size is 1.5 mm isotropic. Dataset is TotalsSegmentator4Organs.',
        'epochs': 1000,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'batch_size': 2,
        'num_samples': 2, #Samples per patches
        'patching': True, # Whether to use patch-based training
        'roi_size': (96, 96, 96),
        'val_interval': 5,
    }

    # Unpack parameters
    num_epochs = parameters_dict['epochs']
    lr = parameters_dict['learning_rate']
    wd = parameters_dict['weight_decay']
    batch_size = parameters_dict['batch_size']
    num_samples = parameters_dict['num_samples']
    patching = parameters_dict['patching']
    roi_size = parameters_dict['roi_size']
    val_interval = parameters_dict['val_interval']
    run_name = parameters_dict['run_name']
    description = parameters_dict['description']
    
    #Start wand.db
    wandb.init(
        # set the wandb project where this run will be logged
        project="SwinUNETR_ProjectWithPeter",
        entity='andreasaspe',
        name=run_name,
        notes = description,
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "epochs": num_epochs,
        'weight_decay': wd,
        'batch_size': batch_size,
        'run_name': run_name,
        }
    )
    
    # Print wandb run ID
    wandb_id = wandb.run.id
    wandb_url = wandb.run.get_url()
    
    print(f"WandB Run ID: {wandb_id}")
    print(f"WandB Run URL: {wandb_url}")

    os.makedirs(checkpoint_dir, exist_ok=True)

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
        transform=get_val_transforms(roi_size=roi_size, num_samples=10)
    )

    # Create dataloaders (num_workers=0 to avoid multiprocessing issues)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,  # Use multiple workers
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=12,  # Use multiple workers
        pin_memory=torch.cuda.is_available()
    )

    # Model
    model = SwinUNETR(
        in_channels=1,
        out_channels=5,
        feature_size=48,
        drop_rate=0.2,  # Add dropout
        use_checkpoint=True,
        spatial_dims=3
    ).to(device)

    # Loss, optimizer, scheduler
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
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
            train_loss = train_epoch_patches(model, train_loader, optimizer, loss_fn, device)
        else:
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Average Loss: {train_loss:.4f}")

        scheduler.step()
        # RandAffined(keys=["image", "label"], prob=0.3, rotate_range=(0.1, 0.1, 0.1)),

        # Validate
        if (epoch + 1) % val_interval == 0:
            # val_loss, dice_score = validate(model, val_loader, dice_metric, loss_fn, device, roi_size)
            val_loss, dice_score = validate_patches(model, val_loader, dice_metric, loss_fn, device, roi_size)
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Dice: {dice_score:.4f}")

            #Log in wandb
            wandb.log({"Train_loss": train_loss, "Validation_loss": val_loss, "epoch": epoch+1, "dice_score": dice_score})

            if dice_score > best_metric:
                best_metric = dice_score
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"{run_name}_{wandb_id}_best_model.pth"))
                print(f"✓ Saved new best model!")
                

    print(f"\n{'='*50}")
    print(f"Training completed! Best Dice: {best_metric:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()