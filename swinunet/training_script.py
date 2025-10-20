import os
from glob import glob
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import SimpleITK as sitk
from monai.data import Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric


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
        img_sitk= sitk.ReadImage(image_path)
        img = sitk.GetArrayFromImage(img_sitk)  # (z, y, x)

        # Load label
        label_sitk = sitk.ReadImage(label_path)
        label = sitk.GetArrayFromImage(label_sitk)
        
        # Convert to float32 tensor and add channel dimension
        img_tensor = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)  # (1, z, y, x)
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
        num_workers=4, #4
        train_val_split=0.8
    )
    
    # Test loading a batch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinUNETR(
    in_channels=1,
    out_channels=4,
    feature_size=24,
    use_checkpoint=True,  # saves GPU memory
    spatial_dims=3
).to(device)

loss_function = DiceCELoss(to_onehot_y=False, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    step = 0
    
    for image, label in tqdm(train_loader):
        #Send to device
        image = image.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        outputs = model(image)
        loss = loss_function(outputs, label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        step += 1

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/step:.4f}")

model.eval()
with torch.no_grad():
    for image, label in train_loader:
        #Send to device
        image = image.to(device)
        label = label.to(device)
        
        outputs = model(image)
        dice_metric(y_pred=outputs, y=label)
    
    dice_score = dice_metric.aggregate().item()
    dice_metric.reset()
    print(f"Dice Score: {dice_score:.4f}")