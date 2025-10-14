import torch
from torch.utils.data import DataLoader
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import Dataset
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
)

# Example: create synthetic 3D images
def generate_synthetic_data(num_samples=10, img_size=(96,96,96), num_classes=4):
    data = []
    for _ in range(num_samples):
        img = torch.randn(1, *img_size)          # 1-channel image
        lbl = torch.randint(0, num_classes, (1, *img_size))  # shape: (1, img_size)
        data.append({"image": img, "label": lbl})
    return data

train_files = generate_synthetic_data()

# Define data augmentation transforms
train_transforms = Compose([
    # Spatial transforms (applied to both image and label)
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
    # Intensity transforms (applied to image only)
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
    RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
])

train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinUNETR(
    in_channels=1,
    out_channels=4,
    feature_size=48,
    use_checkpoint=True,  # saves GPU memory
    spatial_dims=3
).to(device)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    step = 0

    for batch in train_loader:
        inputs, labels = batch["image"].to(device), batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        step += 1

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/step:.4f}")

model.eval()
with torch.no_grad():
    for batch in train_loader:
        inputs, labels = batch["image"].to(device), batch["label"].to(device)
        outputs = model(inputs)
        dice_metric(y_pred=outputs, y=labels)
    
    dice_score = dice_metric.aggregate().item()
    dice_metric.reset()
    print(f"Dice Score: {dice_score:.4f}")