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


train_ds = Dataset(data=train_files)
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