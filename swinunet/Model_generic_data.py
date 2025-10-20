import torch
from torch.utils.data import DataLoader
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import Dataset
from monai.transforms import Compose, AddChanneld, ToTensord, RandSpatialCropd

# -----------------------------
# 1️⃣ Generate synthetic 3D dataset
# -----------------------------
def generate_synthetic_data(num_samples=10, img_size=(96,96,96), num_classes=4):
    data = []
    for _ in range(num_samples):
        img = torch.randn(*img_size)  # single-channel image
        lbl = torch.randint(0, num_classes, img_size)  # integer labels
        data.append({"image": img, "label": lbl})
    return data

train_files = generate_synthetic_data(num_samples=20)

# -----------------------------
# 2️⃣ Define transforms
# -----------------------------
train_transforms = Compose([
    AddChanneld(keys=["image", "label"]),  # ensures channel dim is present
    RandSpatialCropd(keys=["image", "label"], roi_size=(96,96,96), random_size=False),
    ToTensord(keys=["image", "label"]),
])

# -----------------------------
# 3️⃣ Dataset and DataLoader
# -----------------------------
train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

# -----------------------------
# 4️⃣ Model, loss, optimizer
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinUNETR(
    in_channels=1,
    out_channels=4,  # 4-class segmentation
    feature_size=48,
    use_checkpoint=True,
    spatial_dims=3
).to(device)

loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

# -----------------------------
# 5️⃣ Training loop
# -----------------------------
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    step = 0

    for batch in train_loader:
        inputs, labels = batch["image"].to(device), batch["label"].to(device)
        
        # Ensure labels have a single channel
        if labels.shape[1] != 1:
            labels = labels.unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        step += 1

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/step:.4f}")

# -----------------------------
# 6️⃣ Optional: Evaluation Dice score
# -----------------------------
model.eval()
with torch.no_grad():
    for batch in train_loader:
        inputs, labels = batch["image"].to(device), batch["label"].to(device)
        if labels.shape[1] != 1:
            labels = labels.unsqueeze(1)
        outputs = model(inputs)
        dice_metric(y_pred=outputs, y=labels)

    dice_score = dice_metric.aggregate().item()
    dice_metric.reset()
    print(f"Dice Score: {dice_score:.4f}")
