from monai.networks.nets import SwinUNETR
import torch

# for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
model = SwinUNETR(in_channels=1, out_channels=4, feature_size=48)

# Input tensor: batch_size x channels x depth x height x width
x = torch.randn(1, 1, 96, 96, 96)

# Forward pass
y = model(x)

print(y.shape)