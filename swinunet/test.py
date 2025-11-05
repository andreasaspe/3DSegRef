# test_gpu_debug.py
# import debugpy
import torch
import torch.nn as nn

# ===============================
# Step 1: Debugpy setup
# ===============================
# Listen for debugger connection on port 5678
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger to attach…")
# debugpy.wait_for_client()  # Pause here until debugger attaches

# ===============================
# Step 2: Setup GPU device
# ===============================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # ===============================
# # Step 3: Define a tiny neural network
# # ===============================
# class SimpleNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(10, 5)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(5, 2)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x

# model = SimpleNet().to(device)

# # ===============================
# # Step 4: Dummy input and forward pass
# # ===============================
# x = torch.randn(3, 10).to(device)  # batch of 3, 10 features
# output = model(x)

# # Make sure all GPU ops are done
# torch.cuda.synchronize()

# print("Forward pass output:")
# print(output)
