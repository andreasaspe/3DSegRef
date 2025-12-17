import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())

device = torch.device("cuda:0")

x = torch.rand(4000, 4000, device=device)
y = torch.rand(4000, 4000, device=device)
z = torch.matmul(x, y)

print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print(f"Memory reserved:  {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
