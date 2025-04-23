import torch

# Check if CUDA is available and which device is being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")

# If using GPU, print the name of the GPU
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(device)}")