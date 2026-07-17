import torch
import time

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load AnyUp
print("Loading AnyUp upsampler...")
upsampler = torch.hub.load(
    "wimmerth/anyup",
    "anyup_multi_backbone",
    use_natten=True,
).to(device)
upsampler.eval()
print("AnyUp loaded successfully.")

# Create fake tensors
batch_size = 8
lr_h, lr_w = 16, 16  # Low-res patches
channels = 768  # DINO/CLIP embedding dimension
image_h, image_w = 224, 224

images = torch.randn(batch_size, 3, image_h, image_w).to(device)
lr_features = torch.randn(batch_size, channels, lr_h, lr_w).to(device)

print(f"\nTensor shapes:")
print(f"  images: {tuple(images.shape)}")
print(f"  lr_features: {tuple(lr_features.shape)}")

# Time the upsampling
print(f"\nTiming upsampler({batch_size} batches)...")
with torch.no_grad():
    start = time.perf_counter()
    hr_features = upsampler(images, lr_features)
    elapsed = time.perf_counter() - start

print(f"  HR features shape: {tuple(hr_features.shape)}")
print(f"  Time: {elapsed:.3f}s ({elapsed*1000:.1f}ms)")
print(f"  Throughput: {batch_size / elapsed:.2f} batches/sec")
