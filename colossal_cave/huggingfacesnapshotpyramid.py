#!/usr/bin/env python3
"""
Pyramid Flow Model Downloader
=============================
Downloads Pyramid Flow models from Hugging Face for video generation.

Pyramid Flow is a text-to-video and image-to-video model that supports
multiple resolutions and can generate videos up to 10 seconds.

Available models:
    - pyramid-flow-sd3: Higher quality, larger model (~20GB)
    - pyramid-flow-miniflux: Smaller, faster model (~8GB) - DEFAULT

Usage:
    python huggingfacesnapshotpyramid.py

Requirements:
    - huggingface-hub: pip install huggingface-hub
    - Sufficient disk space for selected model
"""

from huggingface_hub import snapshot_download

# Uncomment to download the full SD3-based model (higher quality, larger)
# model_path = '/data/pyramid-flow-sd3'
# print(f"Downloading Pyramid-Flow-SD3 to: {model_path}")
# snapshot_download("rain1011/pyramid-flow-sd3", local_dir=model_path, 
#                   local_dir_use_symlinks=False, repo_type='model')

# Download the miniflux model (recommended for most users - faster, less VRAM)
model_path = '/data/pyramid-flow-miniflux'
print(f"Downloading Pyramid-Flow-Miniflux to: {model_path}")
print("This is the recommended model for most users (faster, less memory)...")

snapshot_download(
    "rain1011/pyramid-flow-miniflux", 
    local_dir=model_path, 
    local_dir_use_symlinks=False, 
    repo_type='model'
)

print(f"✓ Pyramid-Flow-Miniflux successfully downloaded to: {model_path}")
