#!/usr/bin/env python3
"""
LTX-Video Model Downloader
==========================
Downloads the LTX-Video model from Hugging Face for video generation.

LTX-Video is a text-to-video model from Lightricks that can generate
short video clips from text prompts.

Usage:
    python downloader_ltx.py

Requirements:
    - huggingface-hub: pip install huggingface-hub
    - ~15GB disk space for the model

Model will be saved to: /data/LTX-Video (or customize model_path below)
"""

from huggingface_hub import snapshot_download

# Configure download path - change this to your preferred location
model_path = '/data/LTX-Video'

print(f"Downloading LTX-Video model to: {model_path}")
print("This may take a while depending on your connection speed...")

snapshot_download(
    "Lightricks/LTX-Video", 
    local_dir=model_path, 
    local_dir_use_symlinks=False, 
    repo_type='model'
)

print(f"✓ LTX-Video successfully downloaded to: {model_path}")
