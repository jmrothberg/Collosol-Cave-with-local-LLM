#!/usr/bin/env python3
"""
Pyramid Flow Miniflux Model Downloader
======================================
Quick script to download just the Pyramid-Flow-Miniflux model.

This is the recommended video generation model for:
    - Mac with Apple Silicon (MPS acceleration)
    - GPUs with limited VRAM (16-24GB)
    - Faster inference times

Usage:
    python miniflux_download.py

Requirements:
    - huggingface-hub: pip install huggingface-hub
    - ~8GB disk space
"""

from huggingface_hub import snapshot_download

# Configure download path - change this to your preferred location
model_path = '/data/pyramid-flow-miniflux'

print(f"Downloading Pyramid-Flow-Miniflux to: {model_path}")
print("Model size: ~8GB - this may take a few minutes...")

snapshot_download(
    "rain1011/pyramid-flow-miniflux", 
    local_dir=model_path, 
    local_dir_use_symlinks=False, 
    repo_type='model'
)

print(f"✓ Pyramid-Flow-Miniflux successfully downloaded to: {model_path}")
