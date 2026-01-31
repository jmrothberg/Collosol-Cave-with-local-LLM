#!/usr/bin/env python3
"""
FLUX.1-dev Image Generation Test
=================================
Test script for FLUX.1-dev diffusion model from Black Forest Labs.

FLUX is a state-of-the-art text-to-image model that generates high-quality
images from text prompts. This script tests local model loading and generation.

Requirements:
    - Python 3.10 or 3.11
    - pip install torch torchvision torchaudio
    - pip install diffusers transformers accelerate safetensors huggingface-hub
    - FLUX model downloaded to /data/Diffusion_Models/FLUX.1-dev
    - HF_TOKEN environment variable set (optional, for gated model access)

Usage:
    export HF_TOKEN=your_token_here  # Optional
    python flux1_test_mini_dec_19_24.py

Output:
    Generates flux-dev.png in current directory
"""

import torch
from diffusers import FluxPipeline
import os
model_path = '/data/Diffusion_Models/FLUX.1-dev'
print("Files in model directory:")
for root, dirs, files in os.walk(model_path):
    print(f"\nDirectory: {root}")
    for file in files:
        print(f"  {file}")
from huggingface_hub import login
# Login to Hugging Face using environment variable (set HF_TOKEN with your token)
# Get your token from: https://huggingface.co/settings/tokens
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    try:
        login(hf_token)
        print("Successfully logged in to Hugging Face")
    except Exception as e:
        print(f"Warning: Could not login to Hugging Face: {e}")
else:
    print("Note: HF_TOKEN not set. Set environment variable for Hugging Face access.")
# Verify CUDA availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

#pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
#pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power



pipe = FluxPipeline.from_pretrained(
    '/data/Diffusion_Models/FLUX.1-dev',
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    use_safetensors=True
)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-dev.png")
#show the image
image.show()