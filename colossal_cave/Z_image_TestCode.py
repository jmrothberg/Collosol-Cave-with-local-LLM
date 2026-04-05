#!/usr/bin/env python3
"""
Z-Image-Turbo Test and Integration Script
==========================================
Test script for Z-Image-Turbo diffusion model and LMM Adventure integration.

Z-Image-Turbo is an optimized image generation model that produces high-quality
images with fewer inference steps, making it ideal for interactive applications
like the adventure game.

Features:
    - Fast generation (~9 inference steps)
    - Interactive prompt loop
    - Auto-saves to Generated_Art/ folder
    - Tests LMM Adventure integration

Requirements:
    - pip install torch diffusers
    - Z-Image-Turbo model at /home/jonathan/Colossal_Cave/Z-Image-Turbo
      OR download from Hugging Face: Tongyi-MAI/Z-Image-Turbo
    - CUDA GPU recommended (bfloat16 support)

Usage:
    python Z_image_TestCode.py
    
    Then type prompts to generate images, or:
    - 'test' to verify LMM Adventure integration
    - 'quit' or 'exit' to stop
"""

import torch
from diffusers import ZImagePipeline
import os
from datetime import datetime
import re

# 1. Load the pipeline once at startup for efficiency
print("Loading Z-Image pipeline...")
# Use bfloat16 for optimal performance on supported GPUs
pipe = ZImagePipeline.from_pretrained(
    '/home/jonathan/Colossal_Cave/Z-Image-Turbo',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

# [Optional] Attention Backend
# Diffusers uses SDPA by default. Switch to Flash Attention for better efficiency if supported:
pipe.transformer.set_attention_backend("flash")    # Enable Flash-Attention-2
# pipe.transformer.set_attention_backend("_flash_3") # Enable Flash-Attention-3 (not available in this diffusers version)

# [Optional] Model Compilation
# Compiling the DiT model accelerates inference, but the first run will take longer to compile.
# pipe.transformer.compile()

# [Optional] CPU Offloading
# Enable CPU offloading for memory-constrained devices.
# pipe.enable_model_cpu_offload()

print("Pipeline loaded successfully!")

def create_short_filename(prompt, max_length=30):
    """Create a short, filesystem-safe filename from the prompt."""
    # Remove non-alphanumeric characters and replace spaces with underscores
    clean_prompt = re.sub(r'[^\w\s-]', '', prompt)
    clean_prompt = re.sub(r'\s+', '_', clean_prompt)

    # Truncate if too long
    if len(clean_prompt) > max_length:
        clean_prompt = clean_prompt[:max_length].rstrip('_')

    return clean_prompt

def generate_and_save_image(prompt):
    """Generate an image from prompt and save it with timestamp."""
    print(f"Generating image for: {prompt}")

    # Generate image
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=9,  # This actually results in 8 DiT forwards
        guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]

    # Create Generated_Art directory if it doesn't exist
    output_dir = "Generated_Art"
    os.makedirs(output_dir, exist_ok=True)

    # Create filename: short_prompt_YYYYMMDD_HHMMSS.png
    short_prompt = create_short_filename(prompt)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{short_prompt}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    # Save image
    image.save(filepath)
    print(f"Image saved as: {filepath}")

    # Display image
    image.show()

    return filepath

# Test Z-Image integration with LMM Adventure (now uses Z-Image by default)
def test_lmm_integration():
    """Test that LMM Adventure can use Z-Image (now the default image generator)"""
    try:
        print("\n🧪 Testing LMM Adventure's Z-Image integration...")
        print("LMM Adventure now uses Z-Image-Turbo by default for all images.")

        # Import the ImageGenerator from LMM Adventure
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))

        from LMM_adventure_sept_2_25 import ImageGenerator

        print("📦 Creating ImageGenerator (Z-Image is now hardcoded)...")

        # Z-Image is now the default - no parameters needed
        image_gen = ImageGenerator()

        # Test image generation with same settings as your standalone code
        test_prompt = "A mysterious cave entrance with glowing crystals"
        print(f"🎨 Generating test image: '{test_prompt}'")
        print("(This uses LMM Adventure's simplified Z-Image-only ImageGenerator)")

        result_path = image_gen.generate(test_prompt)
        if result_path:
            print(f"✅ SUCCESS! LMM Adventure Z-Image integration works!")
            print(f"📁 Image saved to: {result_path}")
            print("🎮 LMM Adventure is now ready with Z-Image support!")
            return True
        else:
            print("❌ FAILED - ImageGenerator couldn't generate image")
            return False

    except Exception as e:
        print(f"❌ FAILED with error: {e}")
        return False

# Main interactive loop
print("\nWelcome to Z-Image Generator!")
print("Enter your prompts to generate images. Type 'quit' or 'exit' to stop.")
print("Type 'test' to verify LMM Adventure's Z-Image integration works.\n")

while True:
    try:
        prompt = input("Enter image prompt: ").strip()

        if prompt.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if prompt.lower() == 'test':
            test_lmm_integration()
            continue

        if not prompt:
            print("Please enter a valid prompt.")
            continue

        # Generate and save the image
        generate_and_save_image(prompt)

        print("\nReady for next image!\n")

    except KeyboardInterrupt:
        print("\n\nGeneration interrupted. Goodbye!")
        break
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Continuing...\n")
