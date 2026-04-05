# FLUX Image Generation with Multi-GPU Support
# This configuration WORKS with 2x 24GB RTX 4090 GPUs
# 
# KEY LEARNINGS:
# 1. Use "balanced" device mapping for model loading - DON'T change this
# 2. Use 512x512 images for generation - larger sizes cause OOM on generation
# 3. The model loads fine across GPUs, generation needs the smaller size
# 4. Clear cache before generation to free any unused memory
# 5. DON'T try to move model to single GPU - breaks the working balance
# July 23 2025 - added code to work with FLUX models on Mac MPS. 
# July 24 2025 - added code to work with FLUX models on CUDA with 2x 24GB GPUs. 

import sys
import os
import torch
from PIL import Image
from datetime import datetime
import platform

# Add the Pyramid-Flow directory to the Python path (if needed for FLUX)
pyramid_flow_path = os.path.join(os.path.dirname(__file__), 'Pyramid-Flow')
if os.path.exists(pyramid_flow_path):
    sys.path.append(pyramid_flow_path)
    print(f"Added to Python path: {pyramid_flow_path}")

# Device detection and setup
print(f"Platform: {platform.system()}")
print(f"Machine: {platform.machine()}")

# IMPORTANT: Set recommended CUDA memory allocation for fragmentation issues
# This helps with memory fragmentation on multi-GPU setups
if torch.cuda.is_available():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

# Device selection with proper dtype for each platform
if torch.backends.mps.is_available() and platform.system() == 'Darwin':
    device = torch.device("mps")
    print("Using Mac MPS (Metal Performance Shaders)")
    torch_dtype = torch.float32  # MPS works best with float32
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
    torch_dtype = torch.float16  # CUDA works well with float16 for memory efficiency
else:
    device = torch.device("cpu")
    print("Using CPU")
    torch_dtype = torch.float32

print(f"Device: {device}")
print(f"Torch dtype: {torch_dtype}")



# Check GPU memory info - CRITICAL for understanding memory distribution
if torch.cuda.is_available():
    print(f"CUDA devices available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name} - {props.total_memory / 1024**3:.2f} GB")
    
    # Clear any existing GPU memory to start fresh
    torch.cuda.empty_cache()
    print("Cleared GPU cache")

# Import FLUX pipeline
try:
    from diffusers import FluxPipeline
    print("✓ FluxPipeline imported successfully")
except ImportError as e:
    print(f"✗ Error importing FluxPipeline: {e}")
    print("Please install diffusers: pip install diffusers")
    sys.exit(1)

def setup_flux_model():
    """Set up FLUX model with device-specific optimizations
    
    WORKING CONFIGURATION for 2x 24GB GPUs:
    - Use "balanced" device mapping (DON'T change this!)
    - Try FLUX.1-schnell first (faster, smaller memory)
    - Fall back to single GPU if balanced fails
    - Use low_cpu_mem_usage=True for efficiency
    """
    print("Setting up FLUX model...")
    
    # FLUX model options - schnell is faster and uses less memory
    flux_models = [
        "black-forest-labs/FLUX.1-schnell",  # Faster option, smaller memory footprint - WORKS
        "black-forest-labs/FLUX.1-dev",      # Higher quality option, more memory
    ]
    
    # Try to load FLUX model
    for model_name in flux_models:
        try:
            print(f"Attempting to load: {model_name}")
            
            # Clear GPU memory before each attempt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
            
            if device.type == "mps":
                # Mac MPS optimizations for FLUX
                print("Applying Mac MPS optimizations for FLUX...")
                
                # MPS compatibility fixes
                torch.set_default_dtype(torch.float32)
                
                # Monkey patch torch functions for MPS compatibility
                original_from_numpy = torch.from_numpy
                def mps_safe_from_numpy(ndarray):
                    tensor = original_from_numpy(ndarray)
                    if tensor.dtype == torch.float64:
                        tensor = tensor.float()
                    return tensor
                torch.from_numpy = mps_safe_from_numpy
                
                original_arange = torch.arange
                def mps_safe_arange(*args, **kwargs):
                    if 'dtype' in kwargs and kwargs['dtype'] == torch.float64:
                        kwargs['dtype'] = torch.float32
                    return original_arange(*args, **kwargs)
                torch.arange = mps_safe_arange
                
                # Load FLUX with MPS optimizations
                pipe = FluxPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  # Use float32 for MPS
                )
                pipe = pipe.to(device)
                print(f"✓ FLUX model loaded successfully on MPS: {model_name}")
                return pipe
                
            else:
                # CUDA/CPU setup with multi-GPU support
                # CRITICAL: This configuration WORKS - don't change it!
                if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                    print(f"Using multi-GPU setup with {torch.cuda.device_count()} GPUs")
                    try:
                        # Use balanced device mapping - this was working better
                        # IMPORTANT: "balanced" properly distributes model across GPUs
                        pipe = FluxPipeline.from_pretrained(
                            model_name,
                            torch_dtype=torch_dtype,
                            device_map="balanced",  # WORKING CONFIG - don't change!
                            low_cpu_mem_usage=True,  # Memory optimization
                        )
                        print("✓ Successfully loaded with balanced device mapping")
                    except Exception as e:
                        print(f"⚠ Balanced device mapping failed: {e}")
                        print("Falling back to single GPU...")
                        # Clear ALL GPUs before fallback
                        for i in range(torch.cuda.device_count()):
                            torch.cuda.set_device(i)
                            torch.cuda.empty_cache()
                        
                        pipe = FluxPipeline.from_pretrained(
                            model_name,
                            torch_dtype=torch_dtype,
                            low_cpu_mem_usage=True,
                        )
                        pipe = pipe.to(device)
                        
                        # Enable memory optimizations for single GPU
                        try:
                            pipe.enable_attention_slicing()
                            print("✓ Enabled attention slicing for memory efficiency")
                        except:
                            pass
                else:
                    # Single GPU or CPU setup
                    pipe = FluxPipeline.from_pretrained(
                        model_name,
                        torch_dtype=torch_dtype,
                    )
                    pipe = pipe.to(device)
                print(f"✓ FLUX model loaded successfully: {model_name}")
                return pipe
                
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")
            # Clear GPU memory after failed attempt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("Cleared GPU cache after failed attempt")
            continue
    
    print("✗ Failed to load any FLUX model")
    return None

def generate_image(pipe, prompt, output_path=None):
    """Generate an image using FLUX
    
    WORKING CONFIGURATION:
    - Use 512x512 images (CRITICAL - larger sizes cause OOM during generation)
    - Clear cache before generation
    - Use 4 steps for FLUX (standard)
    - guidance_scale=0.0 for FLUX (standard)
    """
    if pipe is None:
        print("✗ No FLUX pipeline available")
        return None
    
    print(f"Generating image with prompt: {prompt}")
    
    try:
        # FLUX-specific settings - these are standard for FLUX
        if isinstance(pipe, FluxPipeline):
            num_inference_steps = 4  # FLUX uses fewer steps
            guidance_scale = 0.0     # FLUX doesn't use guidance
        else:
            num_inference_steps = 25
            guidance_scale = 7.0
        
        print(f"Using {num_inference_steps} inference steps, guidance_scale={guidance_scale}")
        
        # Generate image with memory monitoring
        if torch.cuda.is_available():
            print("GPU memory before generation:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                print(f"  GPU {i}: {allocated:.2f}GB allocated")
            # IMPORTANT: Clear cache before generation to free up any unused memory
            # This is critical for preventing OOM during generation
            torch.cuda.empty_cache()
        
        # CRITICAL: Use 512x512 to fit in available memory during generation
        # Larger sizes (768x768, 1024x1024) cause OOM with balanced mapping
        image = pipe(
            prompt=prompt,
            height=512,   # WORKING SIZE - don't increase without testing memory
            width=512,    # WORKING SIZE - don't increase without testing memory
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
        ).images[0]
        
        # Clear cache after generation to prevent memory buildup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Save image to Flux_Images directory
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "Flux_Images"
            os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
            output_path = os.path.join(output_dir, f"flux_test_{timestamp}.png")
        
        image.save(output_path)
        print(f"✓ Image saved as: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"✗ Error generating image: {e}")
        return None

def main():
    """Main function to test FLUX"""
    print("=== FLUX Image Generation Test ===")
    print("This script will generate images using FLUX. Enter 'quit' to exit.")
    
    # Set up FLUX model
    pipe = setup_flux_model()
    if pipe is None:
        print("Failed to set up FLUX model. Exiting.")
        return
    
    print(f"\n✓ FLUX model ready on {device}")
    print("Enter your image prompts below. Type 'quit' to exit.")
    print("Example prompts:")
    print("  - A beautiful sunset over mountains, digital art")
    print("  - A cozy library with ancient books, warm lighting")
    print("  - A futuristic cityscape at night, neon lights")
    print()
    
    image_count = 0
    
    while True:
        try:
            # Get user input
            prompt = input("\n🎨 Enter your image prompt (or 'quit' to exit): ").strip()
            
            # Check for quit command
            if prompt.lower() in ['quit', 'exit', 'q']:
                print(f"\n👋 Generated {image_count} images. Goodbye!")
                break
            
            # Check for empty prompt
            if not prompt:
                print("❌ Please enter a valid prompt.")
                continue
            
            print(f"\n🔄 Generating image: '{prompt}'")
            print("This may take 20-30 seconds...")
            
            # Generate image
            output_path = generate_image(pipe, prompt)
            
            if output_path:
                image_count += 1
                print(f"✅ Success! Image saved as: {output_path}")
            else:
                print("❌ Failed to generate image. Please try again.")
                
        except KeyboardInterrupt:
            print(f"\n\n👋 Generated {image_count} images. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again with a different prompt.")
    
    print("\n=== FLUX Test Complete ===")

if __name__ == "__main__":
    main() 