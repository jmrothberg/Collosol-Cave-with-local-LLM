"""
July 24 2025 - added code to work with pyramid flow on CUDA with 2x 24GB GPUs. 

PYRAMID FLOW VIDEO GENERATION - OPTIMIZED FOR GPU POWER

July 24 2025 - added code to work with pyramid flow on CUDA with 2x 24GB GPUs. 
Smaller resolution, 5 second videos. 

CRITICAL: NO CPU OFFLOADING unless no MPS and no CUDA available!
- With 48GB GPU memory, CPU offloading is SLOW and STUPID
- Keep everything on GPU for maximum speed
- Only use CPU offload as absolute last resort on weak systems
"""

import sys
import os
import torch
from PIL import Image
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
import glob

# Try different possible paths for Pyramid Flow (robust approach like Colossal Cave)
current_dir = os.path.dirname(os.path.abspath(__file__))
possible_paths = [
    os.path.join(current_dir, 'Pyramid-Flow'),   # Original path with dash
    os.path.join(current_dir, 'Pyramid_Flow'),   # Alternative with underscore  
    os.path.join(current_dir, 'pyramid_flow'),   # All lowercase
    current_dir  # If modules are directly in current directory
]

# Add first existing path to sys.path
pyramid_flow_dir = None
for path in possible_paths:
    if os.path.exists(path):
        pyramid_flow_dir = path
        sys.path.insert(0, path)
        print(f"Found and added to Python path: {pyramid_flow_dir}")
        break

if pyramid_flow_dir is None:
    print("Warning: No Pyramid Flow directory found!")
    print("Searched paths:")
    for path in possible_paths:
        print(f"  - {path}")

# Set model path based on device and download if needed  
from huggingface_hub import snapshot_download

# Force use of miniflux model to match 384p variant (avoid shape mismatch)
if torch.cuda.is_available():
    # Use miniflux model which matches diffusion_transformer_384p variant
    model_path = '/data/pyramid-flow-miniflux'
    if os.path.exists(model_path):
        print(f"Using miniflux model (matches 384p variant): {model_path}")
    else:
        print("CUDA miniflux model not found, downloading...")
        model_path = './pyramid-flow-miniflux'
        if not os.path.exists(model_path):
            print("Downloading Pyramid-Flow miniflux model...")
            snapshot_download("rain1011/pyramid-flow-miniflux", local_dir=model_path, local_dir_use_symlinks=False, repo_type='model')
            print("Model downloaded successfully!")
else:
    # For MPS/CPU, use local miniflux model
    model_path = './pyramid-flow-miniflux'  # Local directory to save the model
    if not os.path.exists(model_path):
        print("Downloading Pyramid-Flow miniflux model...")
        snapshot_download("rain1011/pyramid-flow-miniflux", local_dir=model_path, local_dir_use_symlinks=False, repo_type='model')
        print("Model downloaded successfully!")
    print("Using local miniflux model for MPS/CPU")

print(f"Using model path: {model_path}")

# Import after path setup
try:
    from pyramid_dit import PyramidDiTForVideoGeneration
    print("Successfully imported PyramidDiTForVideoGeneration")
except ImportError as e:
    print(f"Import error: {e}")
    if pyramid_flow_dir:
        print(f"Available modules in {pyramid_flow_dir}:")
        for item in os.listdir(pyramid_flow_dir):
            print(f"  - {item}")
    print("Please ensure Pyramid Flow is properly installed and the pyramid_dit module is available.")
    raise
from diffusers.utils import load_image, export_to_video

# Global model variables
model = None
device = None
torch_dtype = None  # Add torch_dtype as global
width = 640
height = 384

def setup_model():
    """Initialize the model once and keep it loaded"""
    global model, device, torch_dtype
    
    print("Setting up Pyramid-Flow model...")
    
    # Device selection with memory-efficient settings
    if torch.cuda.is_available():
        # Check GPU memory and select best GPU
        num_gpus = torch.cuda.device_count()
        print(f"CUDA detected! Found {num_gpus} GPUs")
        
        # Find GPU with most free memory
        best_gpu = 0
        max_free_memory = 0
        
        for gpu_id in range(num_gpus):
            free_memory, total_memory = torch.cuda.mem_get_info(gpu_id)
            free_gb = free_memory / (1024**3)
            total_gb = total_memory / (1024**3)
            print(f"GPU {gpu_id}: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
            
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu = gpu_id
        
        print(f"Selected GPU {best_gpu} with {max_free_memory/(1024**3):.1f}GB free memory")
        
        # Set the specific GPU device
        torch.cuda.set_device(best_gpu)
        device = f"cuda:{best_gpu}"
        
        # Use same memory-efficient settings as Mac but with bf16 for CUDA
        model_dtype, torch_dtype = 'bf16', torch.bfloat16
        print("Using bf16 for CUDA efficiency")
        
        # Set CUDA memory allocation for better management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
    elif torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders) with Conv3D support!")
        device = "mps"
        # Use float32 for MPS compatibility
        model_dtype, torch_dtype = 'fp32', torch.float32
        print("Using fp32 for MPS compatibility")
        
        # Set default tensor type to float32 for MPS compatibility
        torch.set_default_dtype(torch.float32)
        
        # Set MPS memory management for better stability  
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        
        # Monkey patch torch functions to avoid float64 on MPS
        original_from_numpy = torch.from_numpy
        original_arange = torch.arange
        
        def mps_safe_from_numpy(ndarray):
            tensor = original_from_numpy(ndarray)
            if tensor.dtype == torch.float64:
                tensor = tensor.float()  # Convert to float32
            return tensor
        
        def mps_safe_arange(*args, **kwargs):
            if 'dtype' in kwargs and kwargs['dtype'] == torch.float64:
                kwargs['dtype'] = torch.float32
            elif len(args) >= 4 and args[3] == torch.float64:
                args = list(args)
                args[3] = torch.float32
                args = tuple(args)
            return original_arange(*args, **kwargs)
        
        torch.from_numpy = mps_safe_from_numpy
        torch.arange = mps_safe_arange
        
    else:
        print("No GPU available, using CPU")
        device = "cpu"
        model_dtype, torch_dtype = 'fp32', torch.float32
        print("Using fp32 for CPU compatibility")

    # Use EXACT same initialization as working Colossal Cave
    model = PyramidDiTForVideoGeneration(
        model_path,
        model_name="pyramid_flux", 
        model_dtype=model_dtype,
        model_variant='diffusion_transformer_384p',
    )

    model.vae.enable_tiling()
    
    # Move models to GPU for maximum speed (NO CPU OFFLOADING!)
    if device.startswith("cuda"):
        model.vae.to(device)
        model.dit.to(device) 
        model.text_encoder.to(device)
        print(f"Models loaded on {device} for maximum speed")
    elif device == "mps":
        model.vae.to(device)
        model.dit.to(device)
        model.text_encoder.to(device)
        print(f"Models loaded on MPS for maximum speed")
    # Only if no GPU available
    else:
        print("No GPU available - models staying on CPU")
    
    print("Model setup complete!")

def select_image():
    """Let user select an image file using GUI"""
    # Create a root window but hide it
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Set the initial directory to Adventure_Art
    adventure_art_path = os.path.join(os.path.dirname(__file__), 'Adventure_Art')

    # Open file dialog for PNG selection
    print("Opening file picker dialog...")
    print(f"Looking for PNG files in: {adventure_art_path}")

    # Use the specific file as default if it exists
    target_filename = "A_darkened_room_0123-1716.png"
    target_path = os.path.join(adventure_art_path, target_filename)

    if os.path.exists(target_path):
        # Set the initial file to the target file
        initial_file = target_filename
    else:
        # Find first PNG file as fallback
        png_files = glob.glob(os.path.join(adventure_art_path, '*.png'))
        if png_files:
            initial_file = os.path.basename(png_files[0])
        else:
            initial_file = "*.png"

    # Open file dialog
    selected_file = filedialog.askopenfilename(
        title="Select an image file for video generation",
        initialdir=adventure_art_path,
        initialfile=initial_file,
        filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
    )

    # Close the root window
    root.destroy()

    if not selected_file:
        return None
    
    print(f"Selected file: {os.path.basename(selected_file)}")
    return selected_file

def get_prompt():
    """Get user prompt with default motion prompt"""
    default_prompt = "Add motion to the scene with characters moving naturally and objects behaving as they should, creating a dynamic and engaging atmosphere"
    print(f"\nDefault prompt: {default_prompt}")
    print("Press Enter to use the default prompt, or type your own prompt:")

    user_prompt = input().strip()
    if not user_prompt:
        prompt = default_prompt
        print(f"Using default prompt: {prompt}")
    else:
        prompt = user_prompt
        print(f"Using custom prompt: {prompt}")
    
    return prompt

def generate_video(image_path, prompt):
    """Generate a video from the given image and prompt"""
    global model, device, width, height
    
    if model is None:
        print("Model not initialized. Please run setup_model() first.")
        return None
    
    # Load and resize image
    image = Image.open(image_path).convert("RGB").resize((width, height))

    # Handle autocast based on device
    if device.startswith("cuda"):
        # Use CUDA autocast for efficiency
        context_manager = torch.amp.autocast(device_type="cuda", dtype=torch_dtype)
        print(f"Using CUDA autocast for generation on {device}")
    elif device == "mps":
        # MPS doesn't support autocast, just use no_grad
        context_manager = torch.no_grad()
        print("Using torch.no_grad for MPS generation")
    else:
        # Use autocast for CPU
        context_manager = torch.autocast(device_type="cpu", enabled=True, dtype=torch.float32)
        print("Using CPU autocast for generation")

    with context_manager:
        print(f"Starting video generation with {width}x{height} resolution...")
        
        # Aggressive memory cleanup before generation
        if device.startswith("cuda"):
            torch.cuda.empty_cache()  # Clear CUDA cache
            torch.cuda.ipc_collect()  # Additional cleanup
            print(f"CUDA memory after cleanup on {device}:")
            print(f"GPU memory: {torch.cuda.memory_allocated(device) / 1e9:.1f}GB allocated")
            print(f"GPU memory: {torch.cuda.memory_reserved(device) / 1e9:.1f}GB reserved")
            print("This may take several minutes on CUDA...")
        elif device == "mps":
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()  # Clear MPS cache if available
            print("MPS memory before generation:")
            import psutil
            print(f"RAM usage: {psutil.virtual_memory().percent}%")
            print("This may take several minutes on MPS...")
        
        frames = model.generate_i2v(
            prompt=prompt,
            input_image=image,
            # Current settings for testing (matching working mac_pyramid.py):
            num_inference_steps=[5, 5, 5],       # Reduced for faster testing
            temp=8,                              # Shorter video (8 frames) - safer memory usage
            
            video_guidance_scale=4.0,
            output_type="pil",
            save_memory=False,                   # False = faster, matches working mac version
        )
        
        print(f"Generated {len(frames)} frames successfully!")

    # Save video with timestamp to Pyramid_Movies directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "Pyramid_Movies"
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    video_filename = os.path.join(output_dir, f"image_to_video_sample_{timestamp}.mp4")
    print(f"Saving video as: {video_filename}")
    export_to_video(frames, video_filename, fps=24)
    
    return video_filename

def main_loop():
    """Main loop for continuous video generation"""
    print("=== Pyramid-Flow Video Generator (CUDA/MPS/CPU Support) ===")
    print("Memory-efficient settings: 640x384 resolution, 5-second videos")
    print("Models will be loaded once and kept in memory for faster generation.")
    print("Press Ctrl+C to exit.")
    print()
    
    # Setup model once
    setup_model()
    
    while True:
        try:
            print("\n" + "="*50)
            print("Select an image for video generation...")
            
            # Get image selection
            image_path = select_image()
            if image_path is None:
                print("No file selected. Exiting.")
                break
            
            # Get prompt
            prompt = get_prompt()
            
            # Generate video
            video_path = generate_video(image_path, prompt)
            
            if video_path:
                print(f"Video generated successfully: {video_path}")
            
            # Ask if user wants to continue
            print("\nGenerate another video? (y/n): ", end="")
            continue_choice = input().strip().lower()
            if continue_choice not in ['y', 'yes']:
                print("Exiting. Goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\n\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"Error during video generation: {e}")
            print("Continuing with next video...")

if __name__ == "__main__":
    main_loop()