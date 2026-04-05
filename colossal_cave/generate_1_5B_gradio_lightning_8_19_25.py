# JMR's user interface for Wan2.2
# Updated version compatible with Wan2.2 - supports multiple model types
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import os
import sys
import logging
import random
import warnings
from datetime import datetime

# Disable flash attention to avoid binary compatibility issues
os.environ['FLASH_ATTN_DISABLE'] = '1'

# Enable PyTorch memory segmentation to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

warnings.filterwarnings('ignore')

import torch
import gradio as gr
from PIL import Image

import wan

# Fix flash attention runtime issues by monkey patching
import wan.modules.attention as attn_module

# Force disable flash attention availability
attn_module.FLASH_ATTN_2_AVAILABLE = False
attn_module.FLASH_ATTN_3_AVAILABLE = False

# Create a safe flash_attention function that never asserts
def safe_flash_attention(*args, **kwargs):
    # Always use the fallback attention function
    return attn_module.attention(*args, **kwargs)

# Replace the problematic function
attn_module.flash_attention = safe_flash_attention

# Also patch any imports in the model module
import wan.modules.model as model_module
if hasattr(model_module, 'flash_attention'):
    model_module.flash_attention = safe_flash_attention

# Import and patch all modules that might use flash_attention
import sys
for module_name, module in sys.modules.items():
    if module_name.startswith('wan.modules') and hasattr(module, 'flash_attention'):
        module.flash_attention = safe_flash_attention

from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.utils.prompt_extend import QwenPromptExpander
from wan.utils.utils import save_video, save_image

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)]
)

# Example prompts for different model types
EXAMPLE_PROMPTS = {
    "t2v-A14B": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    "i2v-A14B": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression.",
    "ti2v-5B": "A black and tan Rottweiler on the bow of a boat moving through the waves with breaching male grey humpback whales on one side fighting each other, camera behind and to the side of the dog, so you look over the dog and see the whales, sunny, clear blue skies."
}

# Default model paths - users can override these
DEFAULT_MODEL_PATHS = {
    "t2v-A14B": "/data/Wan2.2-T2V-A14B",
    "i2v-A14B": "/data/Wan2.2-I2V-A14B", 
    "ti2v-5B": "/data/Wan2.2-TI2V-5B"
}

# Lightning LoRA discovery
LIGHTNING_BASE_DIR = "/data/Wan2.2-Lightning"

def discover_lora_dirs(base_dir=LIGHTNING_BASE_DIR):
    try:
        if not os.path.isdir(base_dir):
            return ["None"]
        entries = []
        for name in sorted(os.listdir(base_dir)):
            full = os.path.join(base_dir, name)
            if os.path.isdir(full):
                # Only include dirs that look like Wan2.2 Lightning exports
                low = os.path.join(full, "low_noise_model.safetensors")
                high = os.path.join(full, "high_noise_model.safetensors")
                if os.path.isfile(low) or os.path.isfile(high):
                    entries.append(full)
        return ["None"] + entries if entries else ["None"]
    except Exception:
        return ["None"]

# Create output directory if it doesn't exist
OUTPUT_DIR = "WanVideos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variables to store models
wan_models = {}

def validate_inputs(task, size, frame_num, ckpt_dir):
    """Validate inputs similar to generate.py"""
    # Check if task is supported
    if task not in WAN_CONFIGS:
        return False, f"Unsupported task: {task}. Supported tasks: {list(WAN_CONFIGS.keys())}"
    
    # Check if checkpoint directory exists
    if not os.path.exists(ckpt_dir):
        return False, f"Model directory does not exist: {ckpt_dir}"
    
    # Check if size is supported for this task
    if size not in SUPPORTED_SIZES[task]:
        return False, f"Unsupported size {size} for task {task}. Supported sizes: {SUPPORTED_SIZES[task]}"
    
    # Check frame number (should be 4n+1 for optimal results)
    if frame_num % 4 != 1:
        logging.warning(f"Frame number {frame_num} is not optimal. Consider using 4n+1 values (5, 9, 13, etc.) for better results.")
    
    return True, ""

def load_model(task, ckpt_dir, device_id=0, offload_model=True, t5_cpu=False, convert_model_dtype=True):
    """Load the appropriate Wan2.2 model based on task"""
    global wan_models
    
    model_key = f"{task}_{ckpt_dir}"
    
    # Only load if not already loaded
    if model_key not in wan_models:
        logging.info(f"Loading Wan2.2 {task} model from {ckpt_dir}")
        logging.info(f"Memory optimizations: offload_model={offload_model}, t5_cpu={t5_cpu}, convert_model_dtype={convert_model_dtype}")
        
        # Validate inputs first
        is_valid, error_msg = validate_inputs(task, SUPPORTED_SIZES[task][0], 9, ckpt_dir)
        if not is_valid:
            raise RuntimeError(error_msg)
        
        try:
            # Get the model configuration
            cfg = WAN_CONFIGS[task]
            logging.info(f"Using model config: {cfg.__name__}")
            
            # Print some debug info about the config
            logging.info(f"VAE checkpoint: {cfg.vae_checkpoint}")
            logging.info(f"T5 checkpoint: {cfg.t5_checkpoint}")
            
            # Check for the specific VAE file
            vae_path = os.path.join(ckpt_dir, cfg.vae_checkpoint)
            if not os.path.exists(vae_path):
                raise RuntimeError(f"VAE checkpoint not found: {vae_path}")
            
            # Create the appropriate model based on task
            if "t2v" in task:
                wan_model = wan.WanT2V(
                    config=cfg,
                    checkpoint_dir=ckpt_dir,
                    device_id=device_id,
                    rank=0,
                    t5_fsdp=False,
                    dit_fsdp=False,
                    use_sp=False,  # Updated from use_usp to use_sp for wan2.2
                    t5_cpu=t5_cpu,
                    convert_model_dtype=convert_model_dtype,
                )
            elif "ti2v" in task:
                wan_model = wan.WanTI2V(
                    config=cfg,
                    checkpoint_dir=ckpt_dir,
                    device_id=device_id,
                    rank=0,
                    t5_fsdp=False,
                    dit_fsdp=False,
                    use_sp=False,
                    t5_cpu=t5_cpu,
                    convert_model_dtype=convert_model_dtype,
                )
            elif "i2v" in task:
                wan_model = wan.WanI2V(
                    config=cfg,
                    checkpoint_dir=ckpt_dir,
                    device_id=device_id,
                    rank=0,
                    t5_fsdp=False,
                    dit_fsdp=False,
                    use_sp=False,
                    t5_cpu=t5_cpu,
                    convert_model_dtype=convert_model_dtype,
                )
            else:
                raise RuntimeError(f"Unknown task type: {task}")
                
            logging.info("Model loaded successfully")
            
            # Apply aggressive memory optimizations for 24GB GPU
            logging.info("Applying memory optimizations for 24GB GPU...")
            
            # SKIP WARMUP to avoid OOM - will generate on first actual use
            logging.info("Skipping warmup to conserve GPU memory - model will initialize on first generation")
            
            wan_models[model_key] = wan_model
            
        except Exception as e:
            import traceback
            error_msg = f"Error initializing model: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)
    
    return wan_models[model_key]


def apply_lora_if_selected(model, task, lora_dir: str):
    """Apply Lightning LoRA safetensors to the loaded model if a directory is selected.

    Only applies to T2V/I2V models that have low/high noise submodels.
    """
    if not lora_dir or lora_dir == "None":
        return
    try:
        from safetensors.torch import load_file
    except Exception:
        logging.warning("safetensors not available; skipping LoRA application")
        return

    try:
        if "t2v" in task:
            low_path = os.path.join(lora_dir, "low_noise_model.safetensors")
            high_path = os.path.join(lora_dir, "high_noise_model.safetensors")
            if os.path.isfile(low_path):
                missing, unexpected = model.low_noise_model.load_state_dict(load_file(low_path), strict=False)
                logging.info(f"Applied LoRA to low_noise_model (missing={len(missing)}, unexpected={len(unexpected)})")
            if os.path.isfile(high_path):
                missing, unexpected = model.high_noise_model.load_state_dict(load_file(high_path), strict=False)
                logging.info(f"Applied LoRA to high_noise_model (missing={len(missing)}, unexpected={len(unexpected)})")
        elif "i2v" in task:
            low_path = os.path.join(lora_dir, "low_noise_model.safetensors")
            high_path = os.path.join(lora_dir, "high_noise_model.safetensors")
            if os.path.isfile(low_path):
                missing, unexpected = model.low_noise_model.load_state_dict(load_file(low_path), strict=False)
                logging.info(f"Applied LoRA to low_noise_model (missing={len(missing)}, unexpected={len(unexpected)})")
            if os.path.isfile(high_path):
                missing, unexpected = model.high_noise_model.load_state_dict(load_file(high_path), strict=False)
                logging.info(f"Applied LoRA to high_noise_model (missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            logging.info("LoRA application is currently not implemented for TI2V unified model")
    except Exception as e:
        logging.warning(f"Failed to apply LoRA from {lora_dir}: {e}")

def generate_video(
    prompt, 
    task="ti2v-5B",
    size="480*832",   # Small size for 24GB GPU
    frame_num=5,      # Minimal frames for 24GB GPU
    sample_solver="unipc",
    sample_steps=35,   # Reduce default steps for faster/less memory
    sample_shift=5.0,
    guide_scale=5.0,
    seed=-1,
    use_prompt_extend=False,
    ckpt_dir=None,
    input_image=None,
    offload_model=True,  # Enable by default for 24GB GPU
    t5_cpu=False,        # Keep T5 on GPU for speed
    convert_model_dtype=True,  # Convert model dtype to save memory
    lora_dir="None",
    progress=gr.Progress()
):
    """Generate a video using the appropriate Wan2.2 model"""
    
    # Set default checkpoint directory if not provided
    if ckpt_dir is None:
        ckpt_dir = DEFAULT_MODEL_PATHS.get(task, DEFAULT_MODEL_PATHS["ti2v-5B"])
    
    # Validate inputs
    if not prompt.strip():
        return None, "Please enter a prompt"
    
    # Validate task-specific requirements
    if "i2v" in task and input_image is None:
        return None, f"Task {task} requires an input image"
    
    is_valid, error_msg = validate_inputs(task, size, frame_num, ckpt_dir)
    if not is_valid:
        return None, error_msg
    
    if seed == -1:
        seed = random.randint(0, 2147483647)
    
    # Clear GPU cache before loading
    torch.cuda.empty_cache()
    
    # Load the model if not already loaded
    try:
        model = load_model(task, ckpt_dir, 
                          offload_model=offload_model, 
                          t5_cpu=t5_cpu, 
                          convert_model_dtype=convert_model_dtype)
        logging.info(f"Model ready for generation")
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        logging.error(error_msg)
        return None, error_msg

    # Apply LoRA, if any
    try:
        apply_lora_if_selected(model, task, lora_dir)
    except Exception as e:
        logging.warning(f"LoRA apply failed: {e}")
    
    # Process the prompt if extension is enabled
    original_prompt = prompt
    prompt_output = None
    if use_prompt_extend:
        progress(0.1, "Extending prompt...")
        try:
            prompt_expander = QwenPromptExpander(
                model_name=None,
                task=task,
                is_vl=(input_image is not None),
                device=0
            )
            prompt_output = prompt_expander(
                prompt, 
                image=input_image,
                tar_lang="en", 
                seed=seed
            )
            if prompt_output.status:
                extended_prompt = prompt_output.prompt
                logging.info(f"Extended prompt: {extended_prompt}")
                prompt = extended_prompt
            else:
                logging.info(f"Prompt extension failed: {prompt_output.message}")
        except Exception as e:
            logging.warning(f"Prompt extension error: {e}")
            # Continue with original prompt if extension fails
    
    # Generate the video
    progress(0.2, "Generating video...")
    try:
        logging.info(f"Starting generation:")
        logging.info(f"  Task: {task}")
        logging.info(f"  Prompt: {prompt}")
        logging.info(f"  Size: {size}")
        logging.info(f"  Frames: {frame_num}")
        logging.info(f"  Seed: {seed}")
        logging.info(f"  Memory optimizations: offload={offload_model}, t5_cpu={t5_cpu}")
        
        # Clear GPU cache before generation
        torch.cuda.empty_cache()
        
        # Generate based on task type
        if "t2v" in task:
            # CHANGE: cast guide_scale to float to avoid int being non-subscriptable in library
            video = model.generate(
                prompt,
                size=SIZE_CONFIGS[size],
                frame_num=frame_num,
                shift=sample_shift,
                sample_solver=sample_solver,
                sampling_steps=sample_steps,
                guide_scale=float(guide_scale),
                seed=seed,
                offload_model=offload_model
            )
        elif "ti2v" in task:
            # CHANGE: cast guide_scale to float to avoid int being non-subscriptable in library
            video = model.generate(
                prompt,
                img=input_image,
                size=SIZE_CONFIGS[size],
                max_area=MAX_AREA_CONFIGS[size],
                frame_num=frame_num,
                shift=sample_shift,
                sample_solver=sample_solver,
                sampling_steps=sample_steps,
                guide_scale=float(guide_scale),
                seed=seed,
                offload_model=offload_model
            )
        elif "i2v" in task:
            # CHANGE: cast guide_scale to float to avoid int being non-subscriptable in library
            video = model.generate(
                prompt,
                input_image,
                max_area=MAX_AREA_CONFIGS[size],
                frame_num=frame_num,
                shift=sample_shift,
                sample_solver=sample_solver,
                sampling_steps=sample_steps,
                guide_scale=float(guide_scale),
                seed=seed,
                offload_model=offload_model
            )
        
        # Save the video
        progress(0.8, "Saving video...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_file = os.path.join(OUTPUT_DIR, f"wan_{task}_{timestamp}.mp4")
        
        cfg = WAN_CONFIGS[task]
        save_video(
            tensor=video[None],
            save_file=save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
        
        # Build success message
        success_msg = f"Video generated successfully with {task} model using seed {seed}"
        if use_prompt_extend and prompt_output and prompt_output.status:
            success_msg += f"\n\nExtended prompt:\n{prompt}"
        
        logging.info(f"Video saved to: {save_file}")
        
        # Clean up GPU memory after generation
        del video
        torch.cuda.empty_cache()
        
        return save_file, success_msg
        
    except Exception as e:
        error_msg = f"Error generating video: {str(e)}"
        logging.error(error_msg)
        import traceback
        logging.error(traceback.format_exc())
        return None, error_msg

def update_ui_for_task(task):
    """Update UI elements based on selected task"""
    # Update prompt placeholder
    example_prompt = EXAMPLE_PROMPTS.get(task, "Describe the video you want to generate...")
    
    # Update supported sizes
    supported_sizes = SUPPORTED_SIZES[task]
    default_size = supported_sizes[0]
    
    # Show/hide image input based on task - more explicit logic
    image_visible = task in ["i2v-A14B", "ti2v-5B"]  # Show for I2V and TI2V tasks
    
    return (
        gr.update(value=example_prompt, placeholder=f"Describe the video you want to generate using {task}..."),
        gr.update(choices=supported_sizes, value=default_size),
        gr.update(visible=image_visible),  # Show/hide based on task
        gr.update(value=DEFAULT_MODEL_PATHS.get(task, DEFAULT_MODEL_PATHS["ti2v-5B"])),
        gr.update(choices=discover_lora_dirs(), value="None")
    )

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="JMR's Wan2.2 Multi-Model Video Generator") as demo:
        gr.Markdown("# JMR's Wan2.2 Multi-Model Video Generator")
        gr.Markdown("Generate videos from text prompts and/or images using Wan2.2 models (T2V-A14B, I2V-A14B, TI2V-5B)")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Task selection
                task_dropdown = gr.Dropdown(
                    label="Model Type",
                    choices=list(WAN_CONFIGS.keys()),
                    value="ti2v-5B",
                    info="Choose the model type: T2V (text-to-video), I2V (image-to-video), TI2V (text+image-to-video)"
                )
                
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the video you want to generate...",
                    value=EXAMPLE_PROMPTS["ti2v-5B"],
                    lines=3
                )
                
                # Image input (hidden by default, shown for I2V tasks)
                image_input = gr.Image(
                    label="Input Image (Required for I2V, Optional for TI2V)",
                    type="pil",
                    visible=True  # Start visible since default is TI2V which supports images
                )
                
                with gr.Row():
                    generate_btn = gr.Button("Generate Video", variant="primary")
                    clear_btn = gr.Button("Clear")
                
                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        with gr.Column():
                            size_dropdown = gr.Dropdown(
                                label="Resolution",
                                choices=SUPPORTED_SIZES["ti2v-5B"],
                                value="704*1280"  # 720P default for memory efficiency
                            )
                            frame_num = gr.Slider(
                                label="Number of Frames",
                                minimum=5,
                                maximum=121,  # Further reduced for 24GB GPU
                                step=4,  # Encourage 4n+1 values
                                value=5,  # Start with minimal frames
                                info="Recommended: 4n+1 values (5, 9, 13, 17, etc.). Use 5-9 for 24GB GPU"
                            )
                            sample_solver = gr.Radio(
                                label="Sampling Solver",
                                choices=["unipc", "dpm++"],
                                value="unipc",
                                info="unipc is faster, dpm++ may give better quality"
                            )
                        
                        with gr.Column():
                            sample_steps = gr.Slider(
                                label="Sampling Steps",
                                minimum=4,
                                maximum=60,  # Reduce max steps 
                                step=5,
                                value=35,    # Reduce default steps
                                info="Higher = better quality but slower. 30-40 recommended for 24GB GPU"
                            )
                            sample_shift = gr.Slider(
                                label="Sampling Shift",
                                minimum=1.0,
                                maximum=10.0,
                                step=0.5,
                                value=5.0,
                                info="Controls motion intensity (3.0-7.0 recommended)"
                            )
                            guide_scale = gr.Slider(
                                label="Guidance Scale",
                                minimum=1.0,
                                maximum=10.0,
                                step=0.5,
                                value=5.0,
                                info="Higher = more prompt adherence"
                            )
                    
                    with gr.Row():
                        seed = gr.Number(
                            label="Seed",
                            value=-1,
                            precision=0,
                            info="Set to -1 for random seed"
                        )
                        use_prompt_extend = gr.Checkbox(
                            label="Use Prompt Extension",
                            value=False,
                            info="Improves generation by expanding your prompt"
                        )
                        lora_dropdown = gr.Dropdown(
                            label="Lightning LoRA (optional)",
                            choices=discover_lora_dirs(),
                            value="None",
                            info=f"Searches {LIGHTNING_BASE_DIR} for LoRA exports"
                        )
                    
                    model_path = gr.Textbox(
                        label="Model Path",
                        value=DEFAULT_MODEL_PATHS["ti2v-5B"],
                        info="Path to the model directory"
                    )
                    
                    # Hidden memory optimization controls (set to optimal defaults for 24GB GPU)
                    offload_model_ctrl = gr.Checkbox(value=True, visible=False)   # Hidden, always True
                    t5_cpu_ctrl = gr.Checkbox(value=False, visible=False)        # Hidden, keep on GPU for speed  
                    convert_dtype_ctrl = gr.Checkbox(value=True, visible=False)  # Hidden, always True

            
            with gr.Column(scale=2):
                output_video = gr.Video(label="Generated Video")
                output_message = gr.Textbox(label="Status", interactive=False)
        
        # Set up event handlers
        task_dropdown.change(
            fn=update_ui_for_task,
            inputs=[task_dropdown],
            outputs=[prompt_input, size_dropdown, image_input, model_path, lora_dropdown]
        )
        
        generate_btn.click(
            fn=generate_video,
            inputs=[
                prompt_input, task_dropdown, size_dropdown, frame_num, sample_solver,
                sample_steps, sample_shift, guide_scale, seed,
                use_prompt_extend, model_path, image_input,
                offload_model_ctrl, t5_cpu_ctrl, convert_dtype_ctrl, lora_dropdown
            ],
            outputs=[output_video, output_message]
        )
        
        clear_btn.click(
            fn=lambda: (None, ""),
            inputs=[],
            outputs=[output_video, output_message]
        )
        
        # Example buttons for different model types
        gr.Examples(
            examples=[
                [
                    "A Rottweiler on a boat with breaching humpback whales, camera behind the dog, sunny skies.",
                    "ti2v-5B", "704*1280", 5, "unipc", 35, 5.0, 5.0, -1, False, DEFAULT_MODEL_PATHS["ti2v-5B"], None,
                    True, False, True, "None"  # offload_model, t5_cpu, convert_model_dtype, lora_dir
                ],
                [
                    "Two anthropomorphic cats in boxing gear fighting on a spotlighted stage.",
                    "t2v-A14B", "832*480", 5, "unipc", 35, 5.0, 5.0, -1, False, DEFAULT_MODEL_PATHS["t2v-A14B"], None,
                    True, False, True, "None"
                ],
                [
                    "A majestic eagle soaring through a canyon at sunset, cinematic.",
                    "t2v-A14B", "480*832", 5, "unipc", 35, 5.0, 5.0, -1, False, DEFAULT_MODEL_PATHS["t2v-A14B"], None,
                    True, False, True, "None"
                ]
            ],
            inputs=[
                prompt_input, task_dropdown, size_dropdown, frame_num, sample_solver,
                sample_steps, sample_shift, guide_scale, seed,
                use_prompt_extend, model_path, image_input,
                offload_model_ctrl, t5_cpu_ctrl, convert_dtype_ctrl, lora_dropdown
            ]
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the Gradio interface
    demo = create_interface()
    demo.launch(share=True)
