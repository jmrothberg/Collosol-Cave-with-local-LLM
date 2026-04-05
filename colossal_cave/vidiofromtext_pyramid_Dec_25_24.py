#!/usr/bin/env python3
"""
Pyramid Flow Video Generator with Gradio UI
============================================
JMR's text-to-video and image-to-video generation using Pyramid-Flow model.

Pyramid-Flow is a diffusion-based video generation model that can create
short video clips from text prompts or from text + image combinations.

Features:
    - Text-to-video generation
    - Image-to-video generation (with uploaded image)
    - Configurable video length (5 or 10 seconds)
    - Gradio web interface for easy use

Requirements:
    - Python 3.11 (3.12 has matrix compatibility issues)
    - pip install torch diffusers gradio imageio imageio-ffmpeg
    - Pyramid-Flow-SD3 model at /data/pyramid-flow-sd3/
      Download: python huggingfacesnapshotpyramid.py
    - CUDA GPU with 24GB+ VRAM recommended

Usage:
    python vidiofromtext_pyramid_Dec_25_24.py
    
    Opens web UI at http://127.0.0.1:7860

Notes:
    - Video generation takes several minutes per clip
    - Output videos saved with timestamp filenames
    - Uses bfloat16 for memory efficiency
"""

import sys
import os
import datetime
# Add the current directory and Pyramid_Flow directories to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
pyramid_flow_dir = os.path.join(current_dir, 'Pyramid_Flow')
pyramid_dit_dir = os.path.join(pyramid_flow_dir, 'pyramid_dit')

sys.path.insert(0, current_dir)
sys.path.insert(0, pyramid_flow_dir)
sys.path.insert(0, pyramid_dit_dir)
import torch
from PIL import Image
from Pyramid_Flow.pyramid_dit import PyramidDiTForVideoGeneration
from diffusers.utils import load_image, export_to_video
import gradio as gr
import imageio
import imageio_ffmpeg

# Set up CUDA device
torch.cuda.set_device(0)
model_dtype, torch_dtype = 'bf16', torch.bfloat16  # Use bf16, fp16 or fp32
os.environ["TOKENIZERS_PARALLELISM"] = "false"
PATH =  '/data/pyramid-flow-sd3/'    
#PATH = '/data/pyramid-flow-miniflux/' #failed to load model even downloading from huggingface with snapshot_download

# Initialize the model (do this outside the function to avoid reloading)
model = PyramidDiTForVideoGeneration(
    PATH,  # Replace with the actual path to your downloaded checkpoint dir
    model_dtype,
    model_variant='diffusion_transformer_768p',  # or 'diffusion_transformer_384p'
    low_cpu_mem_usage=False,
    device_map=None
)

# Move model components to CUDA
model.vae.to("cuda")
model.dit.to("cuda")
model.text_encoder.to("cuda")
model.vae.enable_tiling()

def generate_video(prompt, input_image=None, video_length="5 seconds"):
    """
    Generate a video based on the input prompt and optionally an input image.
    """
    # Set temperature based on desired video length
    temp = 16 if video_length == "5 seconds" else 31
    
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch_dtype):
        if input_image is None:
            # Generate video from text
            frames = model.generate(
                prompt=prompt,
                num_inference_steps=[20, 20, 20],
                video_num_inference_steps=[10, 10, 10],
                height=768,     
                width=1280,
                temp=temp,
                guidance_scale=9.0,
                video_guidance_scale=5.0,
                output_type="pil",
            )
        else:
            # Generate video from text and image
            image = Image.open(input_image).convert("RGB").resize((1280, 768))
            frames = model.generate_i2v(
                prompt=prompt,
                input_image=image,
                num_inference_steps=[10, 10, 10],
                temp=temp,
                video_guidance_scale=4.0,
                output_type="pil",
            )
    
    # Export frames to video
    import datetime
    prompt_prefix = prompt[:10].replace(" ", "_")
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{prompt_prefix}_{current_time}.mp4"
    export_to_video(frames, output_path, fps=24)
    
    return output_path

# Define the Gradio interface
iface = gr.Interface(
    fn=generate_video,
    inputs=[
        gr.Textbox(label="Enter prompt for the video"),
        gr.Image(label="Upload an image (optional)", type="filepath"),
        gr.Radio(["5 seconds", "10 seconds"], label="Video Length", value="5 seconds")
    ],
    outputs=gr.Video(label="Generated Video"),
    title="JMR's AI Video Generator Pyramid Flow Version",
    #description="Generate a video from a text prompt, optionally with an input image.",
    article="""
    Instructions:
    1. Enter a descriptive prompt for your desired video.
    2. (Optional) Upload an image if you want to generate a video based on both text and image.
    3. Choose the desired video length (5 or 10 seconds).
    4. Click 'Submit' to generate the video.
    5. Wait for the process to complete - it may take a few minutes.
    6. The generated video will appear below when ready.
    
    Note: Video generation is a complex process and may take several minutes. Please be patient!
    """,
    allow_flagging="never",
    # Add this line to disable the submit button during processing
    
    submit_btn=gr.Button("Generate Video", variant="primary"),
    
)

# Launch the app
if __name__ == "__main__":
    iface.queue().launch(share=True)