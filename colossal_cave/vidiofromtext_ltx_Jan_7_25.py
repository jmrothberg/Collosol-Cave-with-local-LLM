#JMR text to video using Pyramid-Flow had to ad paths and take some code from github
# got matrix error with 3.12, so went back to 3.11.11 had same eerror so tried installing requirements.txt
# updated to use LTX
# pip install -U git+https://github.com/huggingface/diffusers
#git clone https://github.com/Lightricks/LTX-Video.git
#cd LTX-Video
#from huggingface_hub import snapshot_download

#model_path = 'PATH'   # The local directory to save downloaded checkpoint
#snapshot_download("Lightricks/LTX-Video", local_dir=model_path, local_dir_use_symlinks=False, repo_type='model')


import sys
import os
import datetime

import torch
from PIL import Image

from diffusers.utils import load_image, export_to_video
from diffusers import LTXPipeline, LTXImageToVideoPipeline
import gradio as gr


# Add new function to find GPU with most available memory
def get_gpu_with_most_memory():
    if not torch.cuda.is_available():
        print("No CUDA devices available. Using CPU.")
        return "cpu"
    
    available_gpus = range(torch.cuda.device_count())
    max_memory_available = 0
    selected_gpu = 0
    
    print("\nGPU Information:")
    print("-" * 50)
    
    for gpu_id in available_gpus:
        # Get GPU properties
        props = torch.cuda.get_device_properties(gpu_id)
        memory_stats = torch.cuda.mem_get_info(gpu_id)
        free_memory = memory_stats[0]
        total_memory = memory_stats[1]
        used_memory = total_memory - free_memory
        
        # Print detailed information
        print(f"\nGPU {gpu_id}: {props.name}")
        print(f"Total Memory: {total_memory / (1024**3):.2f} GB")
        print(f"Used Memory:  {used_memory / (1024**3):.2f} GB")
        print(f"Free Memory:  {free_memory / (1024**3):.2f} GB")
        
        if free_memory > max_memory_available:
            max_memory_available = free_memory
            selected_gpu = gpu_id
    
    print("\nSelection:")
    print(f"-> Using GPU {selected_gpu} with {max_memory_available / (1024**3):.2f} GB free memory")
    print("-" * 50)
    
    return f"cuda:{selected_gpu}"

# Replace the manual GPU selection with automatic selection
device = get_gpu_with_most_memory()
model_dtype, torch_dtype = 'bf16', torch.bfloat16

# Update model initialization
model = LTXPipeline.from_pretrained("/data/LTX-Video", torch_dtype=torch.bfloat16)
model.to(device)

pipe = LTXImageToVideoPipeline.from_pretrained("/data/LTX-Video", torch_dtype=torch.bfloat16)
pipe.to(device)  # Use same device for both models


def generate_video(prompt, input_image=None, video_length="5 seconds", negative_prompt="worst quality, lack of motion, blurry, jittery, distorted"):
    """
    Generate a video based on the input prompt and optionally an input image.
    """
    num_frames = 161 if video_length == "10 seconds" else 81  # Adjust frame count based on length
    
    with torch.no_grad():
        if input_image is None:
            # Generate video from text using LTX
            video = model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=704,
                height=480,
                num_frames=num_frames,
                num_inference_steps=50,
            ).frames[0]
        else:
            # Generate video from image using LTX
            image = Image.open(input_image).convert("RGB").resize((704, 480))
            video = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=704,
                height=480,
                num_frames=num_frames,
                num_inference_steps=50,
            ).frames[0]
    
    # Export frames to video
    import datetime
    prompt_prefix = prompt[:10].replace(" ", "_")
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{prompt_prefix}_{current_time}.mp4"
    export_to_video(video, output_path, fps=24)
    return output_path

# Define the Gradio interface
iface = gr.Interface(
    fn=generate_video,
    inputs=[
        gr.Textbox(label="Enter prompt for the video"),
        gr.Image(label="Upload an image (optional)", type="filepath"),
        gr.Radio(["5 seconds", "10 seconds"], label="Video Length", value="5 seconds"),
        gr.Textbox(label="Negative prompt", value="worst quality, lack of motion, blurry, jittery, distorted")
    ],
    outputs=gr.Video(label="Generated Video"),
    title="JMR's AI Video Generator LTX Version",
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