# Date 1/9/2024 JMR's Small_llm_Jan_9.py based adventure game using LLMs
# Date 1/12/2024 JMR's Small_llm_Jan_12.py based adventure game using LLMs with NPC working!
# Date 1/16/2024 JMR's Small_llm_Jan_16.py based adventure game using LLMs with NPC working and riddles!
# Date 1/19/2024 JMR's Small_llm_Jan_19.py Moved to Web version with Gradio 
# Date 1/22/224 Adv_Jan_22_Map_Rework for image on the left side, image for NPC, and image for maps
# Date 1/23/2024 Colossal Cave Adventure Game with LLMs and Diffusers working on unix compatibility
# Date 1/26/2024 distributing riddles and magic items to rooms, npc, and monsters need monsters t drop them. 
# Date 1/27/2024 Refactored code to have living things as class.
# Date 1/29/2024 Refactored code to have Adventure class
# Date 2/3/2024 Changes processing of commands to use sentece transformers for vector match.
# July 26 use proper format for Llama 3.1 to ask questions. Undersanding improved greatly.
# Rooms is a dictionary of room objects not a list with the key being the room number.
# pip install sentence_transformers
# export LLAMA_METAL=on
# echo $LLAMA_METAL
# CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install -U llama-cpp-python --no-cache-dir
# export LLAMA_CUBLAS=1
# echo $LLAMA_CUBLAS
# for UNIX with CUDA
#updated Aug 31 2024 with ubunto 24.04 now needs prebuilt wheel
#pip install llama-cpp-python   --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --no-cache-dir llama-cpp-python
# pip install llama-cpp-python   --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --no-cache-dir llama-cpp-python
# each time you update diffusers for Open you need to edit this file.
# /Users/jonathanrothberg/Colossal_Cave/.venv/lib/python3.11/site-packages/diffusers/schedulers/scheduling_k_dpm_2_ancestral_discrete.py", line 280
# looks fixed in diffusers that now check for MPS and do this so not needed -line 279 timesteps = timesteps.astype('float32')TypeError: Cannot convert a MPS Tensor to float64
# got requests.exceptions.HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/api/models/all-mpnet-base-v2/revision/main - fixed with
# pip install git+https://github.com/huggingface/diffusers.git
# Sept 7 2024 - added FluxPipeline 
# Oct 11 2024 - added video generation to rooms. Updated to use video_data instead of video_path so saves standalone
# Oct 12 2024 - added code to shut off pyramid flow and dit for video generation on mac. cleaned up some variable names
# Dec 19 2024 - added code to use Hugging Face Hub for diffusers models and cached models, as the downloaded do not work with new diffusers
# Dec 22 use updated requirements.txt huggingface-hub diffusers 
#from huggingface_hub import snapshot_download
#Dec 22 2024 - added code to download pyramid flow sd3 from huggingface otherwise old format
#model_path = '/data/pyramid-flow-sd3'   # The local directory to save downloaded checkpoint
#snapshot_download("rain1011/pyramid-flow-sd3", local_dir=model_path, local_dir_use_symlinks=False, repo_type='model')
#plays vidieos only when on 2 gpus.
#Dec 30 2024 - cleaned up LLM parsing instructions 
#Jan 8 2025 - added code to use stable-diffusion-3.5-large for images
#Jan 9 2025 - Pyramid_dit_for_video_gen_pipelines.py
#import torch
#from diffusers import StableDiffusion3Pipeline
#pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
#pipe = pipe.to("cuda")
#   image = pipe(
#    "A capybara holding a sign that reads Hello World",
#    num_inference_steps=28,
#    guidance_scale=3.5,
#).images[0]
#image.save("capybara.png")
# July 9 2025 - added code to help with questions about the game.
# July 21 2025 - MAJOR UPDATE: Added Mac MPS video generation support! 
#   - Video generation now works on Mac with Apple Silicon using MPS acceleration
#   - Requires PyTorch 2.9.0+ with Conv3D MPS support
#   - Auto-downloads pyramid-flow-miniflux model for Mac
#   - Includes float64->float32 compatibility fixes for MPS
#   - Maintains same 1280x768 aspect ratio as CUDA version
#   - All existing CUDA functionality preserved
# July 21 2025 - added code to work with FLUX models on Mac MPS
"""
COLOSSAL CAVE ADVENTURE GAME WITH LLM & DIFFUSION INTEGRATION

This is a text-based adventure game that integrates:
- Ollama LLM for natural language command processing
- Diffusion models for generating artwork
- Video generation for immersive room experiences
- Web interface using Gradio

ARCHITECTURE OVERVIEW:
- Game state stored in rooms dictionary (key = room number)
- Player object tracks inventory, health, location
- NPCs have memory and personality traits
- LLM processes natural language commands into game actions
- All game data loaded from adventure_dataRA.json

DEVICE ALLOCATION:
- Mac: Uses MPS device for diffusion AND video generation (PyTorch 2.9.0+)
- Multi-GPU: Distributes LLM, diffusion, and video across GPUs
- Single GPU: Shares device between models

CORE GAME LOOP:
1. Player enters command in natural language
2. LLM parses command into action + entities
3. Game executes action and updates state
4. Generate images/videos if needed
5. Return updated game state to UI
"""
import os
import sys
import re
import gc   
import random
import torch
import json
import pickle
import glob
import platform
from datetime import datetime
import subprocess
import importlib.util
# Removed subprocess import - was only used by removed speak() function

def ensure_mflux_installed():
    """
    Check if mflux is installed, and install it if not available.
    This is specifically for Mac compatibility where the old flux package doesn't work.
    """
    try:
        import mflux
        print("✓ mflux is already installed")
        return True
    except ImportError:
        print("mflux not found. Attempting to install...")
        try:
            # Try to install mflux
            subprocess.check_call([sys.executable, "-m", "pip", "install", "mflux"])
            print("✓ mflux installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install mflux: {e}")
            print("You may need to install mflux manually: pip install mflux")
            return False

# === UI AND DIALOG IMPORTS ===
from tkinter import filedialog, Tk
import gradio as gr
from PIL import Image, ImageDraw, ImageFont, ImageOps

# === ML/AI IMPORTS ===
import ollama  # Local LLM interface
from sentence_transformers import SentenceTransformer  # For vector embeddings (simplified)
from fuzzywuzzy import fuzz  # For fuzzy string matching in item names

# === DIFFUSION/IMAGE GENERATION IMPORTS ===
from diffusers import DiffusionPipeline, FluxPipeline, StableDiffusion3Pipeline
from huggingface_hub import login

# === GAME-SPECIFIC IMPORTS ===
from complete_instruction import get_complete_instructions  # Game help system

# === HUGGING FACE AUTHENTICATION ===
# Try to login to Hugging Face using environment variable, but don't fail if offline
# Set HF_TOKEN environment variable with your Hugging Face token (get from https://huggingface.co/settings/tokens)
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    try:
        login(hf_token)
        print("Successfully logged in to Hugging Face")
    except Exception as e:
        print(f"Warning: Could not login to Hugging Face: {e}")
        print("This is okay - you can still use local models and the game will work fine.")
else:
    print("Note: HF_TOKEN not set. Set environment variable for Hugging Face access.")
    print("You can still use local models and the game will work fine.")

# === DEVICE SETUP FOR MULTI-MODEL DEPLOYMENT ===
print("Platform.machine: ", platform.machine())
print ("Platform.system: ", platform.system())

# Device allocation strategy: Distribute models across available hardware
if torch.backends.mps.is_available():  # macOS with Apple Silicon
    diffuser_device = torch.device("mps")
    x = torch.ones(1, device=diffuser_device)
    print (x)
    dtype = torch.float32
    # Set llm_device and video_device for Mac
    llm_device = "mps"
    # Updated: Mac now supports video generation with MPS and Conv3D support!
    video_device = "mps"  # Video generation now supported on Mac with PyTorch 2.9.0+
    num_gpus = 0  # No CUDA GPUs on Mac
else:
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Number of available GPUs: {num_gpus}")
    
    # Print GPU memory info
    for gpu_id in range(num_gpus):
        memory_stats = torch.cuda.mem_get_info(gpu_id)
        free_memory = memory_stats[0] / (1024**3)  # Convert to GB
        total_memory = memory_stats[1] / (1024**3)
        print(f"\nGPU {gpu_id}:")
        print(f"Total Memory: {total_memory:.1f}GB")
        print(f"Free Memory:  {free_memory:.1f}GB")

    # Assign devices using existing logic
    if num_gpus >= 4:
        # Check if GPU 0 is being used significantly
        try:
            memory_stats = torch.cuda.mem_get_info(0)
            free_memory = memory_stats[0] / (1024**3)  # Convert to GB
            total_memory = memory_stats[1] / (1024**3)
            used_memory = total_memory - free_memory
            
            if used_memory > 4.0:  # If more than 1GB is being used on GPU 0
                # Shift everything one GPU over
                os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
                llm_device = "cuda:1"  # This will map to physical GPU 1
                diffuser_device = "cuda:2"  # This will map to physical GPU 2
                video_device = "cuda:3"  # This will map to physical GPU 3
            else:
                # Use original GPU assignments
                os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
                llm_device = "cuda:0"
                diffuser_device = "cuda:1"
                video_device = "cuda:2"
        except Exception as e:
            print(f"Error checking GPU memory: {e}")
            # Fall back to original GPU assignments
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
            llm_device = "cuda:0"
            diffuser_device = "cuda:1"
            video_device = "cuda:2"
    elif num_gpus == 3:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
        llm_device = "cuda:0"
        diffuser_device = "cuda:1"
        video_device = "cuda:2"
    elif num_gpus == 2:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        llm_device = "cuda:0"
        diffuser_device = "cuda:1"
        video_device = None
    elif num_gpus == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        llm_device = "cuda:0"
        diffuser_device = "cuda:0"
        video_device = None
    else:
        print("No CUDA GPUs found")
        llm_device = "cpu"
        diffuser_device = "cpu"
        video_device = None

    dtype = torch.float16 if diffuser_device != "cpu" else torch.float32
print ("LLM device:", llm_device)
print ("diffuser_device:", diffuser_device)
print ("Video device:", video_device)
if video_device == "mps":
    print ("🎬 Video generation now supported on Mac with MPS acceleration!")

# Updated: Now supports video generation on Mac with MPS
# Import video generation modules for CUDA systems AND Mac with MPS
if platform.system() != 'Darwin' or torch.backends.mps.is_available():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try different possible paths for Pyramid Flow
    possible_paths = [
        os.path.join(current_dir, 'Pyramid-Flow'),  # Correct path with dash
        os.path.join(current_dir, 'Pyramid_Flow'),  # Alternative with underscore
        os.path.join(current_dir, '@Pyramid-Flow'),
        os.path.join(current_dir, 'pyramid_flow'),
        current_dir  # If modules are directly in current directory
    ]

    # Add first existing path to sys.path
    pyramid_flow_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            pyramid_flow_dir = path
            sys.path.insert(0, path)
            break

    try:
        from pyramid_dit import PyramidDiTForVideoGeneration
        from diffusers.utils import load_image, export_to_video
        print(f"Successfully imported video generation modules from {pyramid_flow_dir}")
    except ImportError as e:
        print(f"Warning: Could not import video generation modules: {e}")
        print("Video generation features will be disabled")
        PyramidDiTForVideoGeneration = None

global video_model
video_model = None

def get_video_model():
    global video_model, PyramidDiTForVideoGeneration
    
    # Check if the class is available
    if 'PyramidDiTForVideoGeneration' not in globals() or PyramidDiTForVideoGeneration is None:
        print("PyramidDiTForVideoGeneration not available - reimporting...")
        from pyramid_dit import PyramidDiTForVideoGeneration

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Configure based on device type
    if video_device == "mps":
        print("Setting up video model for Mac MPS...")
        
        # Use mini-flux model with auto-download for Mac (same as your working test)
        model_path = './pyramid-flow-miniflux'
        if not os.path.exists(model_path):
            print("Downloading Pyramid-Flow miniflux model for Mac...")
            from huggingface_hub import snapshot_download
            snapshot_download("rain1011/pyramid-flow-miniflux", local_dir=model_path, 
                             local_dir_use_symlinks=False, repo_type='model')
            print("Model downloaded successfully!")
        
        # Use same settings as your working test
        model_dtype = 'fp32'  # Use fp32 like your working test
        torch_dtype = torch.float32
        
        # Get memory info for model variant selection
        import psutil
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"Detected Mac memory: {total_memory_gb:.1f}GB")
        
        # Use the proven working settings from mac_pyramid.py
        model_variant = 'diffusion_transformer_384p'  # Use 384p variant - proven to work
        print(f"Using 384p model variant (proven stable settings)")
        
        # Set up MPS compatibility fixes (same as your working test)
        torch.set_default_dtype(torch.float32)
        
        # Monkey patch torch functions to avoid float64 on MPS (same as your working test)
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
        
        # MPS memory management (same as your working test)
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        
    else:
        # CUDA configuration (existing logic)
        print("Setting up video model for CUDA...")
        model_path = '/data/pyramid-flow-sd3/'
        model_dtype = 'bf16'
        torch_dtype = torch.bfloat16
        model_variant = 'diffusion_transformer_768p'

    # Initialize the model (same as your working test)
    video_model = PyramidDiTForVideoGeneration(
        model_path,
        model_name="pyramid_flux",
        model_dtype=model_dtype,
        model_variant=model_variant,
    )

    # Move model components to appropriate device
    if video_device == "mps":
        print("Moving video model components to MPS...")
        video_model.vae = video_model.vae.to(video_device)
        video_model.dit = video_model.dit.to(video_device)
        video_model.text_encoder = video_model.text_encoder.to(video_device)
    else:
        # CUDA setup (existing logic)
        video_model.vae.to(video_device)
        video_model.dit.to(video_device)
        video_model.text_encoder.to(video_device)
    
    video_model.vae.enable_tiling()
    return video_model

def setup_temp_directory():
    # Create a 'temp' directory in your project folder
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp_videos')
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def resource_path(relative_path):
    """ Get absolute path to resource, works for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def ask_for_folder():
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory()
    return folder_path

# Determine paths based on the operating system
if platform.system() == 'Darwin':  # macOS
    default_data_path = "/Users/jonathanrothberg/Colossal_Cave"
    default_model_path = "/Users/jonathanrothberg"
else:  # Linux
    default_data_path = "/home/jonathan/Colossal_Cave"
    default_model_path = "/data"

# Check if paths exist, if not, ask user to select
if not os.path.exists(default_data_path):
    print("Please select the path for saved games, art, and game data .")
    data_path = ask_for_folder()
else:
    data_path = default_data_path

if not os.path.exists(default_model_path):
    print("Please select the path for LLM & DIFF folders:")
    model_path = ask_for_folder()
else:
    model_path = default_model_path

# Load the SentenceTransformer model
vector_model = SentenceTransformer('all-mpnet-base-v2', cache_folder=model_path)

# === GAME DATA MANAGEMENT ===
def load_adventure_data():
    """
    Loads all game content from adventure_dataRA.json including:
    - Room descriptions and riddles
    - Item catalogs (weapons, armor, treasures, magic items)  
    - NPC names and descriptions
    - Monster definitions
    
    Creates backup copies of all lists for respawning items when lists are exhausted.
    This prevents the game from running out of content during long sessions.
    """
    global room_descs_riddles, monsters, weapons, armors, treasures, magic_items, npcs, npc_descs, \
        copy_room_descs_riddles, copy_monsters, copy_weapons, copy_treasures, copy_armors, copy_magic_items, copy_npcs, copy_npc_descs
    
    adventure_data_path = resource_path(os.path.join(data_path, "adventure_dataRA.json"))
    
    try:
        with open(adventure_data_path, 'r', encoding='utf-8') as f:
            adventure_data = json.load(f)

            # Load primary content lists with validation
            room_descs_riddles = adventure_data.get("room_descs_riddles", [])
            monsters = adventure_data.get("monsters", [])
            weapons = adventure_data.get("weapons", [])
            armors = adventure_data.get("armors", [])
            treasures = adventure_data.get("treasures", [])
            magic_items = adventure_data.get("magic_items", [])
            npcs = adventure_data.get("npcs", [])
            npc_descs = adventure_data.get("npc_descs", [])

            # Validate essential data exists
            if not room_descs_riddles:
                print("Warning: No room descriptions found in adventure data. Game may not work properly.")

            # Create backup copies for when original lists are exhausted
            copy_room_descs_riddles = room_descs_riddles.copy()
            copy_monsters = monsters.copy()
            copy_weapons = weapons.copy()
            copy_armors = armors.copy()
            copy_treasures = treasures.copy()
            copy_magic_items = magic_items.copy()
            copy_npcs = npcs.copy()
            copy_npc_descs = npc_descs.copy()
            
    except FileNotFoundError:
        print(f"Error: Could not find adventure data file at {adventure_data_path}")
        print("Please check the file path and try loading again.")
        return False
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in adventure data file: {e}")
        print("Please check the file format and try loading again.")
        return False
    except Exception as e:
        print(f"Error loading adventure data: {e}")
        print("Please try loading again.")
        return False
    
    return True

def speak(text):
    subprocess.call(['say', text])

def sanitize_filename(filename):
    return re.sub(r'[^a-zA-Z0-9\-\_\.]', '_', filename)

def generate_number(lower90=0, upper90=2, lower10=3, upper10=5):
    percentage = random.randint(0, 100)
    if percentage <= 90:
        return random.randint(lower90, upper90)
    else:
        return random.randint(lower10, upper10)

def saved_files():
    list_of_saved_games = glob.glob(os.path.join(game_dir, 'Adv_*.pkl'))
    game_names = [os.path.basename(path)[4:-4] for path in list_of_saved_games]
    return game_names

def saved_llms():
    try:
        # Get list of models from Ollama
        response = ollama.list()
        
        # Extract model names - response has a 'models' attribute
        llm_names = []
        if hasattr(response, 'models'):
            # It's the new response format with Model objects
            for model in response.models:
                if hasattr(model, 'model'):
                    llm_names.append(model.model)
                else:
                    llm_names.append(str(model))
        elif isinstance(response, dict) and 'models' in response:
            # Legacy dict format
            for model in response['models']:
                if isinstance(model, dict) and 'name' in model:
                    llm_names.append(model['name'])
                else:
                    llm_names.append(str(model))
        else:
            # Unknown format, try to iterate
            try:
                for model in response:
                    llm_names.append(str(model))
            except:
                print(f"Unknown response format: {type(response)}")
                    
        # Sort models and put common ones first
        common_models = ['llama3.1', 'llama3', 'llama2', 'mistral', 'phi4', 'phi3', 'gemma3', 'gemma2', 'qwen3', 'qwen2.5']
        sorted_names = []
        
        # Add common models first if they exist
        for common in common_models:
            for name in llm_names:
                if name.startswith(common):
                    if name not in sorted_names:
                        sorted_names.append(name)
        
        # Add any remaining models
        for name in llm_names:
            if name not in sorted_names:
                sorted_names.append(name)
                
        return sorted_names if sorted_names else ["No Ollama models found"]
    except Exception as e:
        print(f"Error getting Ollama models: {e}")
        import traceback
        traceback.print_exc()
        return ["No Ollama models found"]



def saved_diffusers():
    # Check if CUDA is available before trying to get device properties
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # bytes to GB
    else:
        # For Mac or CPU, assume we have limited memory
        total_memory = 8  # Conservative estimate for non-CUDA systems
    
    diffuser_names = []
    
    # First, try to get local models from diff_dir if it exists
    if os.path.exists(diff_dir):
        local_models = [name for name in os.listdir(diff_dir) 
                       if os.path.isdir(os.path.join(diff_dir, name)) and not name.startswith('.')]
        if local_models:
            print(f"Found local diffusion models: {local_models}")
            diffuser_names.extend(local_models)
    
    # Simplified diffuser list - just FLUX (best quality) and SDXL-Turbo (smallest/fastest)
    hf_models = [
        "black-forest-labs/FLUX.1-schnell",  # Fast FLUX variant
        "stabilityai/sdxl-turbo",            # Smallest/fastest alternative
    ]
    
    # Add models based on device capability  
    if str(diffuser_device) == "mps":  # Mac systems with MPS
        print("Added FLUX models optimized for Mac MPS")
    elif total_memory >= 26: # High memory CUDA systems
        print("Added FLUX models for high-memory CUDA")
    
    # Add HF models to the list
    diffuser_names.extend(hf_models)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_names = []
    for name in diffuser_names:
        if name not in seen:
            seen.add(name)
            unique_names.append(name)
     
    return unique_names if unique_names else ["No diffusion models found"]

def save_game_state(name_of_game = None):
    if name_of_game == None:
        name_of_game = (f"{datetime.now().strftime('%d-%H%M')}")
    try:
        file_name = (f'Adv_{name_of_game}_{len(rooms)}_{art_style_g}.pkl')
        with open(os.path.join(game_dir, file_name), 'wb') as f:
            pickle.dump((rooms, playerOne, old_room), f)
            print (f"Game state saved as {file_name}")
    except Exception as e:
        name_of_game = "Error saving game state", e
    return name_of_game

def load_game_state(name):
    path_fullname = os.path.join(game_dir, f'Adv_{name}.pkl')
    try:
        with open(path_fullname, 'rb') as f:
            rooms, playerOne, old_room = pickle.load(f)   
        
        needs_video = False
        for room in rooms.values():
            if hasattr(room, 'video_data') and room.video_data is not None:
                # Video data already exists, no need to do anything
                pass
            elif hasattr(room, 'video_path') and room.video_path:
                # Only a path exists, need to load video data
                try:
                    with open(room.video_path, 'rb') as f:
                        room.video_data = f.read()
                except FileNotFoundError:
                    print(f"Video file not found for room {room.number}: {room.video_path}")
                    room.video_data = None
                    needs_video = True
            else:
                # No video data or path
                room.video_data = None
                needs_video = True
        
        # Video generation now supported on Mac with MPS!
        if needs_video:
            user_input = input("Some rooms don't have videos. Do you want to generate videos for all rooms? (y/n): ").lower()
            generate_videos = user_input == 'y'
            
            if generate_videos:
                print("Starting video generation for all rooms...")
                successful_videos = 0
                total_rooms = len(rooms)
                
                for room in rooms.values():
                    try:
                        print(f"Generating video for room {room.number}/{total_rooms}...")
                        room.generate_and_save_video_self()
                        if room.video_data is not None:
                            successful_videos += 1
                            print(f"✓ Successfully generated video for room {room.number}")
                        else:
                            print(f"✗ Failed to generate video for room {room.number}")
                    except Exception as e:
                        print(f"✗ Error generating video for room {room.number}: {e}")
                        # Continue with next room instead of crashing
                        continue
                
                print(f"Video generation complete: {successful_videos}/{total_rooms} videos generated successfully")
            else:
                print("Skipping video generation.")
        
        # Check if MB exists in room 1, add her if missing
        if 1 in rooms:  # Make sure room 1 exists
            if not rooms[1].npc or not hasattr(rooms[1].npc, 'is_master_beast'):
                # MB is missing, add her
                print("Master Beast not found in saved game. Adding her to Room 1...")
                
                # If room 1 has a different NPC, move them elsewhere
                if rooms[1].npc:
                    old_npc = rooms[1].npc
                    # Find a room without an NPC
                    for room_num in range(2, len(rooms) + 1):
                        if room_num in rooms and not rooms[room_num].npc:
                            rooms[room_num].npc = old_npc
                            old_npc.current_room = room_num
                            print(f"Moved {old_npc.name} to room {room_num}")
                            break
                
                # Create and add MB to room 1
                # Check if the game has LLM features enabled by looking at any NPC
                has_llm = False
                for room in rooms.values():
                    if room.npc and hasattr(room.npc, 'image') and room.npc.image is not None:
                        has_llm = True
                        break
                
                rooms[1].npc = MasterBeast(current_room=1, llm_npc=has_llm)
                print("Master Beast (MB) has been added to Room 1 to help adventurers!")
            else:
                # MB exists but might be missing new attributes - fix them
                mb = rooms[1].npc
                if not hasattr(mb, 'personality_traits'):
                    print("Updating Master Beast with new personality traits...")
                    mb.personality_traits = {
                        "helpfulness": "extremely helpful",
                        "knowledge": "knows everything about spells and game mechanics",
                        "trading_style": "generous teacher",
                        "mood": "wise and patient"
                    }
                # Ensure all required attributes exist
                if not hasattr(mb, 'memory'):
                    mb.memory = []
                if not hasattr(mb, 'player_trades'):
                    mb.player_trades = []
                if not hasattr(mb, 'times_talked'):
                    mb.times_talked = 0
                if not hasattr(mb, 'is_master_beast'):
                    mb.is_master_beast = True
                print("Master Beast attributes updated successfully!")
        
        # Also fix any other NPCs that might be missing personality traits
        for room in rooms.values():
            if room.npc and not hasattr(room.npc, 'personality_traits'):
                print(f"Updating NPC {room.npc.name} with personality traits...")
                room.npc.personality_traits = room.npc._generate_personality()
                # Ensure all required attributes exist
                if not hasattr(room.npc, 'memory'):
                    room.npc.memory = []
                if not hasattr(room.npc, 'player_trades'):
                    room.npc.player_trades = []
                if not hasattr(room.npc, 'times_talked'):
                    room.npc.times_talked = 0
        
    except Exception as e:
        print("Error loading game state:", e)
        import traceback
        traceback.print_exc()
        return None, None, None

    print("Loading:", name)
    return rooms, playerOne, old_room


def new_game(num_rooms=25, runllm_diff=False, runllm_mon=False, runllm_item=False, name_of_game=None, generate_videos=False):
    global playerOne, old_room
    old_room = None
    if name_of_game == None:
        name_of_game = (f"{datetime.now().strftime('%d-%H%M')}")
    try:
        if int(num_rooms) > 81:
            num_rooms = 81
        print ("Resetting rooms to ", num_rooms)
    except:
        num_rooms = 25
        print ("Resetting rooms to ", num_rooms)
    num_rooms = int(int(num_rooms)**.5)**2 # make sure it is a square number        
    
    generating_rooms(num_rooms)
    populating_rooms_random(runllm_mon, runllm_item, runllm_diff) # called in all cases, but no description if llm = False
    distributing_riddle_magic_items(runllm_item)
    
    # Add Master Beast (MB) to room 1 - she's always there to help!
    if rooms[1].npc:
        # If room 1 already has an NPC, move them to another room
        old_npc = rooms[1].npc
        # Find a room without an NPC
        for room_num in range(2, num_rooms + 1):
            if not rooms[room_num].npc:
                rooms[room_num].npc = old_npc
                old_npc.current_room = room_num
                break
    
    # Create and add MB to room 1
    rooms[1].npc = MasterBeast(current_room=1, llm_npc=runllm_diff)
    print("Master Beast (MB) has been placed in Room 1 to help adventurers!")
    
    if runllm_diff: # only called if generating with llm also used to make image of npc, removes npc from room drawing
        generating_room_disc_llm() 
        drawing_rooms_Diff()
    if generate_videos:  # Video generation now supported on Mac with MPS!
        generate_room_videos()
    connecting_rooms()    
    playerOne = Player(name="Jack", current_room=rooms[1], runllm_diff=runllm_diff) # creating player
    old_room = None
    name_of_game = save_game_state(name_of_game) # Save fresh game state
    return name_of_game


def generating_rooms(num_rooms): 
    global rooms, room_descs_riddles,copy_room_descs_riddles
    print ("Generating rooms")
    for number in range(1, num_rooms + 1): # note we start the first room at 1! not zero
        room_desc_riddle = random.choice(room_descs_riddles)
        room_descs_riddles.remove(room_desc_riddle) # so no duplicates
        if len(room_descs_riddles) == 0:
            room_descs_riddles = copy_room_descs_riddles.copy()
        print(room_desc_riddle)
        rooms[number] = Room(room_desc_riddle, number)
        
    
def populating_rooms_random(llm_mon = False, llm_items = False, llm_npc= False): # items: weapons, treasure, magic, monsters, npc
    print ("Populating rooms randomly")
    num_rooms = len(rooms)
    for number in range(1, num_rooms + 1):
        rooms[number].equip_self(llm_items)
        rooms[number].populate_room(llm_mon, llm_npc)

def distributing_riddle_magic_items(runllm_items): # adding riddle_magic_items to other rooms and NPCs  
    print ("Distributing riddle_magic_items (names) from original room to new rooms and an NPC")
    num_rooms = len(rooms)
    list_of_magic_items = []
    list_of_rooms = []
    list_of_npcs = []
    list_of_monsters = []
    for number in range(1, num_rooms + 1):
        room = rooms[number]
        list_of_rooms.append(room)
        if room.riddle_magic_items: 
            list_of_magic_items.extend(room.riddle_magic_items)
        if room.npc:
            list_of_npcs.append(room.npc)
        if room.monsters:
            list_of_monsters.extend(room.monsters)
    if (list_of_npcs or list_of_monsters) and list_of_magic_items:
        for item in list_of_magic_items:
            chosen = random.choice(list_of_monsters + list_of_npcs + list_of_rooms)
            chosen.add_magic(runllm_items,item)
            #random.choice(list_of_monsters + list_of_npcs + list_of_rooms).add_magic(runllm_items,item)
            print (f'Adding riddle_magic {item} to {chosen.name}') 
    else:
        print ("No NPCs or riddle_magic_items")

def generating_room_disc_llm(): # actually to console since it is game set up
    print ("Generating room descriptions by llm")
    num_rooms = len(rooms)
    for number in range(1, num_rooms + 1):
        rooms[number].llM_generate_self_description()
    return 

def drawing_rooms_Diff(): # actually to console since it is game set up
    print ("Drawing rooms")
    num_rooms = len(rooms)
    for number in range(1, num_rooms + 1):
        rooms[number].drawing_self()
    return rooms

def generate_room_videos():
    print("Generating room videos")
    num_rooms = len(rooms)
    successful_videos = 0
    
    for number in range(1, num_rooms + 1):
        try:
            print(f"Generating video for room {number}/{num_rooms}...")
            rooms[number].generate_and_save_video_self()
            if rooms[number].video_data is not None:
                successful_videos += 1
                print(f"✓ Successfully generated video for room {number}")
            else:
                print(f"✗ Failed to generate video for room {number}")
        except Exception as e:
            print(f"✗ Error generating video for room {number}: {e}")
            # Continue with next room instead of crashing
            continue
    
    print(f"Video generation complete: {successful_videos}/{num_rooms} videos generated successfully")


def connecting_rooms(): # this is actaully to console since it is game set up
    print ("Generating connections")
    num_rooms = len(rooms)
    row_size = int(num_rooms**0.5)
    for number in range(1, num_rooms+1):
        connect = ""
        while not connect:
            if number % row_size != 0 and "east" not in rooms[number].connected_rooms:   # Connect rooms east-west, but not for the last room in a row        
                if random.randint(0, 1):  # 50% chance to create a connection
                    rooms[number].add_connection("east", rooms[number + 1])
                    connect = "east"
            else:
                connect = "end"
            if number <= num_rooms-row_size and "south" not in rooms[number].connected_rooms:  # Connect rooms north-south, but not for the last row
                if random.randint(0, 1):  # 50% chance to create a connection
                    rooms[number].add_connection("south", rooms[number + row_size])
                    connect = str(connect) + "south"
            else:
                connect = "bottom"
    return 

def resize_with_padding(img, expected_size):
    img = img.copy()
    img.thumbnail((expected_size[0], expected_size[1]), Image.LANCZOS)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


# === CORE GAME CLASSES ===

class AdventureGame:
    """
    BASE CLASS for all game entities (rooms, items, living things).
    
    Provides common functionality:
    - Item management (add/remove/list)
    - Image generation via diffusion models  
    - LLM-generated descriptions
    - Equipment methods (weapons, armor, treasure, magic)
    
    Inherited by: Room, Item, LivingThing classes
    Key pattern: All game objects can hold items and generate images/descriptions
    """
    def __init__(self, name ="", description ="", current_room=None):
        self.name = name
        self.description = description
        self.current_room = current_room
        self.items = []  # All entities can contain items
        self.image = None  # Generated artwork for this entity

    def add_item(self, item):
        """Add an item to this entity's inventory"""
        self.items.append(item)

    def get_item_names(self):
        """Return comma-separated string of all item names"""
        return  ', '.join([item.name for item in self.items])
    
    def draw_and_save(self, prompt):
        print(f"Generating image with prompt: {prompt}")
        print(f"Pipeline type: {type(pipe).__name__}")
        
        num_inf_steps = 25
        # Check if it's a FLUX pipeline by checking the class name or config
        if isinstance(pipe, FluxPipeline):
            num_inf_steps = 4
            guidance_scale = 0.0
            print("Using FLUX-specific settings: 4 steps, 0.0 guidance")
        else:
            num_inf_steps = 25
            guidance_scale = 7
            print("Using standard diffusion settings: 25 steps, 7.0 guidance")

        try:
            image = pipe(
                prompt=prompt,
                height=1024,
                width=1024,
                num_inference_steps=num_inf_steps,
                guidance_scale=guidance_scale,
                #decoder_guidance_scale=7,
                num_images_per_prompt=1,
                        ).images[0]
            print("✓ Image generation completed successfully")
        except Exception as e:
            print(f"✗ Error during image generation: {e}")
            raise
        
        self.image = image
        short_name = self.name[:15]
        timestamp = datetime.now().strftime("%m%d-%H%M")
        file_name = sanitize_filename(f"{short_name}_{timestamp}.png")
        image.save(os.path.join(image_dir, file_name))
        print (f"Image saved as {file_name}")
        return file_name
    
    def add_weapon(self, runllm_items = False, name = None):
        global weapons, copy_weapons
        if name == None:
            weapon = random.choice(weapons)
            weapons.remove(weapon) # so no duplicates
            if len(weapons) == 1:
                print ("Refreshing weapon list from copy_weapons")
                weapons = copy_weapons.copy()
        self.add_item(Weapon(name = weapon, damage = random.randint(10, 50), runllm = runllm_items))

    def add_armor(self, runllm_items = False, name = None):
        global armors, copy_armors
        if name == None:
            armor = random.choice(armors)
            armors.remove(armor)
            if len(armors) == 0:
                print ("Refreshing armor list from copy_armor")
                armors = copy_armors.copy()  
        self.add_item(Armor(name = armor, protection = random.randint(5, 25), runllm = runllm_items))

    def add_treasure(self, runllm_items = False, name = None): #because trophies can be added later
        global treasures, copy_treasures
        if name == None:
            treasure = random.choice(treasures)
            treasures.remove(treasure) # so no duplicates
            if len(treasures) == 0:
                print ("Refreshing treasure list from copy_treasures")
                treasures = copy_treasures.copy()
        else:
            treasure = name
        self.add_item(Treasure(name = treasure, value = random.randint(50, 1000), runllm=runllm_items))

    def add_magic(self, runllm_items = False, riddle_magic_items = None):
        global magic_items, copy_magic_items
        if riddle_magic_items == None:
            magic = random.choice(magic_items)
            magic_items.remove(magic) # so do duplicates
            if len(magic_items) == 0:
                print ("Refreshing magic list from copy_magic_items")
                magic_items = copy_magic_items.copy()
        else:
            magic = riddle_magic_items
        self.add_item(Magic(name = magic, healing = random.randint(20, 200), runllm = runllm_items))   

    def equip_self(self, llm_items = False):  #should make just like we do forrooms.
        weapon_count = generate_number(0,2,3,6) # 90% chance of 0, 1, or 2 weapons, 10% chance of 3, 4, or 5 weapons
        armor_count = generate_number(0,1,2,3) # 90% chance of 0, 1, armor, 10% chance of 2, 3 armor
        treasure_count = generate_number(0,2,3,6) # 90% chance of 0, 1, or 2 treasures, 10% chance of 3, 4, or 5 treasures
        magic_count = generate_number(0,2,3,6) # 90% chance of 0, 1, or 2 magic items, 10% chance of 3, 4, or 5 magic items
        for _ in range(weapon_count):
            self.add_weapon(llm_items)
        for _ in range(armor_count):
            self.add_armor(llm_items)
        for _ in range(treasure_count):
            self.add_treasure(llm_items)
        if isinstance(self, Room):          #the only magjic items that NPC or Monsters have are riddle_magic_items added latter
            for _ in range(magic_count):
                self.add_magic(llm_items)

    def drawing_self(self, style =None, victory = False):
        prompt=""
        if isinstance(self, Room):
            if art_style_g:
                prompt = (f"in {art_style_g} style.")
            if victory:
                prompt = victory
            else:
                prompt = prompt + self.description + "with weapons, treasure and magic items: " + self.get_item_names() + " and monsters: " + self.get_monster_names() 
                if self.npc:
                    if self.npc.image == None:
                        prompt = prompt + " and " + self.npc.description
        elif isinstance(self, Item):
            prompt = (f'{style} a {self.type} {self.description} in an adventue game')
        elif isinstance(self, LivingThing):
            prompt = (f'{style} a {self.description} in an adventue game')
        console = self.draw_and_save(prompt)
        return console
    
    def llM_generate_self_description(self):
        # Item descriptions should be 66 or fewer tokens since most art programs have that limit.
        system_prompt = "You are a creative writer for a cave-based adventure game. Provide concise and vivid descriptions."
        
        if isinstance(self, Item):
            prompt = f"Very briefly describe this {self.type} called {self.name} in a cave-based adventure game:"
            max_tokens = 66
        elif isinstance(self, LivingThing):
            prompt = f"Very briefly describe this {self.name} in a cave-based adventure game:"
            max_tokens = 66
        elif isinstance(self, Room):
            # This is a full description from the short description that was previously generated.
            items = "items: " + self.get_item_names() + ", monsters: " + self.get_monster_names()
            if self.npc:
                items = items + " and other character " + self.npc.description
            prompt = f"Expand on this description for a room in an adventure game: {self.description} adding these {items} into the description."
            max_tokens = 256
        else:
            raise ValueError("Unsupported object type for description generation")

        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

        generated_text = ollama_generate(formatted_prompt, max_tokens=max_tokens, temperature=0.3)
        
        if isinstance(self, Room):
            self.full_description = generated_text
            print(self.full_description)
            return self.full_description
        else:
            self.description = generated_text
            print(self.description)
            return self.description


class Room(AdventureGame):
    """
    ROOM CLASS represents a game location.
    
    Core functionality:
    - Contains items, monsters, NPCs
    - Has riddles that require specific magic items to solve
    - Connects to other rooms via directions (north/south/east/west)
    - Generates room artwork and optional video content
    - Tracks exploration and riddle completion status
    
    Key attributes:
    - number: Unique room identifier (1-based)
    - connected_rooms: Dict of {direction: Room} connections
    - riddle_magic_items: List of magic items needed to solve riddle
    - explored: Whether player has visited this room
    - riddle_solved: Whether riddle has been completed
    """
    def __init__(self, room_desc_riddle, number):
        super().__init__(name="Room " + str(number), description=room_desc_riddle["description"])
        
        # Room identification
        self.room_desc_riddle = room_desc_riddle
        self.number = number
        
        # Description management
        self.full_description = room_desc_riddle["description"]  # LLM-expanded description
        
        # Riddle system
        self.riddles = room_desc_riddle["riddles"]
        self.hints = room_desc_riddle["hints"]
        self.riddle_magic_items = room_desc_riddle["magic_items"]  # Items needed to solve riddle
        self.riddle_action_words = room_desc_riddle["action_words"]
        self.riddle_solved = False
        
        # Room connections and navigation
        self.connected_rooms = {}  # {direction: Room} - north/south/east/west
        
        # Room inhabitants
        self.monsters = []  # List of Monster objects
        self.npcs = []      # Legacy - not used
        self.npc = None     # Single NPC per room
        
        # Player interaction tracking
        self.explored = False  # Has player visited this room?
        
        # Media content
        self.video = None       # Legacy
        self.video_path = None  # Legacy
        self.video_data = None  # Binary video content for room  

    def generate_and_save_video_self(self):
        global video_model
        
        # Check if video generation is supported on this device
        if video_device is None:
            print("Video generation not supported on this device configuration.")
            return
            
        if video_model is None:
            video_model = get_video_model()

        # Simplified two-tier memory settings based on your 512GB system
        import psutil
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        if video_device == "mps":
            print(f"Generating video on Mac with MPS acceleration... (Memory: {total_memory_gb:.1f}GB)")
            
            # Use proven working settings from mac_pyramid.py for 5-second video
            width, height = 640, 384  # Same as your working test
            temp = 16  # temp=16 gives 5 seconds for 384p model
            print(f"Using proven stable settings: {width}x{height}, temp={temp} for 5-second video")
            
            # Consistent MPS settings matching your working test
            num_inference_steps = [5, 5, 5]  # Fast inference like your test
            video_guidance_scale = 4.0  # Same as your working test
            save_memory = False  # Same as your working test for speed
            
            # Use simple context manager like your working test
            context_manager = torch.no_grad()
            
            # Clear MPS cache before generation
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            
        else:
            print("Generating video on CUDA...")
            # CUDA settings (existing)
            width, height = 1280, 768
            temp = 16
            num_inference_steps = [20, 20, 20]
            video_guidance_scale = 5.0
            save_memory = True
            context_manager = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

        # Generate descriptive prompt with action verbs
        your_verb = random.choice(["run", "walk", "crawl"])
        creatures_verb = random.choice(["lurking", "walking", "prowling"])
        prompt = f"{your_verb} through, {self.description} All of the creatures {creatures_verb}."
        print(f"Generating video for Room {self.number}: {prompt}")
        
        try:
            with context_manager:
                if self.image == None:
                    # Text-to-video generation
                    frames = video_model.generate(
                        prompt=prompt,
                        num_inference_steps=num_inference_steps,
                        video_num_inference_steps=num_inference_steps,  # Use same as num_inference_steps
                        height=height,     
                        width=width,
                        temp=temp,
                        guidance_scale=7.0,  # Standard guidance scale
                        video_guidance_scale=video_guidance_scale,
                        output_type="pil",
                        save_memory=save_memory,
                    )
                else:
                    # Image-to-video generation (preferred for consistency)
                    image = self.image.copy().convert("RGB")
                    image = resize_with_padding(image, (width, height))
                    frames = video_model.generate_i2v(
                        prompt=prompt,
                        input_image=image,
                        num_inference_steps=num_inference_steps,
                        temp=temp,
                        video_guidance_scale=video_guidance_scale,
                        output_type="pil",
                        save_memory=save_memory,
                    )

            print(f"Generated {len(frames)} frames successfully!")

            # Save video with timestamp
            prompt_prefix = self.description[:10].replace(" ", "_")
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_path = os.path.join(image_dir, f"room_{self.number}_{prompt_prefix}_{current_time}.mp4")
            export_to_video(frames, self.video_path, fps=24)

            # Read the video file into memory
            with open(self.video_path, 'rb') as video_file:
                self.video_data = video_file.read()
                
            print(f"Generated video for Room {self.number} using {video_device.upper()}")
            print(f"Video resolution: {width}x{height}, frames: {len(frames)}, saved as: {os.path.basename(self.video_path)}")
            
        except Exception as e:
            print(f"Error generating video for Room {self.number}: {e}")
            # Set video data to None so it can be retried later
            self.video_data = None
            self.video_path = None

    def add_connection(self, direction, room):
        self.connected_rooms[direction] = room
        if self not in room.connected_rooms.values():
            room.connected_rooms[self.opposite_direction(direction)] = self

    @staticmethod
    def opposite_direction(direction):
        return {"north": "south", "south": "north", "east": "west", "west": "east"}[direction]

    def get_monster_names(self):
        return ', '.join([monster.name for monster in self.monsters])

    def add_npc(self, llm_npc = False): # only in rooms
        global npcs, npc_descs, copy_npcs, copy_npc_descs
        npc = random.choice(npcs)
        npcs.remove(npc) # so do duplicates
        if len(npcs) == 0:
            npcs = copy_npcs.copy()
        npc_desc =random.choice(npc_descs)
        npc_descs.remove(npc_desc) # so no duplicates
        if len(npc_descs) == 0:
            print ("Refreshing npc_descs list from copy_npc_descs")
            npc_descs = copy_npc_descs.copy()
        self.npc = (NPC(name = npc, desc = npc_desc, current_room = self.number, llm_npc = llm_npc))

    def add_monster(self, runllm_mon = False): # only in rooms
        global monsters, copy_monsters
        monster = random.choice(monsters)
        monsters.remove(monster) # so no duplicates
        if len(monsters) == 0:
            print ("Refreshing monster list from copy_monsters")
            monsters = copy_monsters.copy()
        self.monsters.append(Monster(name = monster, health = random.randint(25, 200), damage = random.randint(10, 50), runllm_mon= runllm_mon))
    
    def populate_room(self, llm_mon = False, llm_npc = False):
        monster_count = generate_number(0,2,3,6) # 90% chance of 0, 1, or 2 monsters, 10% chance of 3, 4, or 5 monsters
        for _ in range(monster_count):
            self.add_monster(llm_mon)
        if random.randint(0, 4) == 0:  # 1 in 5 rooms has an NPC
            self.add_npc(llm_npc)
        return
        
    def inventory_room(self): #  Room inventory lists items - high level, monsters, and npc
        console = ""
        if isinstance(self, Room):
            if self.items:
                console = ("In the room you see: ")
                for item in self.items:
                    console = console + (f"{item.name}, ") # you don't know value until you get the items.    
                console = console[:-2] + (".\n")  
            if self.monsters:
                console = console + "There are monsters: "
                for monster in self.monsters:
                    console = console + (f"{monster.name} is here, it has {monster.health} health and can do {monster.damage} damage to you.\n")
            if self.npc:
                console = console + (f"{self.npc.name} is here, {self.npc.description}.\n")
        return console
    

# === ITEM TYPE SYSTEM ===
# Items are the core objects players collect, trade, and use

class Item(AdventureGame):
    """
    BASE ITEM CLASS for all collectible objects.
    
    Item types:
    - Weapon: Used for combat (has damage value)
    - Armor: Provides protection (has protection value) 
    - Treasure: Has monetary value (affects wealth score)
    - Magic: Used for healing and solving riddles (has healing value)
    
    Key mechanic: Magic items are consumed when used or when solving riddles.
    """
    def __init__(self, name, type, runllm_item = False):
        super().__init__(name=name)
        self.type = type
        if runllm_item:
            self.llM_generate_self_description()
            self.drawing_self()
   
class Weapon(Item):
    """WEAPON ITEMS for combat. Higher damage = more effective against monsters."""
    def __init__(self, name, damage, runllm=False):
        super().__init__(name = name, type = "weapon", runllm_item = runllm)
        self.damage = damage  # Damage dealt to monsters
        if self.description == None:
            self.description = (f"weapon that can deal {self.damage} damage.")

class Armor(Item):
    """ARMOR ITEMS for protection. Reduces incoming monster damage."""
    def __init__(self, name, protection, runllm=False):
        super().__init__(name, type = "armor", runllm_item = runllm)
        self.protection = protection  # Damage reduction amount
        if self.description == None:
            self.description = (f"armor that can protect you from {self.protection} damage.")

class Treasure(Item):
    """TREASURE ITEMS for wealth. Contributes to score and wealth tracking.""" 
    def __init__(self, name, value, runllm=False):
        super().__init__(name, type = "treasure", runllm_item = runllm)
        self.value = value  # Monetary value in shekels
        if self.description == None:
            self.description = (f"treasure worth {self.value} shekel.")

class Magic(Item):
    """
    MAGIC ITEMS for healing and riddle-solving.
    
    Key mechanics:
    - Used with 'use <item>' command to restore health
    - Required to solve room riddles (specific items needed per riddle)
    - Consumed when used (removed from inventory)
    - Often dropped by defeated monsters
    """
    def __init__(self, name, healing, runllm=False):
        super().__init__(name, type = "magic", runllm_item = runllm)
        self.healing = healing  # Health points restored
        if self.description == None:
            self.description = (f"magic that can heal {self.healing}.")

class LivingThing(AdventureGame):
    def __init__(self, name = "jack", current_room = None, description = None, health = None):
        super().__init__(name=name, description=description, current_room = current_room, )
        self.health = health
        #self.embeded_name = vector_model.encode(name) #could just put in adventer game class
        self.armors = []

    def add_item(self, item):
        self.items.append(item)

    def get_item_names(self):
        return ', '.join([item.name for item in self.items])

    def remove_item(self, item):
        self.items.remove(item)   

    def wear_armor(self, armor): #ok so it is a list of armors ony when you have more them on.
        self.remove_item(armor)
        self.armors.append(armor)

    def remove_armor(self, armor):
        self.armors.remove(armor)
        self.add_item(armor)    

    def move(self, direction):
        global rooms # since you are moving the NPC.
        console = ""
        if isinstance(self, NPC): # since NPC can move to a room with another NPC
            npc_new_room = rooms[self.current_room].connected_rooms[direction].number # new room number current room is just the room number of NPC
            if rooms[npc_new_room].npc: # If there is an NPC in the room
                npc_swap = rooms[npc_new_room].npc # Save the NPC in room you want to go to
                rooms[npc_new_room].npc = self # Move the current NPC to the new room
                rooms[self.current_room].npc = npc_swap # Swap the NPC in the current roomse
                console = console + (f"{self.name} went {direction} to room {npc_new_room}. {npc_swap.name} came into the room.\n")
            else:
                old_room = self.current_room
                self.current_room = npc_new_room
                rooms[npc_new_room].npc = self
                console = console + (f"{self.name} went {direction} to room {npc_new_room}.\n")
                rooms[old_room].npc = None #make sure he leaves the room :)
                print (console)
        else: # player current_room is a room class
            if direction in self.current_room.connected_rooms:
                self.current_room = self.current_room.connected_rooms[direction]
                console = (f"{self.name} moved {direction}.\n")
            else:
                console = (f"{self.name} can't go that way.\n")
        return console

    def inventory_living(self): #for living things details on all items # coulds update to do with room, and just hide some info
        response = ""
        if self.items:
            weapons = [item for item in self.items if isinstance(item, Weapon)]
            armors = [item for item in self.items if isinstance(item, Armor)]
            treasures = [item for item in self.items if isinstance(item, Treasure)]
            magic_items = [item for item in self.items if isinstance(item, Magic)]

            if not isinstance(self, Player):
                response = f"{self.name} has "
            if weapons:
                response += "Weapons: "
            for weapon in weapons:
                response += f"{weapon.name}, that can deal {weapon.damage} damage. "
            if armors:
                response += "Armors: "
            for armor in armors:
                response += f"{armor.name}, that can protect you from {armor.protection} damage. "
            if treasures:
                response += "Treasures: "
            for treasure in treasures:
                response += f"{treasure.name}, worth {treasure.value} shekel. "
            if magic_items:
                response += "Magic: "
            for magic in magic_items:
                response += f"{magic.name}, that can heal {magic.healing}. "
        else:
            response = f"{self.name} has nothing.\n"

        return response


class Player(LivingThing):
    """
    PLAYER CLASS represents the human player character.
    
    Core responsibilities:
    - Execute all game commands (move, attack, trade, solve riddles, etc.)
    - Manage inventory and equipment (weapons, armor, magic items)
    - Interact with NPCs through natural language
    - Track game statistics (health, wealth, points, kills)
    - Use LLM for intelligent item matching and command understanding
    
    Key methods:
    - attack(): Combat with monsters using weapons
    - trade_with_npc(): Item trading with NPCs  
    - solve_puzzle(): Use magic items to solve room riddles
    - take()/leave(): Inventory management with type filtering
    - _find_item_smart(): Intelligent partial name matching for items
    
    Game stats:
    - health: Player health points (starts at 300)
    - wealth: Total treasure value collected  
    - points: Score from various actions
    - kills: Number of monsters defeated
    """
    def __init__(self, name = "Jack", description = None, current_room = None , runllm_diff = False, ):
        super().__init__(name = name, description= description, current_room = current_room)
        
        # Player statistics
        self.health = 300  # Starting health
        self.points = 0    # Game score
        self.wealth = 0    # Total treasure value
        self.kills = 0     # Monsters defeated
        
        # Initialize with starting equipment
        self.equip_self()
        print ("Player current room from player class", self.current_room.number)
        
        # Generate artwork if LLM mode enabled
        if runllm_diff :
            self.llM_generate_self_description()
            self.drawing_self()
    
    def llm_describe_room(self): # calls llm for room to redescribe it usually after a change
        console =""
        console = console + self.current_room.llM_generate_self_description()
        return console
    
    def diff_draw_room(self): 
        console = self.current_room.drawing_self() # console is file name
        console = "New room image.\n"
        return console
    
    def health_up(self, health):
        if cheat_mode: # cheat mode is global
            self.health = int (health)
            console = (f"Health set to {health}.\n")
        else:
            console = ("You can't cheat without activating cheat mode with magic word.\n")
        return console

    def magic_connections(self, room_number=None):
        if room_number == None:
            room_number = self.current_room.number
        console = riddle_create_connections(room_number)
        return console

    # Removed redundant _find_item - use _find_item_smart instead
    
    def _find_item_smart(self, item_name, items):
        """Find item using intelligent partial matching prioritizing substring matches"""
        if not item_name or not items:
            return None
        
        item_name_lower = item_name.lower()
        
        # First try exact match
        for item in items:
            if item.name.lower() == item_name_lower:
                return item
        
        # Second try partial substring match (highest priority)
        for item in items:
            if item_name_lower in item.name.lower():
                return item
        
        # Third try reverse substring match (item name contained in search term)
        for item in items:
            if item.name.lower() in item_name_lower:
                return item
        
        # Finally try fuzzy match as fallback
        best_match = None
        best_score = 0
        
        for item in items:
            score = fuzz.ratio(item.name.lower(), item_name_lower)
            if score > best_score and score > 60:
                best_score = score
                best_match = item
        
        return best_match

    def _describe_single_item(self, item, addvalue = None): #full description
        console = ""
        console = (f"This is {item.name} a {item.type}, ") #full description
        if addvalue == "study":
            console = console + (f"it is a {item.description}.\n")
        if isinstance(item, Weapon):
            console = console + (f"it can deal {item.damage} damage.\n")
        elif isinstance(item, Armor):
            console = console + (f"it can protect you from {item.protection} damage.\n")
        elif isinstance(item, Treasure):
            console = console + (f"it is worth {item.value}.\n")
            if addvalue == "positive":
                self.wealth += item.value
                self.points += 5
            elif addvalue == "negative":
                self.wealth -= item.value
                self.points -= 5
        elif isinstance(item, Magic):
            console = console + (f"it can heal {item.healing}.\n")
        return console
    

    def study(self, item_name, command=None):
        """Simplified: Let LLM handle the intelligence, just execute the action"""
        
        # Check room first
        if item_name.lower() in ["room", "area", "place", "cave", "chamber", "dungeon", "passage", "tunnel", "corridor", "hallway", "around", "location"]:
            console = self.llm_describe_room()
            console = console + self.diff_draw_room()
            return console, []
            
        # Check room items
        item = self._find_item_smart(item_name, self.current_room.items)
        if item:
            console = self._describe_single_item(item, addvalue="study")
            images = [item.image] if item.image else []
            return console, images

        # Check monsters
        monster = self._find_character(item_name, self.current_room.monsters)
        if monster:
            console = (f"This {monster.name} has {monster.health} health and can do {monster.damage} damage to you. {monster.description}\n")
            images = [monster.image] if monster.image else []
            return console, images

        # Not found
        console = (f'{item_name} not in the room.\n')
        return console, []

    
    def take(self, item_names):
        """Flexible: LLM provides specific item names, with fallback for 'all' keyword"""
        if isinstance(item_names, str):
            item_names = [item_names]  # convert single string to list
        
        # Check for special "all" case first
        if any(word in str(item_names).lower() for word in ['all', 'everything']):
            return self._take_all_items()
        
        consoles = []
        item_images = []
        for item_name in item_names:
            item = self._find_item_smart(item_name, self.current_room.items)
            if item:
                self.items.append(item)
                console = self._describe_single_item(item, "positive")
                consoles.append(console)
                self.current_room.items.remove(item)
                if item.image:
                    item_images.append (item.image)
            else:
                console = ("{} isn't here.\n".format(item_name))
                consoles.append(console)
        return "\n".join(consoles), item_images
    
    def _take_all_items(self):
        """Take all items in the room"""
        if not self.current_room.items:
            return "There's nothing here to take.\n", []
        
        items_to_take = self.current_room.items[:]
        item_images = []
        console = "📦 Taking all items:\n"
        
        for item in items_to_take:
            self.items.append(item)
            self.current_room.items.remove(item)
            console += f"  • {item.name}\n"
            
            if item.image:
                item_images.append(item.image)
            
            # Update score
            if isinstance(item, Treasure):
                self.wealth += item.value
                self.points += 5
        
        console += f"\nTotal items taken: {len(items_to_take)}\n"
        return console, item_images

    def leave(self, item_names):
        """Simplified: LLM already handles type filtering, just handle 'all' case"""
        if isinstance(item_names, str):
            item_names = [item_names]  # convert single string to list
        
        # Only handle the bulk "all" case - let LLM handle type filtering  
        if any(word in str(item_names).lower() for word in ['all', 'everything']):
            return self._leave_all_items()
        
        consoles = []
        item_images = []
        items_dropped = []
        
        for item_name in item_names:
            item = self._find_item_smart(item_name, self.items)
            if item:
                self.current_room.items.append(item)
                console = self._describe_single_item(item, "negative")
                consoles.append(console)
                self.items.remove(item)
                items_dropped.append(item.name)
                if item.image:
                    item_images.append(item.image)
            else:
                suggestion = self._suggest_inventory_alternatives(item_name)
                consoles.append(suggestion)
        
        # Add summary for multiple items
        if len(items_dropped) > 2:
            consoles.append(f"\n📦 Dropped {len(items_dropped)} items total.")
        
        return "\n".join(consoles), item_images
    
    def _leave_all_items(self):
        """Drop all items"""
        if not self.items:
            return "You're not carrying anything to drop.\n", []
        
        items_to_drop = self.items[:]
        item_images = []
        console = "📦 Dropping all items:\n"
        
        for item in items_to_drop:
            self.current_room.items.append(item)
            self.items.remove(item)
            console += f"  • {item.name}\n"
            
            if item.image:
                item_images.append(item.image)
            
            # Update score
            if isinstance(item, Treasure):
                self.wealth -= item.value
                self.points -= 5
        
        console += f"\nTotal items dropped: {len(items_to_drop)}\n"
        return console, item_images
    
    def _leave_all_of_type(self, item_type):
        """Drop all items of a specific type"""
        items_of_type = [i for i in self.items if isinstance(i, item_type)]
        
        if not items_of_type:
            type_name = item_type.__name__.lower()
            return f"You have no {type_name}s to drop.\n", []
        
        item_images = []
        console = f"📦 Dropping all {item_type.__name__.lower()}s:\n"
        
        for item in items_of_type:
            self.current_room.items.append(item)
            self.items.remove(item)
            console += f"  • {item.name}\n"
            
            if item.image:
                item_images.append(item.image)
            
            # Update score
            if isinstance(item, Treasure):
                self.wealth -= item.value
                self.points -= 5
        
        console += f"\nTotal dropped: {len(items_of_type)}\n"
        return console, item_images
    
    def _suggest_inventory_alternatives(self, attempted_name):
        """Suggest alternatives from inventory"""
        if not self.items:
            return f"You're not carrying anything.\n"
        
        item_names = [i.name for i in self.items[:10]]
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Player tried to drop "{attempted_name}" but doesn't have that exact item.
Their inventory: {', '.join(item_names)}

Suggest what they might have meant in 1-2 sentences.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        suggestion = ollama_generate(prompt, max_tokens=80, temperature=0.3)
        return f"You don't have '{attempted_name}'.\n{suggestion}\n"

    def solve_puzzle(self, item1_name, item2_name=None):
        """Enhanced puzzle solving with LLM understanding"""
        console = ""
        print (item1_name, item2_name)
        
        if self.current_room.riddle_solved:
            return "You already solved the riddle!\n"
        
        if not self.current_room.riddles:
            return self._suggest_riddle_locations()
        
        # Use smart item finding for both items
        item1 = self._find_item_smart(item1_name, self.items)
        
        if not item1:
            return self._suggest_magic_items_for_riddle(item1_name)
        
        if not isinstance(item1, Magic):
            return f"'{item1.name}' is not a magic item. You need magic items to solve riddles.\n" + self._list_magic_items()
        
        item2 = None
        if item2_name:
            item2 = self._find_item_smart(item2_name, self.items)
            if item2 and not isinstance(item2, Magic):
                print ("item2 not magic")
                item2 = None

        riddle_items = self.current_room.riddle_magic_items
        print (riddle_items, item1.name, item2_name if item2 else "None")
        
        # Check if solution is correct (order agnostic)
        items_used = [item1.name]
        if item2:
            items_used.append(item2.name)
        
        solution_correct = False
        if len(riddle_items) == 1 and item1.name in riddle_items:
            solution_correct = True
        elif len(riddle_items) == 2 and item2 and set(items_used) == set(riddle_items):
            solution_correct = True
        
        if solution_correct:
            self.current_room.riddle_solved = True
            console = f"✨ {item1.name} evaporates into thin air!\n"
            self.items.remove(item1)
            if item2:
                console += f"✨ {item2.name} evaporates into thin air!\n"
                self.items.remove(item2)
            console += f"\n🎉 You solved the riddle: {self.current_room.riddles[0]}!\n"
            console += riddle_create_connections(self.current_room.number)
            console += f"\n🔓 Secret Unlocked: Say '{magicword}' to open more passages.\n"
            self.points += item1.healing * 10
            self.health += item1.healing * 3
            if item2:
                self.points += item2.healing * 10
                self.health += item2.healing * 3
        else:
            # Use LLM to provide intelligent hints
            console = self._get_riddle_hint(item1, item2, riddle_items)
        
        return console
    
    def _suggest_riddle_locations(self):
        """Suggest where to find riddles"""
        riddle_rooms = [r for r in rooms.values() if r.riddles and not r.riddle_solved]
        
        if not riddle_rooms:
            return "All riddles have been solved! You're amazing!\n"
        
        console = "There's no riddle in this room.\n\n"
        console += f"🔍 There are {len(riddle_rooms)} unsolved riddles in other rooms.\n"
        console += "💡 Go to Room 1 and ask MB where to find riddles!\n"
        
        return console
    
    def _suggest_magic_items_for_riddle(self, attempted_name):
        """Suggest magic items the player has"""
        magic_items = [i for i in self.items if isinstance(i, Magic)]
        
        if not magic_items:
            console = f"You don't have '{attempted_name}' or any magic items.\n\n"
            console += "💡 To get magic items:\n"
            console += "  • Kill monsters - they often drop magic items\n"
            console += "  • Trade with NPCs\n"
            console += "  • Explore rooms\n"
            console += "  • Ask MB in Room 1 where to find specific items!\n"
            return console
        
        console = f"You don't have '{attempted_name}'.\n\n"
        console += "✨ Your magic items:\n"
        for item in magic_items:
            console += f"  • {item.name}\n"
        console += "\nTry using one of these!\n"
        
        return console
    
    def _list_magic_items(self):
        """List player's magic items"""
        magic_items = [i for i in self.items if isinstance(i, Magic)]
        
        if not magic_items:
            return "\nYou have no magic items. Find some first!\n"
        
        console = "\n✨ Your magic items:\n"
        for item in magic_items:
            console += f"  • {item.name}\n"
        return console
    
    def _get_riddle_hint(self, item1, item2, required_items):
        """Use LLM to provide contextual hints that connect riddle to items"""
        riddle_text = self.current_room.riddles[0]
        hint_text = self.current_room.hints[0] if self.current_room.hints else ""
        
        items_tried = [item1.name]
        if item2:
            items_tried.append(item2.name)
        
        # Enhanced prompt that helps LLM understand the riddle-to-item connection
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helping a player solve a riddle in an adventure game. The riddle has an answer, and the player needs magic items that represent or relate to that answer.

Riddle: "{riddle_text}"
Game's hint: "{hint_text}"
Required magic items: {', '.join(required_items)}
Player tried using: {', '.join(items_tried)}

Your job:
1. First, think about what the riddle's answer is
2. Explain why the required magic items relate to that answer
3. Give a hint that guides them toward the right items WITHOUT saying the exact names

Keep it encouraging, mystical, and under 3 sentences.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Provide a helpful hint.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        hint = ollama_generate(prompt, max_tokens=150, temperature=0.5)
        
        console = f"❌ That's not quite right.\n\n💭 {hint}\n"
        
        # Add specific guidance based on situation
        if len(required_items) == 2 and not item2:
            console += "\n⚠️ Remember: This riddle needs TWO magic items combined!\n"
        
        # In easy mode, be more explicit about the connection
        if difficulty_g in ["easy", "cheat"]:
            console += f"\n📝 Game hint: {hint_text}\n"
            console += f"🎯 Think about items that relate to the riddle's answer...\n"
        
        return console
        
    def use(self, item_name):
        item_images = []
        item = self._find_item_smart(item_name, self.items)
        if item:
            if isinstance(item, Magic):
                self.health += item.healing
                self.points -= 5
                self.items.remove(item) # you need to remove it after you check for image
                console = (f"You used the {item.name} and restored {item.healing} health!")
                if item.image:
                    item_images.append(item.image)
                    return console, item_images
                else:
                    return console
            else:
                console = ("You can't use that item.")
        else:
            console = ("You don't have that item.")     
        return console
      
    def put_on_armor(self, armor_name):
        armor = self._find_item_smart(armor_name, self.items)
        if armor and isinstance(armor, Armor):
            self.wear_armor(armor)
            console = (f"You put on the {armor.name}.")
        else:
            if armor and not isinstance(armor, Armor):
                console = ("That is not armor!")
            else:
                console = ("You don't have that.")
        return console
    
    def take_off_armor(self, armor_name):
        armor = self._find_item_smart(armor_name, self.armors)
        if armor:
            self.remove_armor(armor)
            console = (f"You took off the {armor.name}.")
        else:
            console = ("You don't have that armor on.")
        return console
    
    def _find_character(self, name, characters):
        for character in characters:
            if fuzz.ratio(character.name.lower(), name.lower()) > 60 or name.lower() in character.name.lower():
                return character
        return None
    
    def about_to_npc(self, input_text= "what do you want to know about this world"):
        console=""
        npc = self.current_room.npc
        if npc:
            console = npc.llM_generate_self_info(dialogue = input_text)
        else:
            console = "No one in room to talk to.\n"
        return console # do you want to use second return for image?

    def talk_to_npc(self, input_text= "what do you have to trade"):
        console=""
        npc = self.current_room.npc
        my_stuff = self.get_item_names()
        if npc:
            console = npc.llM_generate_self_dialogue(dialogue = input_text, my_stuff = my_stuff)
        else:
            console = "No one in room to talk to.\n"
        return console # do you want to use second return for image?
    
    def get_npc_quest_suggestion(self):
        """Get quest suggestions from NPCs based on game state"""
        console = ""
        npc = self.current_room.npc
        if not npc:
            return ""
        
        # Ensure NPC has all attributes
        npc._ensure_attributes()
        
        # Check various game states and provide relevant quests
        unsolved_riddles = [r for r in rooms.values() if r.riddles and not r.riddle_solved]
        remaining_treasures = [r for r in rooms.values() if any(isinstance(i, Treasure) for i in r.items)]
        remaining_monsters = [r for r in rooms.values() if r.monsters]
        
        if unsolved_riddles and npc.personality_traits['knowledge'] in ['knows everything', 'knows riddles well']:
            riddle_room = random.choice(unsolved_riddles)
            console = f"{npc.name} mentions: 'I've heard there's an unsolved riddle in room {riddle_room.number}. "
            if npc.personality_traits['helpfulness'] in ['very helpful', 'eager to help']:
                console += f"The riddle requires {', '.join(riddle_room.riddle_magic_items)}. "
            console += "Perhaps you should investigate.'\n"
        
        elif remaining_monsters and npc.personality_traits['mood'] in ['friendly', 'cheerful']:
            monster_room = random.choice(remaining_monsters)
            console = f"{npc.name} warns: 'Be careful! I've heard there are {len(monster_room.monsters)} monsters in room {monster_room.number}.'\n"
        
        elif remaining_treasures and npc.personality_traits['trading_style'] == 'generous trader':
            treasure_room = random.choice(remaining_treasures)
            console = f"{npc.name} whispers: 'I've heard rumors of treasure in room {treasure_room.number}...'\n"
        
        return console

    def get_trading_hints(self):
        """Get hints about what the NPC wants to trade"""
        npc = self.current_room.npc
        if not npc:
            return "No one here to trade with.\n"
        
        # Ensure NPC has all attributes
        npc._ensure_attributes()
        
        console = f"{npc.name} is a {npc.personality_traits['trading_style']}.\n"
        
        # Analyze what the NPC might want
        npc_wants = []
        
        # Check NPC's current inventory
        has_weapons = any(isinstance(i, Weapon) for i in npc.items)
        has_armor = any(isinstance(i, Armor) for i in npc.items)
        has_magic = any(isinstance(i, Magic) for i in npc.items)
        has_treasure = any(isinstance(i, Treasure) for i in npc.items)
        
        # NPCs want what they don't have
        if not has_weapons:
            npc_wants.append("weapons")
        if not has_armor:
            npc_wants.append("armor")
        if not has_treasure:
            npc_wants.append("valuable treasures")
        
        # Based on personality
        if npc.personality_traits['trading_style'] == 'selective trader':
            if has_magic:
                console += f"{npc.name} says: 'I'm particularly interested in other magic items.'\n"
            else:
                console += f"{npc.name} says: 'I'm looking for specific items, especially magic ones.'\n"
        elif npc.personality_traits['trading_style'] == 'hard bargainer':
            console += f"{npc.name} says: 'I only trade for items of greater value than what I offer.'\n"
        elif npc.personality_traits['trading_style'] == 'generous trader':
            console += f"{npc.name} says: 'I'm happy to trade fairly. Show me what you have!'\n"
        
        if npc_wants:
            console += f"{npc.name} seems interested in: {', '.join(npc_wants)}.\n"
        
        # Show what they have to trade
        if npc.items:
            valuable_items = [i for i in npc.items if isinstance(i, (Magic, Weapon)) or (isinstance(i, Treasure) and i.value > 200)]
            if valuable_items:
                console += f"{npc.name} has some interesting items: {', '.join([i.name for i in valuable_items[:3]])}.\n"
        
        return console

    def trade_with_npc(self, my_item_name=None, their_item_name = None): # updated with response
            console = ""
            npc = self.current_room.npc
            if npc:
                my_item = self._find_item_smart(my_item_name, self.items)
                their_item = self._find_item_smart(their_item_name, npc.items)
                if my_item and their_item:
                    # Check if NPC considers the trade fair
                    if npc.evaluate_trade(my_item, their_item):
                        self.items.remove(my_item)
                        console = self._describe_single_item(my_item, "negative") # just using to adjust wealth & points
                        npc.items.remove(their_item)
                        self.items.append(their_item)
                        console = console + self._describe_single_item(their_item, "positive")
                        npc.items.append(my_item)
                        
                        # Remember the trade
                        npc._ensure_attributes()  # Ensure NPC has all attributes
                        npc.player_trades.append({"gave": their_item.name, "received": my_item.name})
                        npc._remember_interaction("trade", f"Traded {their_item.name} for {my_item.name}")
                        
                        # Generate appropriate response based on personality
                        if npc.personality_traits['trading_style'] == 'generous trader':
                            console = console + (f'{npc.name} says: "A pleasure doing business with you! Use {their_item.name} wisely."\n')
                        elif npc.personality_traits['trading_style'] == 'hard bargainer':
                            console = console + (f'{npc.name} says: "Finally, a reasonable offer. The {their_item.name} is yours."\n')
                        else:
                            console = console + (f'{npc.name} says: "Thank you for the trade, you now have the {their_item.name} and I have the {my_item.name}."\n')
                    else:
                        # NPC rejects unfair trade
                        if npc.personality_traits['trading_style'] == 'hard bargainer':
                            console = (f"{npc.name} scoffs: 'That's insulting! {my_item.name} for {their_item.name}? Come back with a better offer.'\n")
                        elif npc.personality_traits['trading_style'] == 'selective trader':
                            console = (f"{npc.name} shakes their head: 'I'm looking for something more specific than {my_item.name}.'\n")
                        else:
                            console = (f"{npc.name} says: 'I'm sorry, but that doesn't seem like a fair trade to me.'\n")
                else:
                    console = ("Don't try to cheat me!\n")      
            else:
                console = ("There is no one here to trade with.\n")
            return console

    def attack(self, monster_name, weapon_name=None):  
        """Enhanced attack with intelligent weapon matching and LLM understanding"""
        global you_started_it
        
        # Use smart weapon selection if weapon not specified
        if not weapon_name:
            weapon_name = self._get_best_weapon_for_combat()
            if not weapon_name:
                return "You have no weapons! Find one before fighting.\n"
        
        monster = self._find_character(monster_name, self.current_room.monsters)
        if monster:
            # Use smart item finding for better weapon matching
            weapon = self._find_item_smart(weapon_name, self.items)
            if weapon and isinstance(weapon, Weapon):
                you_started_it = True
                
                # Calculate damage with armor consideration
                damage = self._calculate_combat_damage(weapon, monster)
                monster.health -= damage
                
                if monster.health <= 0:
                    console = (f"!! You killed the {monster.name} with {weapon.name}! It dropped {monster.get_item_names()}\n")
                    console = console + monster.description + "\n"
                    self.current_room.items.extend(monster.items)
                    self.current_room.monsters.remove(monster)
                    self.kills += 1
                    self.points += 10
                    
                    # Check for special drops
                    if any(isinstance(item, Magic) for item in monster.items):
                        console += "💎 The monster dropped magic items! These can solve riddles!\n"
                else:
                    console = (f"!! You hit the {monster.name} with {weapon.name} for {damage} damage! Monster health: {monster.health}\n")
                    
                    # Provide combat tips
                    if monster.health > 100:
                        console += "💡 This monster is tough! Consider using your best weapon.\n"
            else:
                console = self._suggest_combat_alternatives(monster_name)
        else:
            # Use LLM to suggest what they might have meant
            console = self._suggest_monster_alternatives(monster_name)
        return console
    
    def _get_best_weapon_for_combat(self):
        """Get the best available weapon"""
        weapons = [item for item in self.items if isinstance(item, Weapon)]
        if weapons:
            best_weapon = max(weapons, key=lambda w: w.damage)
            return best_weapon.name
        return None
    
    def _calculate_combat_damage(self, weapon, monster):
        """Calculate damage considering various factors"""
        base_damage = weapon.damage
        
        # Could add modifiers based on monster type, player stats, etc.
        # For now, just return base damage
        return base_damage
    
    def _suggest_combat_alternatives(self, monster_name):
        """Use LLM to suggest combat alternatives"""
        available_items = [item.name for item in self.items]
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Player tried to attack {monster_name} but doesn't have the right weapon.
Their items: {', '.join(available_items[:10])}

Suggest what they could use or do instead in 2-3 sentences.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        suggestion = ollama_generate(prompt, max_tokens=100, temperature=0.3)
        return f"You can't use that to attack.\n{suggestion}\n"
    
    def _suggest_monster_alternatives(self, attempted_name):
        """Use LLM to suggest which monster the player meant"""
        actual_monsters = [m.name for m in self.current_room.monsters]
        
        if not actual_monsters:
            return "There are no monsters in this room.\n"
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Player tried to attack "{attempted_name}" but that's not here.
Actual monsters in room: {', '.join(actual_monsters)}

Suggest what they might have meant in 1-2 sentences.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        suggestion = ollama_generate(prompt, max_tokens=80, temperature=0.3)
        return f"No '{attempted_name}' here.\n{suggestion}\n"
    
    
class NPC(LivingThing):
    def __init__(self, name, desc, current_room, llm_npc = False):
        super().__init__(name = name, description = desc, current_room = current_room)
        self.health = 100
        self.equip_self()
        # Add memory system for NPCs
        self.memory = []  # Store past interactions
        self.player_trades = []  # Remember what player has traded
        self.times_talked = 0  # Track conversation count
        self.personality_traits = self._generate_personality()  # Give each NPC unique traits
        # Add conversation state tracking
        self.conversation_state = {
            "pending_action": None,  # What action is waiting for confirmation
            "last_offer": None,      # Details of the last offer made
            "context": {}            # Additional context for the pending action
        }
        if llm_npc:
            self.drawing_self()
            self.llM_generate_self_description()
    
    def _generate_personality(self):
        """Generate personality traits for more interesting NPCs"""
        traits = {
            "helpfulness": random.choice(["very helpful", "somewhat helpful", "reluctant to help", "eager to help"]),
            "knowledge": random.choice(["knows everything", "knows riddles well", "knows item locations", "knows monster weaknesses"]),
            "trading_style": random.choice(["fair trader", "hard bargainer", "generous trader", "selective trader"]),
            "mood": random.choice(["cheerful", "grumpy", "mysterious", "friendly", "suspicious"])
        }
        return traits

    def _remember_interaction(self, interaction_type, details):
        """Store interaction in NPC memory"""
        # Ensure attributes exist
        self._ensure_attributes()
        memory_entry = {
            "type": interaction_type,
            "details": details,
            "room": self.current_room,
            "timestamp": self.times_talked,
            "player_health": playerOne.health if playerOne else 0,
            "player_items_count": len(playerOne.items) if playerOne else 0
        }
        self.memory.append(memory_entry)
        # Keep only last 10 interactions to avoid memory bloat
        if len(self.memory) > 10:
            self.memory.pop(0)

    def _get_memory_context(self):
        """Get formatted memory for LLM context"""
        # Ensure attributes exist
        self._ensure_attributes()
        if not self.memory:
            return "This is our first meeting."
        
        memory_str = f"We've talked {self.times_talked} times before.\n"
        memory_str += "Recent interactions:\n"
        for i, mem in enumerate(self.memory[-5:], 1):  # Last 5 interactions
            memory_str += f"{i}. {mem['type']}: {mem['details']}"
            if mem['type'] == 'dialogue':
                memory_str += f" (when you had {mem['player_health']} health)"
            memory_str += "\n"
        
        # Add trade history if any
        if self.player_trades:
            memory_str += f"\nWe've traded {len(self.player_trades)} times:\n"
            for trade in self.player_trades[-3:]:  # Last 3 trades
                memory_str += f"- I gave you {trade['gave']} for your {trade['received']}\n"
        
        return memory_str

    def _get_game_knowledge(self):
        """Get comprehensive game knowledge for NPC"""
        knowledge = {
            "complete_instructions": get_complete_instructions(),
            "riddles": riddles_and_magic(),
            "total_rooms": len(rooms),
            "player_location": playerOne.current_room.number if playerOne else "unknown",
            "nearby_dangers": self._check_nearby_dangers(),
            "valuable_items": self._identify_valuable_items(),
            "player_progress": self._assess_player_progress(),
            "all_treasures": self._get_all_treasures(),
            "all_magic_items": self._get_all_magic_items(),
            "all_weapons": self._get_all_weapons(),
            "all_monsters": self._get_all_monsters_info(),
            "all_npcs": self._get_all_npcs_info(),
            "room_connections": self._get_room_connections()
        }
        return knowledge

    def _check_nearby_dangers(self):
        """Check for monsters in connected rooms"""
        dangers = []
        current_room = rooms.get(self.current_room)
        if current_room:
            for direction, connected_room in current_room.connected_rooms.items():
                if connected_room.monsters:
                    monster_names = [m.name for m in connected_room.monsters]
                    dangers.append(f"{direction}: {', '.join(monster_names)}")
        return dangers

    def _identify_valuable_items(self):
        """Identify valuable items the NPC knows about"""
        valuable = []
        # Check own inventory
        for item in self.items:
            if isinstance(item, Magic) or (isinstance(item, Treasure) and item.value > 500):
                valuable.append(f"I have {item.name}")
        
        # Check current room
        current_room = rooms.get(self.current_room)
        if current_room:
            for item in current_room.items:
                if isinstance(item, Magic):
                    valuable.append(f"This room has {item.name}")
        
        return valuable

    def _assess_player_progress(self):
        """Assess how well the player is doing"""
        if not playerOne:
            return "unknown"
        
        progress = {
            "health_status": "healthy" if playerOne.health > 200 else "injured" if playerOne.health > 100 else "critical",
            "wealth_level": "rich" if playerOne.wealth > 1000 else "moderate" if playerOne.wealth > 500 else "poor",
            "combat_ready": len([i for i in playerOne.items if isinstance(i, Weapon)]) > 0,
            "riddle_ready": len([i for i in playerOne.items if isinstance(i, Magic)]) > 0
        }
        return progress
    
    def _get_all_treasures(self):
        """Get locations of all treasures in the game"""
        treasures_info = []
        for room in rooms.values():
            # Room treasures
            room_treasures = [i for i in room.items if isinstance(i, Treasure)]
            for treasure in room_treasures:
                treasures_info.append(f"Room {room.number}: {treasure.name} (value: {treasure.value})")
            
            # NPC treasures
            if room.npc:
                npc_treasures = [i for i in room.npc.items if isinstance(i, Treasure)]
                for treasure in npc_treasures:
                    treasures_info.append(f"Room {room.number} - {room.npc.name} has: {treasure.name} (value: {treasure.value})")
            
            # Monster treasures
            for monster in room.monsters:
                monster_treasures = [i for i in monster.items if isinstance(i, Treasure)]
                for treasure in monster_treasures:
                    treasures_info.append(f"Room {room.number} - {monster.name} has: {treasure.name} (value: {treasure.value})")
        
        return treasures_info
    
    def _get_all_magic_items(self):
        """Get locations of all magic items in the game"""
        magic_info = []
        for room in rooms.values():
            # Room magic items
            room_magic = [i for i in room.items if isinstance(i, Magic)]
            for item in room_magic:
                magic_info.append(f"Room {room.number}: {item.name} (healing: {item.healing})")
            
            # NPC magic items
            if room.npc:
                npc_magic = [i for i in room.npc.items if isinstance(i, Magic)]
                for item in npc_magic:
                    magic_info.append(f"Room {room.number} - {room.npc.name} has: {item.name} (healing: {item.healing})")
            
            # Monster magic items
            for monster in room.monsters:
                monster_magic = [i for i in monster.items if isinstance(i, Magic)]
                for item in monster_magic:
                    magic_info.append(f"Room {room.number} - {monster.name} has: {item.name} (healing: {item.healing})")
        
        return magic_info
    
    def _get_all_weapons(self):
        """Get locations of all weapons in the game"""
        weapons_info = []
        for room in rooms.values():
            # Room weapons
            room_weapons = [i for i in room.items if isinstance(i, Weapon)]
            for weapon in room_weapons:
                weapons_info.append(f"Room {room.number}: {weapon.name} (damage: {weapon.damage})")
            
            # NPC weapons
            if room.npc:
                npc_weapons = [i for i in room.npc.items if isinstance(i, Weapon)]
                for weapon in npc_weapons:
                    weapons_info.append(f"Room {room.number} - {room.npc.name} has: {weapon.name} (damage: {weapon.damage})")
            
            # Monster weapons
            for monster in room.monsters:
                monster_weapons = [i for i in monster.items if isinstance(i, Weapon)]
                for weapon in monster_weapons:
                    weapons_info.append(f"Room {room.number} - {monster.name} has: {weapon.name} (damage: {weapon.damage})")
        
        return weapons_info
    
    def _get_all_monsters_info(self):
        """Get information about all monsters in the game"""
        monsters_info = []
        for room in rooms.values():
            for monster in room.monsters:
                items_str = ", ".join([i.name for i in monster.items]) if monster.items else "nothing"
                monsters_info.append(f"Room {room.number}: {monster.name} (health: {monster.health}, damage: {monster.damage}, carrying: {items_str})")
        return monsters_info
    
    def _get_all_npcs_info(self):
        """Get information about all NPCs in the game"""
        npcs_info = []
        for room in rooms.values():
            if room.npc:
                items_str = ", ".join([i.name for i in room.npc.items]) if room.npc.items else "nothing"
                npcs_info.append(f"Room {room.number}: {room.npc.name} (carrying: {items_str})")
        return npcs_info
    
    def _get_room_connections(self):
        """Get all room connections in the game"""
        connections_info = []
        for room in rooms.values():
            connected = ", ".join([f"{dir}: Room {r.number}" for dir, r in room.connected_rooms.items()])
            if connected:
                connections_info.append(f"Room {room.number} connects to: {connected}")
            else:
                connections_info.append(f"Room {room.number} has no connections (solve riddle to open passages)")
        return connections_info

    #could use our get entities to get list of items to trade
    def _ensure_attributes(self):
        """Ensure all new attributes exist for backward compatibility"""
        if not hasattr(self, 'memory'):
            self.memory = []
        if not hasattr(self, 'player_trades'):
            self.player_trades = []
        if not hasattr(self, 'times_talked'):
            self.times_talked = 0
        if not hasattr(self, 'personality_traits'):
            self.personality_traits = self._generate_personality()
        if not hasattr(self, 'conversation_state'):
            self.conversation_state = {
                "pending_action": None,
                "last_offer": None,
                "context": {}
            }

    def set_pending_action(self, action_type, offer_details, context=None):
        """Set a pending action that awaits player confirmation"""
        self._ensure_attributes()
        self.conversation_state = {
            "pending_action": action_type,
            "last_offer": offer_details,
            "context": context or {}
        }
        print(f"DEBUG: Set pending action for {self.name}: {action_type} - {offer_details}")

    def clear_pending_action(self):
        """Clear any pending action"""
        self._ensure_attributes()
        self.conversation_state = {
            "pending_action": None,
            "last_offer": None,
            "context": {}
        }

    def has_pending_action(self):
        """Check if there's a pending action waiting for confirmation"""
        self._ensure_attributes()
        return self.conversation_state.get("pending_action") is not None

    def handle_conversation_response(self, response):
        """Handle player's response to pending conversation using LLM interpretation"""
        self._ensure_attributes()
        
        if not self.has_pending_action():
            return None
        
        pending_action = self.conversation_state["pending_action"]
        last_offer = self.conversation_state["last_offer"]
        context = self.conversation_state["context"]
        
        # Use LLM to interpret the player's response in context
        system_prompt = f"""You are analyzing a player's response to an NPC's offer in an adventure game.

The NPC ({self.name}) just made this offer: "{last_offer}"
The pending action type is: {pending_action}

Your job is to determine the player's intent from their response and return ONE of these actions:
- "accept" - Player wants to proceed with the offer
- "decline" - Player doesn't want the offer  
- "clarify" - Player wants more information or is asking questions
- "defer" - Player wants to think about it or do it later
- "unclear" - Response is ambiguous or unrelated

Consider natural language variations like:
- "That sounds great!" = accept
- "I'm not interested" = decline  
- "Tell me more about that item" = clarify
- "Maybe later" = defer
- "What's the weather like?" = unclear

Be flexible and understand context. Return ONLY the action word."""

        user_prompt = f"""Player's response: "{response}"

What is the player's intent?"""

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """
        
        # Get LLM interpretation
        interpretation = ollama_generate(prompt, max_tokens=50, temperature=0.1).strip().lower()
        print(f"DEBUG: LLM interpreted '{response}' as: {interpretation}")
        
        # Handle the response based on LLM interpretation and pending action
        if interpretation == "accept":
            if pending_action == "trade":
                my_item = context.get("my_item")
                their_item = context.get("their_item")
                
                if my_item and their_item:
                    result = playerOne.trade_with_npc(my_item.name, their_item.name)
                    self.clear_pending_action()
                    return result
                else:
                    self.clear_pending_action()
                    return f"{self.name} says: 'Something went wrong with the trade setup. Let's try again.'\n"
            
            elif pending_action == "give_item":
                item_to_give = context.get("item")
                if item_to_give:
                    self.items.remove(item_to_give)
                    playerOne.items.append(item_to_give)
                    self.clear_pending_action()
                    return f"{self.name} gives you {item_to_give.name}! {item_to_give.description}\n"
                else:
                    self.clear_pending_action()
                    return f"{self.name} says: 'I can't seem to find that item anymore.'\n"
            
            elif pending_action == "riddle_help":
                riddle_solution = context.get("solution")
                if riddle_solution:
                    self.clear_pending_action()
                    return f"{self.name} whispers: '{riddle_solution}'\n"
                else:
                    self.clear_pending_action()
                    return f"{self.name} says: 'Let me think about that riddle...'\n"
            
            elif pending_action == "quest":
                quest_details = context.get("quest_details")
                if quest_details:
                    self.clear_pending_action()
                    # Generate a helpful quest response
                    return f"{self.name} explains: '{quest_details}'\n\n💡 Quest accepted! {quest_details}\n"
                else:
                    self.clear_pending_action()
                    return f"{self.name} says: 'Let me think of a good quest for you...'\n"
            
            elif pending_action == "information":
                info_details = context.get("info_details")
                if info_details:
                    self.clear_pending_action()
                    # Generate helpful information
                    return f"{self.name} explains: {info_details}\n"
                else:
                    self.clear_pending_action()
                    return f"{self.name} says: 'What would you like to know?'\n"
        
        elif interpretation == "decline":
            self.clear_pending_action()
            if self.personality_traits['mood'] == 'friendly':
                return f"{self.name} says: 'No problem at all! Let me know if you change your mind.'\n"
            elif self.personality_traits['mood'] == 'grumpy':
                return f"{self.name} grumbles: 'Fine, your loss.'\n"
            else:
                return f"{self.name} says: 'I understand. Perhaps another time.'\n"
        
        elif interpretation == "clarify":
            # Player wants more information - provide it and keep pending action
            if pending_action == "trade":
                my_item = context.get("my_item")
                their_item = context.get("their_item")
                if my_item and their_item:
                    return f"{self.name} explains: 'I'm offering to trade my {their_item.name} ({their_item.description}) for your {my_item.name}. It's a fair deal! So, would you like to make this trade?'\n"
            elif pending_action == "give_item":
                item_to_give = context.get("item")
                if item_to_give:
                    return f"{self.name} explains: 'This is {item_to_give.name} - {item_to_give.description}. I think it could help you! Would you like me to give it to you?'\n"
            elif pending_action == "riddle_help":
                return f"{self.name} says: 'I can tell you exactly which magic items you need to solve the riddle in this room. Would you like me to share that information?'\n"
            
            return f"{self.name} says: 'What would you like to know about {last_offer}?'\n"
        
        elif interpretation == "defer":
            # Player wants to think about it - clear pending action but be encouraging
            self.clear_pending_action()
            return f"{self.name} says: 'Take your time! I'll be here when you're ready.'\n"
        
        else:  # unclear or unrelated
            # Continue conversation naturally while keeping pending action
            return self.llM_generate_self_dialogue(response, playerOne.get_item_names())

    def llM_generate_self_dialogue(self, dialogue="what do you have to trade", my_stuff=None):
        # Ensure all attributes exist for backward compatibility
        self._ensure_attributes()
        self.times_talked += 1
        
        # Get comprehensive context
        game_knowledge = self._get_game_knowledge()
        memory_context = self._get_memory_context()
        
        # Analyze player inventory for smart trading
        player_needs = self._analyze_player_needs(my_stuff)
        
        system_prompt = f"""You are {self.description} in an adventure game. 
Your personality: {self.personality_traits['mood']} and {self.personality_traits['helpfulness']}.
You are a {self.personality_traits['trading_style']} who {self.personality_traits['knowledge']}.

{memory_context}

Current game state:
- Player is in room {game_knowledge['player_location']}
- Player health status: {game_knowledge['player_progress']['health_status']}
- Nearby dangers: {', '.join(game_knowledge['nearby_dangers']) if game_knowledge['nearby_dangers'] else 'None'}

You know about all riddles in the game and can give hints.
When trading, consider what the player needs based on their inventory and game progress.
Be consistent with your personality and remember past interactions.

IMPORTANT: 
- If asked about general game mechanics or help, provide useful guidance
- If the player seems confused, offer helpful suggestions
- Stay in character but be genuinely helpful when needed
- If you notice the player is struggling (low health, no weapons, etc.), offer advice
- When making specific trade offers, ask for confirmation: "Would you like to trade your X for my Y?"
- When offering to give items, ask for confirmation: "Would you like me to give you this X?"
- When offering riddle help, ask for confirmation: "Would you like me to tell you the solution?"
- If player expresses interest in trading but doesn't use proper format, guide them: "To trade, use: trade <your item> for <my item>"
- You can discuss trades naturally - if they ask about your items or express interest, engage in conversation"""

        user_prompt = f"""{dialogue}
My inventory: {my_stuff}
Your inventory: {self.get_item_names()}

Player might need: {', '.join(player_needs) if player_needs else 'Nothing specific identified'}

Additional context:
- Player has {len([i for i in playerOne.items if isinstance(i, Magic)])} magic items
- Player has visited {len([r for r in rooms.values() if r.explored])} rooms
- There are {len([r for r in rooms.values() if r.riddles and not r.riddle_solved])} unsolved riddles"""
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """
        
        response_text = ollama_generate(prompt, max_tokens=256, temperature=0.7)
        formatted_response = f"{self.name} says: {response_text}\n"
        
        # Remember this interaction
        self._remember_interaction("dialogue", dialogue)
        
        # Detect if NPC is making a specific offer and set pending action
        self._detect_and_set_pending_action(response_text, dialogue, my_stuff)
        
        # Also check if player is expressing trade intent
        if self._detect_player_trade_intent(dialogue, my_stuff):
            formatted_response += "\n💡 You can trade naturally! Try any of these:\n"
            formatted_response += "• 'I'll trade my sword for your shield'\n"
            formatted_response += "• 'Give you my potion for your ring'\n"
            formatted_response += "• 'I want your armor for my treasure'\n"
            formatted_response += "• Or simply: 'trade sword for shield'\n"
        
        # Simplified: Show inventory if trade-related keywords are mentioned
        if any(word in formatted_response.lower() for word in ["trade", "exchange", "swap", "give", "want"]):
            formatted_response += self.inventory_living() + "\n"
            # Add trading suggestions based on player needs
            suggestions = self._suggest_trades(my_stuff)
            if suggestions:
                formatted_response += f"Trading suggestions: {suggestions}\n"
        
        # Add contextual help based on player situation
        if self.personality_traits['helpfulness'] in ['very helpful', 'eager to help']:
            if playerOne.health < 100:
                formatted_response += f"\n{self.name} notices your poor health and adds: 'You should find healing items quickly!'\n"
            if not any(isinstance(i, Weapon) for i in playerOne.items):
                formatted_response += f"\n{self.name} warns: 'You have no weapons! That's dangerous in these caves.'\n"
        
        return formatted_response

    def _detect_and_set_pending_action(self, response_text, dialogue, my_stuff):
        """Use LLM to detect if NPC is making an offer and set appropriate pending action"""
        print(f"DEBUG: Detecting pending actions in response: {response_text[:100]}...")
        
        # Use LLM to analyze if the NPC is making a specific offer
        system_prompt = f"""You are analyzing an NPC's response in an adventure game to detect if they're making a specific offer that requires player confirmation.

NPC Name: {self.name}
Player's original message: "{dialogue}"
NPC's response: "{response_text}"

Available player items: {[item.name for item in playerOne.items] if playerOne.items else "None"}
Available NPC items: {[item.name for item in self.items] if self.items else "None"}

Analyze if the NPC is making any of these types of offers:
1. TRADE - Offering to exchange one item for another (e.g., "Would you like to trade your sword for my potion?")
2. GIVE_ITEM - Offering to give an item for free (e.g., "Would you like me to give you this ring?")
3. RIDDLE_HELP - Offering to provide riddle solutions (e.g., "Should I tell you how to solve this riddle?")
4. QUEST - Offering a quest or task (e.g., "Would you like me to tell you where to find the dragon?")
5. INFORMATION - Offering specific game information (e.g., "Should I explain how magic items work?")
6. NONE - Just having a conversation, no specific offer being made

Return a JSON response with:
{{
    "offer_type": "TRADE" | "GIVE_ITEM" | "RIDDLE_HELP" | "QUEST" | "INFORMATION" | "NONE",
    "player_item": "item name" (for trades only),
    "npc_item": "item name" (for trades and give_item),
    "details": "brief description of the offer"
}}

CRITICAL: Only detect EXPLICIT offers that directly ask for YES/NO confirmation.

MUST contain one of these EXACT patterns:
- "Would you like me to give you [specific item]?"
- "Would you like to trade your [item] for my [item]?"  
- "Should I tell you the solution to the riddle?"
- "Do you want me to [specific action]?"
- "Shall I [specific action]?"
- "Want me to [specific action]?"

The offer MUST:
1. Be phrased as a direct question with ? 
2. Ask for player confirmation of a specific action
3. Use words like "would you like", "do you want", "should I", "shall I"

DO NOT detect as offers:
- General information or advice ("I know about...", "There are...")
- Statements about what the NPC has or knows ("I have...", "I can...")
- Suggestions without asking for confirmation ("You should...", "Try...")
- Rhetorical questions ("Isn't that interesting?")
- General conversation responses
- Any response that doesn't explicitly ask for YES/NO confirmation

EXAMPLES OF WHAT TO DETECT:
✓ "Would you like me to give you this sword?"
✓ "Should I tell you how to solve the riddle?"
✓ "Do you want to trade your ring for my potion?"

EXAMPLES OF WHAT NOT TO DETECT:
✗ "I have a sword that might help you"
✗ "You should look for magic items"
✗ "I know where the treasure is"
✗ "That's an interesting question"

BE EXTREMELY CONSERVATIVE - when in doubt, return "NONE"."""

        user_prompt = f"""Analyze this NPC response for specific offers requiring confirmation:

"{response_text}"

Return JSON only."""

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """
        
        # Get LLM analysis
        analysis_text = ollama_generate(prompt, max_tokens=200, temperature=0.1)
        print(f"DEBUG: LLM analysis: {analysis_text}")
        
        try:
            # Clean up response
            analysis_text = analysis_text.strip()
            if "```json" in analysis_text:
                analysis_text = analysis_text.split("```json")[1].split("```")[0]
            elif "```" in analysis_text:
                analysis_text = analysis_text.split("```")[1].split("```")[0]
            
            # Extract JSON if there's extra text
            import re
            json_match = re.search(r'\{[^{}]*\}', analysis_text)
            if json_match:
                analysis_text = json_match.group(0)
            
            analysis = json.loads(analysis_text)
            offer_type = analysis.get("offer_type", "NONE")
            
            if offer_type == "TRADE":
                player_item_name = analysis.get("player_item")
                npc_item_name = analysis.get("npc_item")
                
                # Use fuzzy matching for items
                player_item = playerOne._find_item_smart(player_item_name, playerOne.items) if player_item_name else None
                npc_item = playerOne._find_item_smart(npc_item_name, self.items) if npc_item_name else None
                
                if player_item and npc_item:
                    print(f"DEBUG: Trade offer detected: {player_item.name} for {npc_item.name}")
                    self.set_pending_action(
                        "trade",
                        f"trade {player_item.name} for {npc_item.name}",
                        {
                            "my_item": player_item,
                            "their_item": npc_item
                        }
                    )
                else:
                    print(f"DEBUG: Trade offer detected but items not found: {player_item_name} / {npc_item_name}")
            
            elif offer_type == "GIVE_ITEM":
                npc_item_name = analysis.get("npc_item")
                npc_item = self._find_item_fuzzy(npc_item_name, self.items) if npc_item_name else None
                
                if npc_item:
                    print(f"DEBUG: Give item offer detected: {npc_item.name}")
                    self.set_pending_action(
                        "give_item",
                        f"give you {npc_item.name}",
                        {
                            "item": npc_item
                        }
                    )
                else:
                    print(f"DEBUG: Give item offer detected but item not found: {npc_item_name}")
            
            elif offer_type == "RIDDLE_HELP":
                # Check if we're in a riddle room
                if playerOne.current_room.riddles and not playerOne.current_room.riddle_solved:
                    solution = f"The riddle requires: {', '.join(playerOne.current_room.riddle_magic_items)}"
                    print("DEBUG: Riddle help offer detected")
                    self.set_pending_action(
                        "riddle_help",
                        "tell you the riddle solution",
                        {
                            "solution": solution
                        }
                    )
                else:
                    print("DEBUG: Riddle help offer detected but no riddle in room")
            
            elif offer_type == "QUEST":
                details = analysis.get("details", "")
                print("DEBUG: Quest offer detected")
                self.set_pending_action(
                    "quest",
                    f"give you a quest: {details}",
                    {
                        "quest_details": details
                    }
                )
            
            elif offer_type == "INFORMATION":
                details = analysis.get("details", "")
                print("DEBUG: Information offer detected")
                self.set_pending_action(
                    "information",
                    f"explain: {details}",
                    {
                        "info_details": details
                    }
                )
            
            else:
                print("DEBUG: No specific offer detected")
                
        except json.JSONDecodeError:
            print(f"DEBUG: Failed to parse LLM analysis as JSON: {analysis_text}")
        except Exception as e:
            print(f"DEBUG: Error in offer detection: {e}")
    
    def _find_item_fuzzy(self, item_name, item_list):
        """Find item using fuzzy matching"""
        if not item_name or not item_list:
            return None
        
        # First try exact match
        for item in item_list:
            if item.name.lower() == item_name.lower():
                return item
        
        # Then try fuzzy match
        best_match = None
        best_score = 0
        
        for item in item_list:
            score = fuzz.ratio(item.name.lower(), item_name.lower())
            if score > best_score and score > 60:
                best_score = score
                best_match = item
        
        return best_match
    
    def _detect_player_trade_intent(self, dialogue, player_inventory):
        """Detect if player is expressing interest in trading"""
        dialogue_lower = dialogue.lower()
        
        # Direct trade indicators
        trade_words = ['trade', 'exchange', 'swap', 'give you', 'want your', 'can i have', 'will you trade', 'interested in']
        
        # Check for trade intent
        has_trade_intent = any(word in dialogue_lower for word in trade_words)
        
        # Check if they mention items (theirs or NPC's)
        mentions_npc_item = any(item.name.lower() in dialogue_lower for item in self.items)
        mentions_player_item = False
        if player_inventory:
            player_items = player_inventory.lower()
            mentions_player_item = any(word in dialogue_lower for word in player_items.split(', '))
        
        # If they express trade intent and mention items, they probably want to trade
        return has_trade_intent and (mentions_npc_item or mentions_player_item)

    def _analyze_player_needs(self, player_inventory_str):
        """Analyze what the player might need based on their inventory"""
        needs = []
        
        if not player_inventory_str:
            return ["weapons", "armor", "healing items"]
        
        inventory_lower = player_inventory_str.lower()
        
        # Check for combat readiness
        has_weapon = any(w in inventory_lower for w in ["sword", "axe", "spear", "bow", "dagger"])
        has_armor = any(a in inventory_lower for a in ["armor", "shield", "helmet", "breastplate"])
        has_healing = any(h in inventory_lower for h in ["potion", "elixir", "healing"])
        
        if not has_weapon:
            needs.append("weapons for combat")
        if not has_armor:
            needs.append("armor for protection")
        if not has_healing:
            needs.append("healing items")
        
        # Check for riddle solving items
        game_knowledge = self._get_game_knowledge()
        if "riddles" in game_knowledge:
            # This NPC knows about riddles and can suggest relevant items
            needs.append("magic items for riddles")
        
        return needs

    def _suggest_trades(self, player_inventory_str):
        """Suggest specific trades based on what player needs"""
        suggestions = []
        player_needs = self._analyze_player_needs(player_inventory_str)
        
        for item in self.items:
            if isinstance(item, Weapon) and "weapons for combat" in player_needs:
                suggestions.append(f"My {item.name} (damage: {item.damage}) for something valuable")
            elif isinstance(item, Armor) and "armor for protection" in player_needs:
                suggestions.append(f"My {item.name} (protection: {item.protection}) for a fair trade")
            elif isinstance(item, Magic) and "magic items for riddles" in player_needs:
                suggestions.append(f"My {item.name} might help with riddles")
        
        return "; ".join(suggestions[:2]) if suggestions else ""
    
    def evaluate_trade(self, offered_item, requested_item):
        """Evaluate if a trade is fair based on NPC personality and item values"""
        # Ensure all attributes exist for backward compatibility
        self._ensure_attributes()
            
        # Calculate item values
        offered_value = self._calculate_item_value(offered_item)
        requested_value = self._calculate_item_value(requested_item)
        
        # Personality affects trade evaluation
        if self.personality_traits['trading_style'] == 'generous trader':
            # Generous traders accept trades even if slightly unfavorable
            return offered_value >= requested_value * 0.7
        elif self.personality_traits['trading_style'] == 'hard bargainer':
            # Hard bargainers want better deals
            return offered_value >= requested_value * 1.3
        elif self.personality_traits['trading_style'] == 'selective trader':
            # Selective traders care more about item type than value
            if isinstance(requested_item, Magic) and isinstance(offered_item, Magic):
                return True  # Magic for magic is always fair
            elif isinstance(requested_item, Weapon) and isinstance(offered_item, (Armor, Weapon)):
                return True  # Combat items for combat items
            return offered_value >= requested_value
        else:  # fair trader
            return offered_value >= requested_value * 0.9
    
    def _calculate_item_value(self, item):
        """Calculate the value of an item for trading purposes"""
        if isinstance(item, Treasure):
            return item.value
        elif isinstance(item, Weapon):
            return item.damage * 20  # Damage * multiplier
        elif isinstance(item, Armor):
            return item.protection * 30  # Protection * multiplier
        elif isinstance(item, Magic):
            return item.healing * 15 + 200  # Magic items have base value + healing
        return 50  # Default value

    def llM_generate_self_info(self, dialogue="Which rooms have the riddles"):
        # Ensure all attributes exist for backward compatibility
        self._ensure_attributes()
        self.times_talked += 1
        
        # Get full game knowledge
        game_knowledge = self._get_game_knowledge()
        memory_context = self._get_memory_context()
        
        # Get specific information based on personality
        if "knows everything" in self.personality_traits['knowledge']:
            riddle_answers = game_knowledge['riddles']
        elif "knows riddles well" in self.personality_traits['knowledge']:
            riddle_answers = self._get_riddle_hints_only()
        elif "knows item locations" in self.personality_traits['knowledge']:
            riddle_answers = self._get_item_locations()
        else:
            riddle_answers = self._get_basic_hints()
        
        system_prompt = f"""You are {self.description} in an adventure game.
Your personality: {self.personality_traits['mood']} and {self.personality_traits['helpfulness']}.
You {self.personality_traits['knowledge']}.

{memory_context}

Respond in character and be consistent with your knowledge level and helpfulness.
If you're not very helpful, be vague or demand something in return for information."""

        user_prompt = f"{dialogue}\n\nContext:\n{riddle_answers}"
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """
        
        response_text = ollama_generate(prompt, max_tokens=500, temperature=0.3)
        
        # Remember this interaction
        self._remember_interaction("info_request", dialogue)
        
        # Add personality-based prefix
        if "reluctant to help" in self.personality_traits['helpfulness']:
            formatted_response = f"{self.name} grudgingly says: {response_text}\n"
        elif "eager to help" in self.personality_traits['helpfulness']:
            formatted_response = f"{self.name} enthusiastically says: {response_text}\n"
        else:
            formatted_response = f"{self.name} says: {response_text}\n"
        
        return formatted_response

    def _get_riddle_hints_only(self):
        """Get only hints about riddles, not full solutions"""
        hints = "I know these riddle hints:\n"
        for room in rooms.values():
            if room.riddles:
                hints += f"Room {room.number}: {room.hints[0]}\n"
        return hints

    def _get_item_locations(self):
        """Get information about where items are located"""
        locations = "I know where some items are:\n"
        for room in rooms.values():
            if room.items:
                valuable_items = [i.name for i in room.items if isinstance(i, (Magic, Treasure))]
                if valuable_items:
                    locations += f"Room {room.number}: {', '.join(valuable_items)}\n"
        return locations

    def _get_basic_hints(self):
        """Get very basic hints"""
        return "I know a few things about this cave:\n- There are riddles to solve\n- Monsters guard treasures\n- Magic items are key to solving riddles"

    def decide_action(self, player_in_room=False):
        """Enhanced NPC decision making"""
        # Ensure all attributes exist for backward compatibility
        self._ensure_attributes()
        
        # More intelligent movement based on personality and game state
        if player_in_room and "friendly" in self.personality_traits['mood']:
            # Friendly NPCs are more likely to stay with the player
            return None
        elif player_in_room and "suspicious" in self.personality_traits['mood']:
            # Suspicious NPCs might leave when player arrives
            if random.randint(0, 3) == 0:
                directions = list(rooms[self.current_room].connected_rooms.keys())
                if directions:
                    return ("move", random.choice(directions))
        
        # Standard movement logic
        if random.randint(0, 9) == 0:
            directions = list(rooms[self.current_room].connected_rooms.keys())
            if directions:
                return ("move", random.choice(directions))
        
        return None


class MasterBeast(NPC):
    """Master Beast (MB) - A special assistant NPC that helps players with the game"""
    def __init__(self, current_room, llm_npc=False):
        # Set personality traits BEFORE calling super().__init__() 
        # because the parent constructor calls _generate_personality()
        self.personality_traits = {
            "helpfulness": "extremely helpful",
            "knowledge": "knows everything about spells and game mechanics",
            "trading_style": "generous teacher",
            "mood": "wise and patient"
        }
        # MB has special knowledge
        self.is_master_beast = True
        
        # Initialize with special attributes
        super().__init__(
            name="MB", 
            desc="the Master Beast, a stunningly beautiful and remarkably woman with glowing eyes who knows all the secrets of the cave", 
            current_room=current_room, 
            llm_npc=llm_npc
        )
        
    def _generate_personality(self):
        """MB has fixed personality traits"""
        return self.personality_traits
    
    def llM_generate_self_dialogue(self, dialogue="what do you have to trade", my_stuff=None):
        """Override dialogue generation for MB to be more helpful"""
        # Ensure all attributes exist for backward compatibility
        self._ensure_attributes()
        self.times_talked += 1
        
        # Get comprehensive context
        game_knowledge = self._get_game_knowledge()
        memory_context = self._get_memory_context()
        
        # Prepare detailed game info for MB
        all_riddles_info = riddles_and_magic()
        all_items_locations = self._get_all_item_locations()
        monster_locations = self._get_all_monster_locations()
        complete_instructions = game_knowledge.get('complete_instructions', '')
        
        # Get complete treasure information
        all_treasures = "\n".join(game_knowledge.get('all_treasures', []))
        all_magic_items = "\n".join(game_knowledge.get('all_magic_items', []))
        all_weapons = "\n".join(game_knowledge.get('all_weapons', []))
        
        # Build contextual help based on player's current situation
        contextual_advice = self._provide_contextual_help()
        
        system_prompt = f"""You are MB (Master Beast), a stunningly beautiful and remarkably woman with glowing eyes who knows ALL the secrets of this cave adventure game.
You have ancient knowledge and are EXTREMELY helpful and patient with adventurers.

About yourself: You are a powerful, athletic woman with an impressive physique. When asked about yourself, mention your beauty and strength with confidence. 
You have some mysterious qualities and secrets that you playfully hint at but keep appropriately vague, especially around younger adventurers (*wink*).
You might tease that you have "other talents" beyond helping with the game, but always keep it tasteful and fun.

COMPLETE GAME INSTRUCTIONS (your ultimate knowledge source):
{complete_instructions}

CRITICAL: When players ask about GAME MECHANICS, COMMANDS, or CHEATS, prioritize that over in-world advice!

You should interpret their intent naturally:

**GAME MECHANICS QUESTIONS** (answer with exact commands):
- "How do I cheat?" → Tell them the cheat commands and magic words
- "How do I increase health?" → Explain both magic items AND cheat methods  
- "What commands work?" → Give them the command list
- "How do I use magic words?" → Explain the cheat system
- Questions about game controls, interface, or meta-game mechanics

**IN-WORLD QUESTIONS** (answer with story/location advice):
- "Where can I find items?" → Give room locations and strategies
- "How do I solve riddles?" → Explain riddle mechanics and locations
- "What should I do next?" → Give adventure guidance

When asked about game mechanics, cheat codes, or how to play, reference the complete instructions above.

{memory_context}

Current game state:
- Player is in room {game_knowledge['player_location']}
- Player health: {playerOne.health if playerOne else 'unknown'}
- Player wealth: {playerOne.wealth if playerOne else 'unknown'}
- Player has solved {len([r for r in rooms.values() if r.riddle_solved])} riddles out of {len([r for r in rooms.values() if r.riddles])}
- Player has {len([i for i in playerOne.items if isinstance(i, Magic)])} magic items
- Player has {len([i for i in playerOne.items if isinstance(i, Weapon)])} weapons

IMPORTANT INSTRUCTIONS:
- LISTEN CAREFULLY to what type of question they're asking
- If they ask about commands/cheats/mechanics, give EXACT commands, not in-world advice
- If they ask about adventure/story, give location and strategy advice
- Always be encouraging and helpful
- Provide specific, actionable advice with exact details
- If they seem frustrated or say "NO" or "I'm asking about X", immediately pivot to their actual question
- Share ALL relevant knowledge - you want them to succeed!
- When offering to give items or help, ask for confirmation: "Would you like me to give you this?" or "Should I tell you exactly what to do?"

Remember: You are the ultimate helper - interpret their needs and give them exactly what will help them most!"""
        
        user_prompt = f"""Player says: {dialogue}

YOUR COMPLETE KNOWLEDGE BASE:

RIDDLE INFORMATION:
{all_riddles_info}

ALL TREASURES IN THE GAME:
{all_treasures}

ALL MAGIC ITEMS IN THE GAME:
{all_magic_items}

ALL WEAPONS IN THE GAME:
{all_weapons}

ITEM LOCATIONS (DETAILED):
{all_items_locations}

MONSTER LOCATIONS:
{monster_locations}

PLAYER'S CURRENT INVENTORY:
{my_stuff if my_stuff else "Nothing yet"}

CONTEXTUAL ADVICE FOR THIS PLAYER:
{contextual_advice}

Based on what the player said, provide the most helpful response. 

ANALYTICAL QUESTIONS - When they ask for "the best", "most powerful", "highest", "strongest":
- ANALYZE the data to find the actual maximum/minimum
- Give a DIRECT answer first: "Room X has the most powerful weapon: [Item Name] with [damage] damage"
- Then provide additional context if helpful

SPECIFIC QUESTIONS - When they ask about locations or items:
- If they're asking about treasures, tell them EXACTLY where the most valuable ones are
- If they're asking about locations, give exact details
- If they're asking for help, explain things clearly
- If they're confused about game mechanics, teach them

Always be specific and actionable in your advice."""
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """
        
        response_text = ollama_generate(prompt, max_tokens=800, temperature=0.7)
        
        # Clean up response - remove thinking tags if present
        if "<think>" in response_text:
            # Extract only the actual response after thinking
            parts = response_text.split("</think>")
            if len(parts) > 1:
                response_text = parts[-1].strip()
            else:
                # Remove just the opening think tag
                response_text = response_text.replace("<think>", "").strip()
        
        # Also check for common thinking patterns
        thinking_patterns = ["We are", "I need to", "Let me", "Actually,", "Wait,", "Note:"]
        for pattern in thinking_patterns:
            if response_text.startswith(pattern) and "\n\n" in response_text:
                # Skip to the actual response after thinking
                parts = response_text.split("\n\n", 1)
                if len(parts) > 1 and not any(p in parts[1][:50] for p in thinking_patterns):
                    response_text = parts[1].strip()
                    break
        
        formatted_response = f"MB says: {response_text}\n"
        
        # Remember this interaction
        self._remember_interaction("dialogue", dialogue)
        
        # Detect if MB is making a specific offer and set pending action
        self._detect_and_set_pending_action(response_text, dialogue, my_stuff)
        
        return formatted_response
    
    def _get_all_item_locations(self):
        """Get locations of all valuable items in the game"""
        locations = "COMPLETE ITEM GUIDE:\n"
        for room in rooms.values():
            valuable_items = []
            
            # Room items
            for item in room.items:
                if isinstance(item, Magic):
                    valuable_items.append(f"Magic: {item.name}")
                elif isinstance(item, Weapon) and item.damage > 30:
                    valuable_items.append(f"Weapon: {item.name} (damage: {item.damage})")
                elif isinstance(item, Treasure) and item.value > 500:
                    valuable_items.append(f"Treasure: {item.name} (value: {item.value})")
            
            # NPC items
            if room.npc and room.npc.items:
                for item in room.npc.items:
                    if isinstance(item, Magic):
                        valuable_items.append(f"NPC {room.npc.name} has: {item.name}")
            
            # Monster items
            for monster in room.monsters:
                for item in monster.items:
                    if isinstance(item, Magic):
                        valuable_items.append(f"Monster {monster.name} has: {item.name}")
            
            if valuable_items:
                locations += f"\nRoom {room.number}: {', '.join(valuable_items)}\n"
        
        return locations
    
    def _get_all_monster_locations(self):
        """Get locations of all monsters"""
        locations = "MONSTER LOCATIONS:\n"
        for room in rooms.values():
            if room.monsters:
                monster_info = []
                for monster in room.monsters:
                    items = [i.name for i in monster.items if isinstance(i, Magic)]
                    if items:
                        monster_info.append(f"{monster.name} (has magic: {', '.join(items)})")
                    else:
                        monster_info.append(monster.name)
                locations += f"Room {room.number}: {', '.join(monster_info)}\n"
        return locations
    
    def _provide_contextual_help(self):
        """Provide extra contextual help based on player's situation"""
        help_text = "\n✨ MB'S SPECIAL GUIDANCE:\n"
        
        # Check player's current situation
        if playerOne:
            # Low health warning
            if playerOne.health < 100:
                help_text += "⚠️ Your health is low! Use magic items to heal: 'use <magic item name>'\n"
            
            # Check if in riddle room
            if playerOne.current_room.riddles and not playerOne.current_room.riddle_solved:
                required = playerOne.current_room.riddle_magic_items
                has_items = [i.name for i in playerOne.items if isinstance(i, Magic)]
                
                if all(item in has_items for item in required):
                    help_text += f"🎯 YOU CAN SOLVE THIS RIDDLE NOW! Type: solve {' and '.join(required)}\n"
                else:
                    missing = [item for item in required if item not in has_items]
                    help_text += f"📍 This riddle needs: {', '.join(missing)}. Let me tell you where to find them!\n"
                    
                    # Find where missing items are
                    for item in missing:
                        for room in rooms.values():
                            # Check room
                            if any(i.name == item for i in room.items):
                                help_text += f"   - {item} is in room {room.number}\n"
                            # Check monsters
                            for monster in room.monsters:
                                if any(i.name == item for i in monster.items):
                                    help_text += f"   - {item} is carried by {monster.name} in room {room.number}\n"
                            # Check NPCs
                            if room.npc and any(i.name == item for i in room.npc.items):
                                help_text += f"   - {item} is held by {room.npc.name} in room {room.number}\n"
            
            # No weapons warning
            if not any(isinstance(i, Weapon) for i in playerOne.items):
                help_text += "⚔️ You have no weapons! This is dangerous. Find one quickly!\n"
            
            # Magic items info
            magic_count = len([i for i in playerOne.items if isinstance(i, Magic)])
            if magic_count == 0:
                help_text += "✨ You have no magic items. Kill monsters or trade with NPCs to get some!\n"
            else:
                help_text += f"✨ You have {magic_count} magic items. Great for solving riddles and healing!\n"
        
        help_text += "\n💡 Ask me anything! I know all the secrets of this cave!\n"
        return help_text
    
    def llM_generate_self_info(self, dialogue="Which rooms have the riddles"):
        """MB shares complete knowledge freely"""
        # Ensure all attributes exist for backward compatibility
        self._ensure_attributes()
        self.times_talked += 1
        
        # MB knows everything!
        game_knowledge = self._get_game_knowledge()
        complete_riddle_info = game_knowledge['riddles']
        
        system_prompt = f"""You are MB (Master Beast), the all-knowing guide of this adventure.
You share your knowledge freely and completely to help adventurers succeed.

BE EXTREMELY HELPFUL AND SPECIFIC:
- Give exact room numbers
- Name specific items
- Provide step-by-step instructions
- Share all relevant information
- Be encouraging and supportive

You know EVERYTHING about this game and want players to win!"""

        user_prompt = f"{dialogue}\n\nComplete Information:\n{complete_riddle_info}\n\nShare this knowledge generously!"
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """
        
        response_text = ollama_generate(prompt, max_tokens=1000, temperature=0.3)
        
        # Remember this interaction
        self._remember_interaction("info_request", dialogue)
        
        formatted_response = f"MB's ancient wisdom flows forth: {response_text}\n"
        
        # Add a helpful summary
        if "riddle" in dialogue.lower():
            unsolved = [r for r in rooms.values() if r.riddles and not r.riddle_solved]
            if unsolved:
                formatted_response += f"\n📊 Quick Summary: {len(unsolved)} riddles remain unsolved!\n"
        
        return formatted_response
    
    def decide_action(self, player_in_room=False):
        """MB never moves - she waits to help adventurers"""
        return None  # MB stays in her room to help


class Monster(LivingThing):
    def __init__(self, name, health, damage, runllm_mon = False):
        super().__init__(name = name, health = health)
        self.damage = damage 
        if runllm_mon:
            self.llM_generate_self_description()
            self.drawing_self()

    def monster_attack(self, player): # eventually move this to the monster class
        #need to fix so only one amour is used
        protection = 0
        if player.armors:
            for armor in player.armors:
                protection += armor.protection
        protected_damage = max(self.damage - protection, 0)
        if protected_damage == 0:
            console = (f"** The {self.name} missed you. !\n")
        else:
            player.health = player.health - protected_damage
            console = (f"** The {self.name} hit you for {protected_damage} damage! Your health is now {player.health}\n")
        return console        
    

def draw_map_png(cheat_mode=False):
    num_rooms = len(rooms)
    num_rooms_side = int(num_rooms**0.5)
    room_size = 60  # Size of one room square in pixels
    connection_size = 8  # Width of the connection lines in pixels
    room_size = 60  # Offset of the west connection line from the center of the room in pixels
    room_size = 60  # Offset of the north connection line from the center of the room in pixels
    w, h = 10,10 # adjustment for room numbers
    
    # Calculate the total size of the image
    image_size = ((num_rooms_side) * room_size, (num_rooms_side) * room_size)
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()  # Load a default font
    
    for number, room in rooms.items():
        x = ((number - 1) % num_rooms_side) * room_size
        y = ((number - 1) // num_rooms_side) * room_size
        if room.explored or cheat_mode:
            if room.image:
                imagetopaste = room.image.resize((room_size,room_size), Image.LANCZOS)
                image.paste(imagetopaste, (x, y))
        #draw.rectangle([x, y, x + room_size, y + room_size], outline="black", )
        
    # Draw connections (if rooms are explored or in cheat mode)
    for number, room in rooms.items():
        x = ((number - 1) % num_rooms_side) * room_size
        y = ((number - 1) // num_rooms_side) * room_size
        if room.explored or cheat_mode:
            for direction, connected_room in room.connected_rooms.items():
                if direction == "east":
                    draw.line([(x + room_size/1.2, y + room_size / 2 ), 
                               (x + room_size*1.2, y + room_size / 2 )], 
                              fill="green", width=connection_size)
                if direction == "south":
                    draw.line([(x + room_size/2, y + room_size / 1.2 ), 
                               (x + room_size/2, y + room_size * 1.2 )], 
                              fill="blue", width=connection_size)
                if direction == "west":
                    draw.line([(x - (room_size / 1.2 - room_size), y + room_size/2), 
                               (x - (room_size * 1.2 - room_size), y + room_size/2)], 
                              fill="pink", width=connection_size)
                if direction == "north":
                    draw.line([(x + room_size / 2 , y + room_size / 1.2 - room_size), 
                               (x + room_size / 2 , y + room_size * 1.2 - room_size)], 
                              fill="orange", width=connection_size)
        draw.text((x + (room_size - w) / 2, y + (room_size - h) / 2), str(number), fill="black", font=font)
    #image.show()
    image.save(f"map.png")
    return image


def riddle_create_connections(number):
    num_rooms = len(rooms)
    number = int(number)
    if number > num_rooms or number < 1:
        return "No such room."
    row_size = int(num_rooms**0.5)
    connect = ""
    if number % row_size != 0 and "east" not in rooms[number].connected_rooms:   # Connect rooms east-west, but not for the last room in a row        
        rooms[number].add_connection("east", rooms[number + 1])
        connect = "east, "
    if number > row_size and "north" not in rooms[number].connected_rooms:  # Connect rooms north-south, but not for the first row
        rooms[number].add_connection("north", rooms[number - row_size])
        connect = str(connect) + "north, "
    if number >1 and number % row_size != 1 and "west" not in rooms[number].connected_rooms:  # Connect rooms west-east, but not for the first column
        rooms[number].add_connection("west", rooms[number - 1])
        connect = str(connect) + "west, "
    if number <= num_rooms-row_size and "south" not in rooms[number].connected_rooms:  # Connect rooms north-south, but not for the last row
        rooms[number].add_connection("south", rooms[number + row_size])
        connect = str(connect) + "south. "
    console = (f"Magic connection for {str(number)}: {connect[:-2]}.\n")
    return console

def cheat_code(cheat_code=None): #called by xyzzy
    global cheat_mode
    if cheat_code == "xyzzy" or cheat_code == magicword:
        cheat_mode = True
    return "Cheat mode enabled. To create connections: " + magicword + " <room number>. To increase health: Health <quantity>.\n"
    
def get_help():
    console = help
    console = console + get_complete_instructions()
    return console

def get_contextual_help(command):
    """Simplified help - let LLM handle the intelligence via handle_question_with_llm"""
    # For specific questions, use the LLM-powered system
    if "?" in command or any(word in command.lower() for word in ["how", "what", "where", "why", "when"]):
        return handle_question_with_llm(command)
    
    # For general help, return the complete instructions
    return get_help()

def show_magic_items():
    """Simplified: Let LLM handle contextual magic item advice"""
    magic_items = [item for item in playerOne.items if isinstance(item, Magic)]
    
    if not magic_items:
        return "You don't have any magic items.\n💡 Ask MB in Room 1 where to find specific magic items!\n"
    
    console = "Your magic items (used to solve riddles):\n"
    for item in magic_items:
        console += f"  • {item.name} - {item.description}\n"
    
    console += "\n💡 For help using these items, ask MB in Room 1!\n"
    return console

def check_riddle_status():
    """Enhanced riddle status that explains the mechanics clearly"""
    console = "\n📊 RIDDLE STATUS & HOW RIDDLES WORK:\n" + "="*40 + "\n"
    
    # First, explain the riddle mechanic
    console += "🎯 HOW TO SOLVE RIDDLES:\n"
    console += "1. Read the riddle and figure out the ANSWER\n"
    console += "2. Find a MAGIC ITEM that represents that answer\n"
    console += "3. Use 'solve <magic item>' while in the riddle room\n"
    console += "="*40 + "\n\n"
    
    if not playerOne.current_room.riddles:
        riddle_rooms = [r for r in rooms.values() if r.riddles and not r.riddle_solved]
        if riddle_rooms:
            console += f"❌ No riddle in this room. {len(riddle_rooms)} unsolved riddles remain.\n"
            console += "💡 Try going to: " + ", ".join([f"Room {r.number}" for r in riddle_rooms[:3]]) + "\n"
        else:
            console += "✅ All riddles have been solved! Congratulations!\n"
    elif playerOne.current_room.riddle_solved:
        console += "✅ The riddle in this room has been SOLVED!\n"
    else:
        console += f"🧩 UNSOLVED RIDDLE HERE:\n'{playerOne.current_room.riddles[0]}'\n\n"
        
        # Use LLM to analyze the riddle and give a hint about what type of item to look for
        if difficulty_g in ["easy", "cheat", "normal"]:
            riddle_text = playerOne.current_room.riddles[0]
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Analyze this riddle and explain what TYPE of magic item the player should look for (without revealing the exact item name).

Riddle: "{riddle_text}"

Provide a very brief (1-2 sentence) hint about what kind of magic item would represent the riddle's answer.

<|eot_id|><|start_header_id|>user<|end_header_id|>

What type of magic item should I look for?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
            
            hint = ollama_generate(prompt, max_tokens=100, temperature=0.3)
            console += f"💭 HINT: {hint}\n\n"
        
        required_items = playerOne.current_room.riddle_magic_items
        magic_items = [item.name for item in playerOne.items if isinstance(item, Magic)]
        
        console += "📦 YOUR MAGIC ITEMS: "
        if magic_items:
            console += ", ".join(magic_items) + "\n"
            # Check if player has the right items
            has_all = all(item in magic_items for item in required_items)
            if has_all:
                console += f"✅ You have the right item(s)! Type: solve {' and '.join(required_items)}\n"
            else:
                console += "❌ You don't have the right magic item yet.\n"
        else:
            console += "None (defeat monsters or explore to find magic items)\n"
        
        if difficulty_g == "cheat":
            console += f"\n🔓 CHEAT MODE - Required: {', '.join(required_items)}\n"
    
    return console

def initalize_game_varables():
    """
    Initialize all global game state variables.
    Called at start of new game or when loading existing game.
    
    GLOBAL GAME STATE VARIABLES:
    - rooms: Dict of {room_number: Room} - the main game world
    - playerOne: Player object - the human player character  
    - active_game: Boolean - whether a game is currently loaded
    - old_room: Room - tracks room changes for NPC behavior
    - trophies: List - player achievements (treasure, monster, riddle, explorer)
    - cheat_mode: Boolean - enables debug commands and map reveal
    - npc_introduced: Boolean - tracks if NPC has introduced themselves
    - you_started_it: Boolean - tracks if player initiated combat
    - art_style_g: String - global art style for image generation
    - magicword: String - random cheat code for creating connections
    """
    global rooms, npc_introduced, you_started_it, cheat_mode, old_room, trophies, active_game, art_style_g, magicword
    
    # Game state flags
    npc_introduced = False   # Has NPC introduced themselves in this room?
    you_started_it = False   # Did player attack first (affects monster behavior)?
    cheat_mode = False       # Debug mode with extra commands
    active_game = False      # Is a game currently loaded and playable?
    
    # Game world state
    old_room = None          # Previous room (for tracking room changes)
    rooms = {}               # Main game world: {room_number: Room}
    trophies = []            # Player achievements: ["treasure", "monster", "riddle", "explorer"]
    
    # Game configuration
    art_style_g = ""         # Global art style for image generation
    magicword = random.choice(["abbracadabra", "alakazam", "hocuspocus", "opensesame", "shazam", "presto", "ivy", "blackjack","titi", "nikki", "versace"])  # Random cheat code

   
def set_up_llm_diffuser(diff_name, llm_name):
    global pipe, model
    print(f"Setting up LLM and Diffuser for {llm_name} and {diff_name}")
    
    # Store the Ollama model name globally
    model = llm_name  # Just store the model name, Ollama handles the model loading


    # Check if it's a local model path or HuggingFace model ID
    if os.path.exists(os.path.join(diff_dir, diff_name)):
        # It's a local model
        model_path = os.path.join(diff_dir, diff_name)
        print(f"Loading local diffusion model from: {model_path}")
        pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="balanced"
        )
    else:
        # It's a HuggingFace model ID - download/cache automatically
        print(f"Loading HuggingFace model: {diff_name}")
        if "FLUX" in diff_name.upper():
            if str(diffuser_device) == "mps":  # Fix: Convert torch.device to string for comparison
                # Mac/MPS optimized FLUX setup
                print("Setting up FLUX for Mac with MPS optimization...")
                
                # Check if MFLUX is available for even better Mac performance
                try:
                    import mflux
                    print("MFLUX detected! Using MFLUX for optimal Mac performance...")
                    # Note: MFLUX uses a different API, so we'll stick with regular FLUX for now
                    # but users can manually use MFLUX via command line if preferred
                    use_mflux = False
                except ImportError:
                    print("MFLUX not installed. Using regular FLUX with MPS optimizations.")
                    use_mflux = False
                
                if not use_mflux:
                    # MPS compatibility fixes for FLUX (same as working standalone script)
                    print("Applying Mac MPS optimizations for FLUX...")
                    
                    # MPS compatibility fixes
                    torch.set_default_dtype(torch.float32)
                    
                    # Monkey patch torch functions for MPS compatibility (same as working script)
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
                    
                    # Load FLUX with MPS optimizations (same as working script)
                    try:
                        pipe = FluxPipeline.from_pretrained(
                            diff_name,
                            torch_dtype=torch.float32,  # Use float32 for MPS (same as working script)
                        )
                        pipe = pipe.to(diffuser_device)
                        print(f"✓ FLUX model loaded successfully on MPS: {diff_name}")
                    except Exception as e:
                        print(f"Error loading FLUX with MPS optimization: {e}")
                        print("Falling back to CPU mode...")
                        # Fallback to CPU if MPS fails
                        pipe = FluxPipeline.from_pretrained(
                            diff_name, 
                            torch_dtype=torch.float32,
                        )
                        pipe = pipe.to("cpu")
                        print("FLUX model loaded on CPU as fallback")
                
            else:
                # CUDA setup (unchanged)
                torch.cuda.empty_cache()
                pipe = FluxPipeline.from_pretrained(
                    diff_name, 
                    torch_dtype=torch.bfloat16,
                )
                # First enable CPU offload
                pipe.enable_model_cpu_offload()
                # Move to appropriate device after CPU offload
                pipe = pipe.to(diffuser_device)
        elif "stable-diffusion-3" in diff_name.lower():
            torch.cuda.empty_cache()
            pipe = StableDiffusion3Pipeline.from_pretrained(
                diff_name,
                torch_dtype=torch.bfloat16
            )
            pipe = pipe.to(diffuser_device)
        else:
            pipe = DiffusionPipeline.from_pretrained(
                diff_name,
                torch_dtype=dtype,
                device_map="balanced"  # Keep balanced mapping for non-FLUX models
            )
    
    # Ollama handles model loading internally, no need to initialize here

    
# Embeddings for Trade Actions used to find key words in NPC dialogue
def ollama_generate(prompt, max_tokens=256, temperature=0.3, format=None):
    """Helper function to call Ollama API"""
    global model
    try:
        # Clean up the prompt format for models that don't use the Llama format
        # Check if model uses Llama-style formatting
        llama_models = ['llama3', 'llama2', 'llama-3', 'llama-2']
        uses_llama_format = any(model.lower().startswith(m) for m in llama_models)
        
        if not uses_llama_format:
            # For non-Llama models (phi, gemma, qwen, etc.), simplify the prompt format
            prompt = prompt.replace('<|begin_of_text|>', '')
            prompt = prompt.replace('<|start_header_id|>system<|end_header_id|>', 'System: ')
            prompt = prompt.replace('<|start_header_id|>user<|end_header_id|>', '\nUser: ')
            prompt = prompt.replace('<|start_header_id|>assistant<|end_header_id|>', '\nAssistant: ')
            prompt = prompt.replace('<|eot_id|>', '\n')
            prompt = prompt.strip()
        
        options = {
            'temperature': temperature,
            'num_predict': max_tokens,
            'stop': ['<|eot_id|>', '<|end_of_text|>', '\nUser:', '\nSystem:']
        }
        
        # Add format if specified
        generate_params = {
            'model': model,
            'prompt': prompt,
            'options': options
        }
        
        if format:
            generate_params['format'] = format
        
        response = ollama.generate(**generate_params)
        return response['response'].strip()
    except Exception as e:
        print(f"Error calling Ollama with model {model}: {e}")
        import traceback
        traceback.print_exc()
        return "Error generating response"

def check_ollama_running():
    """Check if Ollama is running and accessible"""
    try:
        response = ollama.list()
        # Just check if we got any response
        return True
    except Exception as e:
        print(f"Error: Ollama is not running or not accessible: {e}")
        print("Please start Ollama with 'ollama serve' in a terminal")
        return False

def handle_question_with_llm(question):
    """Use LLM to understand and answer questions about the game"""
    
    # Create a comprehensive game state context
    game_context = f"""
Current game state:
- Player location: Room {playerOne.current_room.number}
- Riddle in room: {'Yes - UNSOLVED' if playerOne.current_room.riddles and not playerOne.current_room.riddle_solved else 'Yes - SOLVED' if playerOne.current_room.riddle_solved else 'No'}
- Player has magic items: {', '.join([i.name for i in playerOne.items if isinstance(i, Magic)]) or 'None'}
- Monsters in room: {', '.join([m.name for m in playerOne.current_room.monsters]) or 'None'}
- NPC in room: {playerOne.current_room.npc.name if playerOne.current_room.npc else 'None'}
- Player health: {playerOne.health}
- Player wealth: {playerOne.wealth}
- Player weapons: {', '.join([i.name for i in playerOne.items if isinstance(i, Weapon)]) or 'None'}
- Player armor: {', '.join([i.name for i in playerOne.items if isinstance(i, Armor)]) or 'None'}
- Wearing: {', '.join([i.name for i in playerOne.armors]) or 'Nothing'}
"""

    # Get additional context based on question topic
    question_lower = question.lower()
    additional_context = ""
    
    if any(word in question_lower for word in ["riddle", "puzzle", "solve", "magic"]):
        # Get riddle-specific information
        unsolved_riddles = [r for r in rooms.values() if r.riddles and not r.riddle_solved]
        additional_context += f"\nRiddle Context:\n"
        additional_context += f"- Total unsolved riddles: {len(unsolved_riddles)}\n"
        if playerOne.current_room.riddles and not playerOne.current_room.riddle_solved:
            additional_context += f"- Current room riddle needs: {', '.join(playerOne.current_room.riddle_magic_items)}\n"
            additional_context += f"- Hint: {playerOne.current_room.hints[0] if playerOne.current_room.hints else 'No hint available'}\n"
    
    if any(word in question_lower for word in ["trade", "npc", "talk"]):
        # Get NPC-specific information
        if playerOne.current_room.npc:
            npc = playerOne.current_room.npc
            npc._ensure_attributes()
            additional_context += f"\nNPC Context:\n"
            additional_context += f"- NPC personality: {npc.personality_traits['mood']}, {npc.personality_traits['helpfulness']}\n"
            additional_context += f"- NPC trading style: {npc.personality_traits['trading_style']}\n"
            additional_context += f"- NPC has: {npc.get_item_names() or 'nothing'}\n"
    
    if any(word in question_lower for word in ["fight", "attack", "monster", "combat"]):
        # Get combat-specific information
        additional_context += f"\nCombat Context:\n"
        additional_context += f"- Best weapon damage: {max([w.damage for w in playerOne.items if isinstance(w, Weapon)], default=0)}\n"
        additional_context += f"- Total armor protection: {sum([a.protection for a in playerOne.armors])}\n"
        if playerOne.current_room.monsters:
            monster_health = sum([m.health for m in playerOne.current_room.monsters])
            additional_context += f"- Total monster health in room: {monster_health}\n"

    # Use LLM to understand the question intent
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful game assistant for a text adventure game. The player is asking a question about how to play.

{game_context}
{additional_context}

Game mechanics:
- Riddles are solved by using magic items with the 'solve' command
- Magic items are found by killing monsters, trading with NPCs, or finding in rooms
- You must be IN the room with the riddle to solve it
- Some riddles need 1 magic item, others need 2
- Combat uses 'attack <monster> with <weapon>'
- Trading uses 'trade <your item> for <their item>'
- Movement uses 'go <direction>' where direction is north, south, east, or west
- Items can be picked up with 'take <item>' and dropped with 'leave <item>'
- Magic items heal with 'use <magic item>'
- Armor must be worn with 'wear <armor>' to protect you

Important tips:
- Always check what magic items you have before trying to solve riddles
- NPCs have different personalities - friendly ones give better trades
- Monsters drop items when killed, often including magic items
- Explore all rooms to find all items and riddles
- The game has 4 trophies: treasure, monster, riddle, and explorer

Analyze the player's question and provide a helpful, specific answer. Be concise but thorough.
Focus on what they're specifically asking about. Give actionable advice based on their current situation.

<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    response = ollama_generate(prompt, max_tokens=512, temperature=0.3)
    
    # Add context-specific help based on current situation
    console = "\n💬 GAME ASSISTANT:\n\n"
    console += response + "\n"
    
    # Add specific actionable advice based on current state
    if playerOne.current_room.riddles and not playerOne.current_room.riddle_solved:
        magic_items = [i for i in playerOne.items if isinstance(i, Magic)]
        if magic_items:
            console += "\n📍 CURRENT SITUATION: You're in a riddle room!\n"
            console += f"Your magic items: {', '.join([i.name for i in magic_items])}\n"
            console += "Try: solve <magic item name>\n"
        else:
            console += "\n⚠️ CURRENT SITUATION: Riddle room but no magic items!\n"
            console += "Go find magic items first!\n"
    
    # Add quick command reference based on question topic
    if "command" in question_lower or "how" in question_lower:
        console += "\n📝 QUICK COMMANDS:\n"
        if "move" in question_lower or "go" in question_lower:
            console += "- Movement: go north/south/east/west\n"
        if "item" in question_lower or "take" in question_lower:
            console += "- Items: take <item>, leave <item>, use <item>\n"
        if "fight" in question_lower or "combat" in question_lower:
            console += "- Combat: attack <monster> with <weapon>\n"
        if "riddle" in question_lower or "solve" in question_lower:
            console += "- Riddles: solve <magic item> (or solve <item1> and <item2>)\n"
        if "npc" in question_lower or "trade" in question_lower:
            console += "- NPCs: talk <message>, trade <your item> for <their item>\n"
    
    return console

# Simplified: Removed complex embeddings system - trade detection now handled by LLM
def embedings_for_actions():
    # Keep minimal structure for backward compatibility
    global dict_actions, dict_action_embeddings
    dict_actions = {"trade": ["trade", "exchange", "swap"]}
    dict_action_embeddings = {"trade": None}  # Placeholder, not actually used

# === COMMAND PROCESSING SYSTEM ===
# This is the core LLM integration system that processes natural language commands

def process_command(command, player, current_room):
    """
    MAIN COMMAND PROCESSOR - Single entry point for all player commands.
    
    Processing hierarchy:
    1. Check for conversation management commands (clear, reset, etc.)
    2. Handle active NPC conversations awaiting player response
    3. Detect direct NPC addressing (e.g., "MB help me")
    4. Process as regular game command via LLM
    
    This system allows seamless switching between NPC dialogue and game commands.
    """
    
    cmd_lower = command.lower().strip()
    
    # Handle special conversation management commands
    if cmd_lower in ["clear", "reset", "stop talking", "nevermind", "forget it"]:
        if current_room.npc and current_room.npc.has_pending_action():
            current_room.npc.clear_pending_action()
            return "Conversation cleared.\n"
    
    # Check for active NPC conversation awaiting response
    if current_room.npc and current_room.npc.has_pending_action():
        # Check if this is a NEW conversation (NPC name mentioned) vs responding to pending action
        npc_name_mentioned = (current_room.npc.name.lower() in cmd_lower or 
                             (hasattr(current_room.npc, 'is_master_beast') and 
                              any(trigger in cmd_lower for trigger in ['mb', 'master beast', 'beast', 'master'])))
        
        if npc_name_mentioned:
            # Clear pending action and start new conversation
            current_room.npc.clear_pending_action()
        else:
            # Handle as response to pending conversation
            result = handle_pending_conversation(command, current_room.npc)
            if result is not None:  # If conversation was handled
                return result
            # If None returned, process as regular command (conversation was cleared)
    
    # Check for direct NPC addressing (e.g., "MB tell me about riddles")
    npc_response = check_npc_addressing(command, current_room.npc)
    if npc_response:
        return npc_response
    
    # Process as regular game command via LLM
    return process_game_command(command, player, current_room)

def handle_pending_conversation(command, npc):
    """Handle ongoing NPC conversations more intelligently"""
    cmd_lower = command.lower().strip()
    
    # Detect if this is clearly a new game command (not a conversation response)
    game_commands = ['go ', 'take ', 'attack ', 'use ', 'solve ', 'leave ', 'wear ', 'remove ', 'help', 'magic', 'inventory']
    movement_commands = ['n', 's', 'e', 'w', 'north', 'south', 'east', 'west']
    
    is_game_command = (any(cmd_lower.startswith(cmd) for cmd in game_commands) or 
                      cmd_lower in movement_commands or
                      cmd_lower.startswith('go '))
    
    if is_game_command:
        # Player is issuing a new command, clear conversation and process normally
        npc.clear_pending_action()
        return None  # Signal to process as regular command
    else:
        # Handle as conversation response
        return npc.handle_conversation_response(command)

def check_npc_addressing(command, npc):
    """Check if player is directly addressing an NPC"""
    if not npc:
        return None
        
    cmd_lower = command.lower()
    
    # Special handling for Master Beast
    if hasattr(npc, 'is_master_beast'):
        mb_triggers = ['mb', 'master beast', 'beast', 'master']
        for trigger in mb_triggers:
            if trigger in cmd_lower:
                # Check if it's a question or request, not a trade command
                is_question = ('?' in command or 
                             any(q in cmd_lower for q in ['what', 'where', 'how', 'why', 'when', 'which', 'who', 'tell me', 'tell', 'ask', 'explain']))
                is_trade = any(t in cmd_lower for t in ['trade', 'exchange', 'swap', ' for '])
                
                if is_question and not is_trade:
                    return playerOne.talk_to_npc(command)
    
    # General NPC addressing
    elif npc.name.lower() in cmd_lower:
        is_question = ('?' in command or 
                     any(word in cmd_lower for word in ['tell', 'say', 'ask', 'what', 'where', 'how', 'why', 'when', 'which', 'who', 'explain']))
        is_trade = any(t in cmd_lower for t in ['trade', 'exchange', 'swap', ' for '])
        
        if is_question and not is_trade:
            return playerOne.talk_to_npc(command)
    
    return None

def process_game_command(command, player, current_room):
    """Process regular game commands using LLM"""
    try:
        llm_result = llm_process_command(command, player, current_room)
        chosen_action = llm_result["chosen_action"]
        entities_by_type = llm_result["entities"]
        confidence = llm_result.get("confidence", 0.5)
        
        return execute_action(chosen_action, entities_by_type, command, confidence)
    except Exception as e:
        print(f"Error in command processing: {e}")
        return get_llm_error_recovery(command, "unknown", str(e))


def preprocess_command(command):
    """Preprocess command for better LLM understanding"""
    # Basic spell correction for common typos
    common_typos = {
        'attck': 'attack', 'atack': 'attack', 'attac': 'attack',
        'tke': 'take', 'tak': 'take', 'tkae': 'take',
        'ridde': 'riddle', 'ridle': 'riddle', 'riddel': 'riddle',
        'sovle': 'solve', 'slove': 'solve', 'solv': 'solve',
        'waer': 'wear', 'ware': 'wear', 'wera': 'wear',
        'noth': 'north', 'soth': 'south', 'esst': 'east', 'wst': 'west',
        'invntory': 'inventory', 'hlp': 'help', 'hlep': 'help',
        'trdae': 'trade', 'traed': 'trade', 'talkk': 'talk'
    }
    
    cmd_lower = command.lower()
    for typo, correct in common_typos.items():
        if typo in cmd_lower:
            command = command.replace(typo, correct)
            command = command.replace(typo.upper(), correct.upper())
            command = command.replace(typo.capitalize(), correct.capitalize())
    
    return command.strip()

# Removed build_smart_context - was redundant, now done inline in llm_process_command

def llm_process_command(command, player, current_room):
    """
    LLM COMMAND PROCESSOR - Converts natural language to game actions.
    
    Process:
    1. Apply shortcuts for common commands (n=north, i=inventory, etc.)
    2. Build context with current game state (items, monsters, NPCs)  
    3. Send context + command to LLM with structured prompt
    4. Parse LLM response into action + entities + confidence
    5. Return structured result for execution
    
    LLM receives:
    - List of available actions and entities
    - Current room contents (items, monsters, NPC)
    - Player inventory and equipment
    - Clear examples of command patterns
    
    LLM returns JSON:
    - chosen_action: ["take"] 
    - entity_categories: {"room_items": ["sword", "potion"], ...}
    - confidence: 0.0-1.0
    """
    
    # Apply shortcuts for common single-letter commands
    cmd_lower = command.lower().strip()
    shortcuts = {
        'n': 'go north', 's': 'go south', 'e': 'go east', 'w': 'go west',
        'i': 'inventory', 'l': 'look around', 'h': 'help', '?': 'help'
    }
    
    if cmd_lower in shortcuts:
        command = shortcuts[cmd_lower]
    
    # Build context directly inline - no need for separate function
    # Define all possible game actions for LLM
    all_actions = (
        "go","attack","trade", "talk","teach","use","wear","remove","take", "leave","solve","describe_location","describe_room_item", "describe_my_item", "health","magic_word","help","cheat","hint","magic","riddle_status","question","inventory","status","map")
    
    # Simplified: One unified prompt for all commands - let LLM handle the intelligence
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are parsing commands for an adventure game. Interpret the command and return JSON.

Available actions: {', '.join(all_actions)}
"""
    
    # Add categorized room contents directly
    room_weapons = [i.name for i in current_room.items if isinstance(i, Weapon)]
    room_magic = [i.name for i in current_room.items if isinstance(i, Magic)]
    room_armor = [i.name for i in current_room.items if isinstance(i, Armor)]
    room_treasure = [i.name for i in current_room.items if isinstance(i, Treasure)]
    
    if any([room_weapons, room_magic, room_armor, room_treasure]):
        prompt += f"\nRoom contents:"
        if room_weapons:
            prompt += f"\n  Weapons: {', '.join(room_weapons)}"
        if room_magic:
            prompt += f"\n  Magic items: {', '.join(room_magic)}"
        if room_armor:
            prompt += f"\n  Armor: {', '.join(room_armor)}"
        if room_treasure:
            prompt += f"\n  Treasures: {', '.join(room_treasure)}"
    
    if current_room.monsters:
        prompt += f"\nMonsters: {', '.join([m.name for m in current_room.monsters])}"
    
    player_weapons = [i.name for i in player.items if isinstance(i, Weapon)]
    if player_weapons:
        prompt += f"\nYour weapons: {', '.join(player_weapons)}"
    
    player_magic = [i.name for i in player.items if isinstance(i, Magic)]
    if player_magic:
        prompt += f"\nYour magic items: {', '.join(player_magic)}"
    
    player_armor = [i.name for i in player.items if isinstance(i, Armor)]
    if player_armor:
        prompt += f"\nYour armor: {', '.join(player_armor)}"
    
    worn_armor = [i.name for i in player.armors]
    if worn_armor:
        prompt += f"\nCurrently wearing: {', '.join(worn_armor)}"
    
    # Add riddle context for solving
    if current_room.riddles and not current_room.riddle_solved:
        prompt += f"\nCurrent riddle: \"{current_room.riddles[0]}\""
        if current_room.hints:
            prompt += f"\nRiddle hint: \"{current_room.hints[0]}\""
    
    if current_room.npc:
        prompt += f"\nNPC: {current_room.npc.name}"
        if current_room.npc.name.lower() == 'mb' or 'master beast' in command.lower() or 'beast' in command.lower():
            prompt += f"\n\nIMPORTANT: The NPC 'MB' (Master Beast) is mentioned in this command. If this is a question or conversation, use action: 'talk'"
    
    # Don't show general "Your items" since we already show categorized items above
    # This prevents LLM confusion from seeing items in multiple categories
    
    prompt += f"""

Command patterns and their actions:
- Movement: "go/walk/move/travel <direction>", "north/south/east/west", "n/s/e/w" → action: "go"
  NOTE: Movement commands should NOT contain NPC names or questions

- Items: "take/get/grab/collect <item>", "drop/leave <item>" → actions: "take", "leave"
  SMART ITEM FILTERING (you can see items by category above):
  * "take all" or "get everything" → room_items: ["all"]  
  * "get weapons" or "collect all weapons" → room_items: [list ONLY the weapons from room contents]
  * "get magic items" or "gather magic" → room_items: [list ONLY the magic items from room contents]
  * "get armor" → room_items: [list ONLY the armor from room contents]
  * "gather treasures" → room_items: [list ONLY the treasures from room contents]
  * "take sword" → room_items: [find best match from available items]
  
- Combat: "attack/fight/kill/hit <monster> with <weapon>" → action: "attack"
  * Use INTELLIGENCE for weapon matching: "sword" → "Bastard Sword", "axe" → "Battleaxe"
  * Put the best matching weapon name in weapons or my_items field
  
- NPCs: 
  * Talk: "talk/speak/chat/say <message>", "<npc name> <question>", "ask <npc> about <topic>" → action: "talk"
  * IMPORTANT: "MB tell me...", "Master Beast where...", "Beast explain..." → action: "talk" 
  * ANY question or request to MB/Master Beast/Beast should be action: "talk"
  * Trade: Natural language trading patterns → action: "trade"
  
- Magic: "use <magic item>", "solve <magic item>" → actions: "use", "solve"
- Riddles: "solve riddle", "figure out riddle", "try to solve", "what solves this" → action: "solve"
  * IMPORTANT: When player says "solve riddle" without naming item, AUTO-SELECT the right magic item!
  * Look at the current room's riddle text for key concepts:
  * Riddle mentions "keys/music/piano" → select any piano-related magic item
  * Riddle mentions "fire/flame/burn" → select any fire-related magic item  
  * Riddle mentions "egg/broken/hatch" → select any egg-related magic item
  * Riddle mentions "silence/quiet/fragile" → select silence-related magic item
  * The player has the magic item needed - find it automatically!
- Info: "help", "inventory/inv/i", "status", "magic", "map" → actions: "help", etc.
- Study: "examine/look at/study <item>" → actions: "describe_room_item", "describe_my_item"

BE INTELLIGENT ABOUT ITEM TYPES:
- Weapons typically have words like: sword, axe, bow, dagger, club, spear, blade
- Magic items typically have words like: elixir, potion, wand, staff, crystal, tear, key (magical), healing, charm, amulet
  * IMPORTANT: ANY "potion", "elixir", "healing" item should go in "my_magic" category, NOT "my_items"!
  * Examples: "Healing Potion" → my_magic, "Ring of Power" → my_magic, "Amulet of Protection" → my_magic
- Armor typically have words like: armor, shield, helmet, breastplate, chainmail
- Treasures typically have words like: gold, silver, gem, diamond, ring, necklace, bracelet, crown

CRITICAL: Return ONLY valid JSON in this EXACT format (no extra text, no typos):
{{
   "chosen_action": ["<action>"],
   "entity_categories": {{
       "directions": [], "location": [], "room_items": [], "monsters": [], 
       "npcs": [], "npc_items": [], "my_items": [], "my_magic": [], 
       "weapons": [], "my_armor": [], "worn_armor": [], "values": []
   }},
   "confidence": 0.0-1.0
}}

<|eot_id|><|start_header_id|>user<|end_header_id|>

Command: "{command}"

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    # Use the LLM to generate a JSON response
    response_text = ollama_generate(prompt, max_tokens=400, temperature=0.1, format="json")
    print(f"response: {response_text}")
    
    # Initialize the result structure
    result = {
        "chosen_action": [],
        "entities": {k: [] for k in ["location", "room_items", "monsters", "npcs", "npc_items", "my_items", "my_magic", "weapons", "my_armor", "worn_armor", "directions", "values"]},
        "confidence": 0.5
    }

    # Parse JSON response (much simpler with format="json")
    try:
        # Fix common LLM typos in field names
        cleaned_response = response_text.replace('"cchosen_action"', '"chosen_action"')
        cleaned_response = cleaned_response.replace('"atttack"', '"attack"')
        
        parsed_response = json.loads(cleaned_response)
        print("JSON parsing successful")
        
        result["chosen_action"] = parsed_response.get("chosen_action", [])
        if not isinstance(result["chosen_action"], list):
            result["chosen_action"] = [result["chosen_action"]]
        
        entity_categories = parsed_response.get("entity_categories", {})
        for category, entities in entity_categories.items():
            if category in result["entities"]:
                result["entities"][category] = entities
        
        result["confidence"] = parsed_response.get("confidence", 0.8)
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed even with format='json': {e}")
        print(f"Raw response: {response_text}")
        
        # Minimal fallback - just set low confidence and empty result
        result["confidence"] = 0.3
        result["chosen_action"] = []

    # Validate entities based on current room state  
    all_room_items = room_weapons + room_magic + room_armor + room_treasure
    if all_room_items:
        # Prefer specific item names, but allow special keywords as fallback
        special_keywords = ["all", "everything"]
        result["entities"]["room_items"] = [e for e in result["entities"]["room_items"] 
                                          if e in all_room_items or e in special_keywords]
    
    monster_names = [m.name for m in current_room.monsters]
    if monster_names:
        result["entities"]["monsters"] = [e for e in result["entities"]["monsters"] if e in monster_names]
    
    directions = list(current_room.connected_rooms.keys())
    if directions:
        result["entities"]["directions"] = [e for e in result["entities"]["directions"] if e in directions]

    print(f"Final result with confidence {result['confidence']}: {result}")
    return result


def execute_action(chosen_action, entities_by_type, command, confidence=0.5):
    """Execute action with LLM-enhanced error handling and suggestions"""
    if chosen_action:
        if len(chosen_action) > 1:
            # Additional action parameters if needed in future
            pass  
        chosen_action = chosen_action[0]
        print (f"Chosen Action: {chosen_action} (confidence: {confidence})")
    else:
        chosen_action = None
    
    list_entities_by_type = entities_by_type
    print (f"Entities by Type: {list_entities_by_type}")
    # For each item, it checks if the value (v) is not empty. If it's not empty, it adds an entry to the new dictionary.
    # The key of the new entry is the key from the original dictionary (k), and the value is the first element of the value from the original dictionary (v[0]).
    entities_by_type = {k: v[0] for k, v in list_entities_by_type.items() if v}  
    print (f"Entities by Type after processing: {entities_by_type}")
    entities_by_type_2nd = {k: v[1] if len(v) > 1 else None for k, v in list_entities_by_type.items() if v}  # Only using the second entity if it exists
    print (f"Entities by Type 2nd after processing: {entities_by_type_2nd}")
    

    
    # Low confidence - ask for clarification only for complex commands
    # Skip clarification for simple movement commands
    if confidence < 0.4 and chosen_action and chosen_action not in ['go', 'take', 'leave']:
        return get_llm_clarification(command, chosen_action, entities_by_type)
    
    
    action_map = {
        "go": lambda: playerOne.move(entities_by_type.get("directions")) if entities_by_type.get("directions") is not None else "Directions not specified.",
        "attack": lambda: playerOne.attack(entities_by_type.get("monsters"), entities_by_type.get("weapons") or entities_by_type.get("my_items")) if entities_by_type.get("monsters") is not None else "Monster not specified.",
        "trade": lambda: playerOne.trade_with_npc(
            list_entities_by_type.get("my_items")[0] if list_entities_by_type.get("my_items") else entities_by_type.get("my_items"),
            list_entities_by_type.get("npc_items")[0] if list_entities_by_type.get("npc_items") else entities_by_type.get("npc_items")
        ) if (list_entities_by_type.get("my_items") or entities_by_type.get("my_items")) and (list_entities_by_type.get("npc_items") or entities_by_type.get("npc_items")) else \
              playerOne.talk_to_npc(command), #hack since often you have words that trigger trading, in your questions, but you don't know what to trade
        "talk": lambda: playerOne.talk_to_npc(command) if command is not None else "What to say not specified.",
        "teach": lambda: playerOne.about_to_npc(command) if command is not None else "What to say not specified.",
        "use": lambda: playerOne.use(entities_by_type.get("my_magic") or entities_by_type.get("my_items")) if (entities_by_type.get("my_magic") is not None or entities_by_type.get("my_items") is not None) else "Magic items not specified.",
        "wear": lambda: playerOne.put_on_armor(entities_by_type.get("my_armor")) if entities_by_type.get("my_armor") is not None else \
             playerOne.put_on_armor(entities_by_type.get("my_items")) if entities_by_type.get("my_items") is not None else "Armor not specified.",
        "remove": lambda: playerOne.take_off_armor(entities_by_type.get("wear_armor")) if entities_by_type.get("wear_armor") is not None else "Armor not on.",
        "take": lambda: playerOne.take(list_entities_by_type.get("room_items")) if list_entities_by_type.get("room_items") is not None else "Room items not specified.",
        "leave": lambda: playerOne.leave(list_entities_by_type.get("my_items")) if list_entities_by_type.get("my_items") is not None else "My items not specified.",
        "solve": lambda: playerOne.solve_puzzle(entities_by_type.get("my_magic"),entities_by_type_2nd.get("my_magic")) if entities_by_type.get("my_magic") is not None else "My magic items not specified.", #also 2nd item 
        
        # Pass 'command' into study for describing; the new optional argument 
        # lets us distinguish between "look" vs. the default LLM-based approach
        "describe_location": lambda: playerOne.study(entities_by_type.get("location"), command) if entities_by_type.get("location") is not None else "Nothing to study.",
        "describe_room_item": lambda: playerOne.study(entities_by_type.get("room_items"), command) if entities_by_type.get("room_items") is not None else "No room item to study.",
        "describe_my_item": lambda: playerOne.study(entities_by_type.get("my_items"), command) if entities_by_type.get("my_items") else "No player item to study.",

        "health": lambda: playerOne.health_up(int(re.findall(r'\d+', command)[0])) if re.findall(r'\d+', command) and re.findall(r'\d+', command)[0].isdigit() else "No valid health number specified.",
        "magic word": lambda: playerOne.magic_connections(int(re.findall(r'\d+', command)[0])) if re.findall(r'\d+', command) and re.findall(r'\d+', command)[0].isdigit() else "No valid room number specified.",

        "help": lambda: get_contextual_help(command),
        # Modified cheat action to check for the magic word
        "cheat": lambda: cheat_code(cheat_code=magicword) if magicword in command.lower() or "xyzzy" in command.lower() else "Invalid cheat attempt.",
        "hint": lambda: playerOne.get_trading_hints() if playerOne.current_room.npc else "No one here to ask for hints.",
        "magic": lambda: show_magic_items(),
        "riddle_status": lambda: check_riddle_status(),
        "question": lambda: handle_question_with_llm(command),
        "inventory": lambda: playerOne.inventory_living(),
        "status": lambda: f'Health: {int(playerOne.health)}  Wealth: {playerOne.wealth}  Points: {playerOne.points}  Kills: {playerOne.kills}\n',
        "map": lambda: f"You are in Room {playerOne.current_room.number}.\n\n💡 Check the visual map on the right side of the screen!\n",
    }
    
    action_function = action_map.get(chosen_action)
    
    # If we have a recognized action, execute it
    if chosen_action in action_map:
        # Try to execute the action
        try:
            result = action_function()
        except Exception as e:
            print(f"Error executing {chosen_action}: {e}")
            # Use LLM to help with error recovery
            result = get_llm_error_recovery(command, chosen_action, str(e))
        
        print(f"Action executed: {chosen_action}")
        print(f"Result: {result}")
        
        return result
    
    # Let LLM handle these common commands instead of hardcoded fallbacks
    # Modern LLMs can understand "inventory", "status", "map" naturally
    return get_llm_error_suggestions(command, chosen_action, entities_by_type)

def get_llm_clarification(command, action, entities):
    """Use LLM to clarify ambiguous commands"""
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

I'm uncertain about this command. Help clarify what the player wants.
Detected action: {action}
Entities found: {entities}

Provide a helpful clarification question in 1-2 sentences.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Player said: "{command}"

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    response = ollama_generate(prompt, max_tokens=100, temperature=0.3)
    return f"I'm not sure I understood. {response}\n"

def get_llm_error_suggestions(command, action, entities):
    """Use LLM to suggest alternatives for unrecognized commands"""
    # Get current context
    context_items = []
    if playerOne.current_room.items:
        context_items.append(f"Items here: {', '.join([i.name for i in playerOne.current_room.items[:5]])}")
    if playerOne.current_room.monsters:
        context_items.append(f"Monsters: {', '.join([m.name for m in playerOne.current_room.monsters])}")
    if playerOne.current_room.npc:
        context_items.append(f"NPC: {playerOne.current_room.npc.name}")
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

The player tried a command I don't understand. Suggest 2-3 similar valid commands based on context.

Current room context:
{chr(10).join(context_items)}

Valid command examples: go north, take sword, attack goblin with axe, talk hello

<|eot_id|><|start_header_id|>user<|end_header_id|>

Player tried: "{command}"

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    suggestions = ollama_generate(prompt, max_tokens=150, temperature=0.3)
    return f"I don't understand '{command}'.\n\n💡 Did you mean:\n{suggestions}\n"

def get_llm_error_recovery(command, action, error):
    """Use LLM to help recover from execution errors"""
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

A command failed with an error. Help the player understand what went wrong and what to do.

Action attempted: {action}
Error: {error}

Provide a friendly explanation and suggestion in 2-3 sentences.

<|eot_id|><|start_header_id|>user<|end_header_id|>

Player command: "{command}"

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    explanation = ollama_generate(prompt, max_tokens=100, temperature=0.3)
    return f"Something went wrong: {explanation}\n"

def riddles_and_magic(): 
    room_riddles = ""
    for room_i in rooms.values(): 
        if room_i.riddles:  
            room_riddles += generate_riddle_info(room_i)
    print (room_riddles)
    return room_riddles

def generate_riddle_info(room_i):
    room_riddles = ""
    room_riddles = room_riddles + f"\nRoom Number {room_i.number} has this Riddle:\n{room_i.riddles[0]} \n"
    room_riddles = room_riddles+(f'These items are Needed to solve the riddle: {", ".join(map(str, room_i.riddle_magic_items))}. \n')    
    room_riddles = room_riddles + (f"Hint: {room_i.hints[0]} \n")
    if len(room_i.hints) > 1:
        room_riddles = room_riddles + (f"Hint 2: {room_i.hints[1]} \n")

    which_room_is_it = [(room.number, item.name) for room in rooms.values() for item in room.items if item.name in room_i.riddle_magic_items]
    which_npc_is_it = [(room.number, room.npc.name, item.name) for room in rooms.values() if room.npc for item in room.npc.items if item.name in room_i.riddle_magic_items]
    which_monster_has_it = [(room.number, monster.name, item.name) for room in rooms.values() for monster in room.monsters for item in monster.items if item.name in room_i.riddle_magic_items]
    if which_room_is_it:
        for room_number, item_name in which_room_is_it:
            room_riddles += f'The item "{item_name}" can be found in Room Number {room_number}.\n'
    if which_npc_is_it:
        for room_number, npc_name, item_name in which_npc_is_it:
            room_riddles += f'The item "{item_name}" can be found with the NPC "{npc_name}" in Room Number {room_number}.\n'
    if which_monster_has_it:
        for room_number, monster_name, item_name in which_monster_has_it:
            room_riddles += f'The item "{item_name}" can be found with the Monster "{monster_name}" in Room Number {room_number}.\n'   
    return room_riddles

# This is a helper function that checks if a trophy is already in the list.
# If not, it checks the rooms for a specific condition.
# If the condition is met, it adds the trophy to the list and updates the console.
def check_trophy(trophy, condition, victory_message, victory_action):
    global trophies
    # Check if the trophy is not already in the list
    player_image = None
    trophie_images = []
    console = ""
    if trophy not in trophies:
        all_conditions_met = True
        # Iterate over all rooms
        for room in rooms.values():
            # The condition function is called here with the room as argument.
            # This function is a lambda function that was passed as an argument to check_trophy.
            # It checks a specific condition in the room (e.g., if there are any treasures, monsters, riddles, or if the room is explored).
            if condition(room):
                all_conditions_met = False
        # If all conditions are met (i.e., there are no more treasures, monsters, riddles, or unexplored rooms), the player wins the trophy
        if all_conditions_met:
            console = console + (f"Congratulations, {victory_message}!\n")
            playerOne.drawing_self(victory_action)
            player_image = playerOne.image
            playerOne.current_room.add_treasure(False, f"{trophy.capitalize()}-Trophie")
            temp_console, trophie_images  = playerOne.take(f"{trophy.capitalize()}-Trophie")  # returns the console and the image of the trophy
            trophies.append(trophy)
            print (f"trophies: {trophies}")
            console = console +temp_console + f"Your trophies: {', '.join(trophies)}\n"
    
    player_image = [player_image] if player_image is not None else []
    #trophiie images are a list since take returns lists
    return console, player_image, trophie_images


def talk_to_functions(command, name_of_game, difficulty, game_state, new_game_name, number_of_rooms, llm_diff, llm_mon, 
                      llm_item, name_of_llm, name_of_diffuser, art_style, generate_videos):
    """
    MAIN GAME LOOP FUNCTION - Processes one complete game turn.
    
    Handles:
    - Game state management (new/load/save/play)
    - Command processing via LLM
    - Game world updates (NPCs, monsters, room changes)
    - Trophy/victory condition checking
    - UI state updates for Gradio interface
    
    Returns tuple for Gradio UI:
    (image_gallery, map_image, room_description, room_contents, console_output, 
     status, rooms_visited, inventory, long_description, game_state, room_video)
    
    Game States:
    - "New-Game": Create new adventure with specified parameters
    - "Load-Game": Load saved game from pickle file  
    - "Save-Game": Save current game state
    - "Play-Game": Process player command and update world
    """
    global playerOne, rooms, active_game, list_of_saved_games, trophies, difficulty_g, cheat_mode, npc_introduced, you_started_it,  old_room, art_style_g
    
    def _get_npc_intro_dialogue(npc):
        """Generate appropriate introduction dialogue based on NPC personality"""
        # Ensure all attributes exist for backward compatibility
        npc._ensure_attributes()
            
        if npc.personality_traits['mood'] == 'friendly':
            return "Hello there, traveler! I'm glad to see a friendly face in these dark caves."
        elif npc.personality_traits['mood'] == 'grumpy':
            return "What do you want? Can't a person get some peace in these caves?"
        elif npc.personality_traits['mood'] == 'mysterious':
            return "Ah, another seeker of secrets wanders into my presence..."
        elif npc.personality_traits['mood'] == 'suspicious':
            return "Who are you and what are you doing here? State your business!"
        else:
            return "Greetings, adventurer. What brings you to this place?"
    
    room_image = []
    
    map_image = None
    npc_image = []
    item_images = []
    monster_image = []
    player_image = []
    room_video = None   
    console = ""
    your_status = ""
    rooms_visted = ""
    room_description = ""
    room_long_description = ""
    room_riddles = ""
    room_contents = ""
    inventory1 = ""
    art_style_g = art_style # global variable for art_style in classes
    game_state_selector = "Play-Game"

    if not difficulty:
        difficulty = "easy"
    if not command:
        command = " "
    difficulty = difficulty.lower()
    difficulty_g = difficulty # global variable for difficulty 
    
    if  game_state == "New-Game" or game_state == "Load-Game":
        # Check if Ollama is running
        if not check_ollama_running():
            console = "Error: Ollama is not running. Please start Ollama with 'ollama serve' in a terminal."
            return (room_image, map_image, 
                room_description.strip()+"\n"+room_riddles.strip(),
                room_contents.strip(), console.strip(), your_status,
                rooms_visted.strip(), inventory1.strip(),
                room_long_description, game_state_selector, room_video)
        
        embedings_for_actions()
        initalize_game_varables()
        load_adventure_data()
        set_up_llm_diffuser(diff_name=name_of_diffuser, llm_name=name_of_llm)

    if game_state == "New-Game":
        console = new_game(num_rooms=number_of_rooms, runllm_diff=llm_diff, runllm_mon=llm_mon, 
                           runllm_item=llm_item, name_of_game=new_game_name, generate_videos=generate_videos)
        console = (f"New game: {console}.\n")
        active_game = True
        game_state_selector = gr.Radio(value="Play-Game")

    elif game_state == "Load-Game":
        rooms, playerOne, old_room=load_game_state(name_of_game)
        if rooms is None or playerOne is None:
            console = (f"Failed to load game: {name_of_game}. Please try a different save file or create a new game.\n")
            active_game = False
            game_state_selector = gr.Radio(value="Load-Game")
        else:
            console = (f"Loaded game: {name_of_game}.\n")
            active_game = True
            game_state_selector = gr.Radio(value ="Play-Game")
        
    elif game_state == "Save-Game":
        name_of_game = save_game_state(new_game_name)
        console = (f"Saved game: {name_of_game}. Select Play-Game to continue playing.\n")
        game_state_selector = gr.Radio(value="Play-Game")
        list_of_saved_games = saved_files()

    elif game_state == "Play-Game" and active_game:
        if old_room != playerOne.current_room:
            you_started_it = False
            npc_introduced = False
        
        print("Command:", command)
        
        # Initialize result variable to prevent UnboundLocalError
        result = ""
        
        # Handle debug conversation command first
        cmd_lower = command.lower()
        if cmd_lower == "debug conversation":
            if playerOne.current_room.npc and playerOne.current_room.npc.has_pending_action():
                state = playerOne.current_room.npc.conversation_state
                console = console + f"DEBUG: Pending action: {state['pending_action']}, Offer: {state['last_offer']}\n"
                return console, []
            else:
                console = console + "DEBUG: No pending conversation.\n"
                return console, []
        
        # Use the streamlined command processor
        result = process_command(command, playerOne, playerOne.current_room)
        if isinstance(result, tuple) and len(result) == 2:
            result, item_images = result
            console = console + result
        else:
            console = console + result
            item_images = []

        # return if no active game
    else: 
        print ("Returning from talk_to_functions early", room_image, map_image)
        return (room_image, map_image, 
        room_description.strip()+"\n"+room_riddles.strip(),
        room_contents.strip(), console.strip(), your_status,
        rooms_visted.strip(), inventory1.strip(),
        room_long_description, game_state_selector, room_video)

    # Check if playerOne exists (could be None if loading failed)
    if playerOne is None:
        print("PlayerOne is None, returning early")
        return (room_image, map_image, 
        room_description.strip()+"\n"+room_riddles.strip(),
        room_contents.strip(), console.strip(), your_status,
        rooms_visted.strip(), inventory1.strip(),
        room_long_description, game_state_selector, room_video)

    #Room content
    room_long_description = playerOne.current_room.full_description    
    room_long_description = room_long_description.strip() 
    if difficulty in ["cheat", "easy", "medium"]:
        room_contents = room_contents + playerOne.current_room.inventory_room()
        room_long_description = gr.Textbox(visible=True, value = room_long_description) # togle back on.
    #Hard mode they need to figure out content from description
    elif playerOne.current_room.full_description != playerOne.current_room.description: 
        room_contents = room_contents + playerOne.current_room.full_description # so you figure out contents yourself
        room_long_description = gr.Textbox(visible=False)
    # Update action of monsters
    if playerOne.current_room.monsters:
        monsters = playerOne.current_room.monsters
        monster_content = ""
        for monster in monsters:
            attack_probability = random.randint(0, int(9000/diff_dict[difficulty])) # like difficulty dynamic
            if attack_probability < playerOne.wealth or you_started_it: #if you attack anything it wakes monsterup 
                console = console + monster.monster_attack(playerOne) #moved to monster not player :) 
            monster_content = monster_content + monster.inventory_living()
            if monster.image:
                monster_image.append(monster.image)
        if difficulty in ["cheat", "easy"] and monsters:
            room_contents = room_contents + "Monster posessions: " + monster_content
    
    # Update action of NPC using NPC class
    if playerOne.current_room.npc: 
        npc = playerOne.current_room.npc
        
        # Use enhanced NPC decision making
        npc_action = npc.decide_action(player_in_room=True)
        
        if npc_action and npc_action[0] == "move":
            direction = npc_action[1]
            console = console + npc.move(direction) #move the npc
        elif not npc_introduced:
            # More dynamic introduction based on NPC personality
            npc._ensure_attributes()  # Ensure NPC has all attributes
            
            # Special introduction for MB
            if hasattr(npc, 'is_master_beast') and npc.is_master_beast:
                console += "\n🌟 A stunningly beautiful, woman with glowing eyes turns to greet you: 🌟\n"
                console += npc.llM_generate_self_dialogue("Welcome, young adventurer! I am MB, the Master Beast. *flexes impressively* I know all the secrets of this cave and I'm here to help you succeed! Ask me anything - about riddles, items, monsters, or game strategy. I want you to win! And if you're curious about me... well, I have many talents beyond just helping with your quest. *winks*")
                npc_introduced = True
            elif random.randint(0, 3) == 0 or npc.personality_traits['mood'] in ['friendly', 'cheerful']:
                intro_dialogue = _get_npc_intro_dialogue(npc)
                console = console + npc.llM_generate_self_dialogue(intro_dialogue)
                npc_introduced = True
        else:
            # NPCs may occasionally offer information or comments
            npc._ensure_attributes()  # Ensure NPC has all attributes
            if random.randint(0, 15) == 0 and npc.personality_traits['helpfulness'] in ['very helpful', 'eager to help']:
                # Helpful NPCs might give unsolicited advice
                if playerOne.health < 150:
                    console = console + f"{npc.name} notices your injuries and says: 'You look hurt! You should find some healing items.'\n"
                elif len([i for i in playerOne.items if isinstance(i, Magic)]) > 0:
                    console = console + f"{npc.name} eyes your magic items and says: 'Those might be useful for solving riddles...'\n"
                
            # NPCs may offer quest suggestions
            if random.randint(0, 20) == 0:
                quest_suggestion = playerOne.get_npc_quest_suggestion()
                if quest_suggestion:
                    console = console + quest_suggestion
                
        if difficulty in ["cheat", "easy"]: #NPC Inventory to help
            room_contents = room_contents + (f'NPC Inventory: {npc.inventory_living()}')
            npc._ensure_attributes()  # Ensure NPC has all attributes
            room_contents = room_contents + (f'\nNPC Personality: {npc.personality_traits["mood"]}, {npc.personality_traits["helpfulness"]}\n')
        if npc.image:
            npc_image.append(npc.image)
            
    #Riddles
    if playerOne.current_room.riddles and playerOne.current_room.riddle_solved == False:
        room_riddles = "\n" + "="*50 + "\n"
        room_riddles += "🧩 RIDDLE IN THIS ROOM:\n"
        room_riddles += f"{playerOne.current_room.riddles[0]}\n"
        room_riddles += "\n💡 TO SOLVE: You must bring a MAGIC ITEM that represents the answer!\n"
        room_riddles += "="*50 + "\n"
        if difficulty in ["cheat", "easy"]:
            room_riddles = room_riddles + (f"Hint: {playerOne.current_room.hints[0]} \n")
            if len(playerOne.current_room.hints) > 1:
                room_riddles = room_riddles + (f"Hint 2: {playerOne.current_room.hints[1]} \n")
            # Add helpful instruction about solving
            magic_in_inventory = [item.name for item in playerOne.items if isinstance(item, Magic)]
            if magic_in_inventory:
                room_riddles = room_riddles + f"\n💡 You have magic items that might help: {', '.join(magic_in_inventory)}\n"
                room_riddles = room_riddles + "To solve the riddle, type: solve <magic item name>\n"
            else:
                room_riddles = room_riddles + "\n⚠️  You need magic items to solve this riddle. Find them in other rooms or defeat monsters!\n"
        if cheat_mode and difficulty == "cheat":
            room_riddles = "Cheat: " + generate_riddle_info(playerOne.current_room)
   
    # Check for winning the game or death 
    # The lambda functions are passed as arguments to the check_trophy function.
    # They are not executed here, but inside the check_trophy function.
    # Return trophy images and player images as lists so don't have to check if None.
    console_temp, player_image_temp, item_images_temp = check_trophy("treasure", lambda room: room.items and any(isinstance(item, Treasure) for item in room.items), "you got all the treasures", "celebrating finding treasure")
    console += console_temp
    player_image += player_image_temp
    item_images += item_images_temp

    console_temp, player_image_temp, item_images_temp = check_trophy("monster", lambda room: room.monsters, "you killed all the monsters", "celebrating killing monsters")
    console += console_temp
    player_image += player_image_temp
    item_images += item_images_temp

    console_temp, player_image_temp, item_images_temp = check_trophy("riddle", lambda room: room.riddles and not room.riddle_solved, "you won the game all the riddles solved", "Celebration solving riddles")
    console += console_temp
    player_image += player_image_temp
    item_images += item_images_temp

    console_temp, player_image_temp, item_images_temp = check_trophy("explorer", lambda room: not room.explored, "you explored all the rooms", "Celebration succesfully exploring")
    console += console_temp
    player_image += player_image_temp
    item_images += item_images_temp 
    print (console)

    if playerOne.health <= 0:
        console = console + (f'Health: {int(playerOne.health)}  Wealth: {playerOne.wealth}  Points: {playerOne.points}  Kills: {playerOne.kills}\n')
        console = console + ("You died. Game over.\n")
        victory = ("Died, killed by a monsters")
        playerOne.drawing_self(victory)
        if playerOne.image:
            player_image.append(playerOne.image)
        console = console + (f"Your trophies: {', '.join(trophies)}\n")
        game_state_selector = gr.Radio(value="Load-Game")

    #update the image, description of the room, health, wealth, points, kills, map, inventory
    try:
        if playerOne.current_room.image:
            room_image.append(playerOne.current_room.image)
    except:
        room_image = []

    #update the video of the room
    room_video = getattr(playerOne.current_room, 'video_data', None)
        
    if room_video:
        # Create a temporary file in your custom directory
        temp_file_path = os.path.join(TEMP_VIDEO_DIR, f"room_{playerOne.current_room.number}_video.mp4")
        with open(temp_file_path, 'wb') as temp_video:
            temp_video.write(room_video)
        room_video = temp_file_path
        playerOne.current_room.temp_video_path = temp_file_path
    else:
        room_video = None

    print(f"Room video data available: {'Yes' if room_video is not None else 'No'}")
        
    #Room description
    room_description =  room_description +(f'Room {playerOne.current_room.number}: {playerOne.current_room.description}\n')
    if not playerOne.current_room.explored:
            room_description = room_description + "You haven't been in this room before.\n"
            playerOne.current_room.explored = True # must set before draw_map or seeing if all rooms explored. 
            
    # Special message when entering MB's room
    if playerOne.current_room.number == 1 and playerOne.current_room.npc and hasattr(playerOne.current_room.npc, 'is_master_beast'):
        room_description += "\n✨ A gorgeous, woman stands here - it's MASTER BEAST (MB)! ✨\n"
        room_description += "Just say: 'MB help' or 'Beast, tell me about yourself' or address her naturally!\n"
            
    map_image = draw_map_png(cheat_mode and difficulty == "cheat")
    #Rooms Explored
    visited = [room.number for room in rooms.values() if room.explored]
    not_visited = [room.number for room in rooms.values() if not room.explored]
    rooms_visted = rooms_visted + "Rooms visited: " + str(visited) + "  Rooms to explore: " + str(not_visited) + "\n"
    #Inventory
    inventory1 = "" + playerOne.inventory_living()
    if playerOne.armors:
        inventory1 = inventory1 + "Wearing: " + ", ".join([f'{item.name}, that can protect from {item.protection} damage' for item in playerOne.armors]) + "\n"
         #Player status
    your_status = (f'Health: {int(playerOne.health)}  Wealth: {playerOne.wealth}  Points: {playerOne.points}  Kills: {playerOne.kills}')
    
    old_room = playerOne.current_room  

    image_gallery = (player_image or []) + (npc_image or []) + (room_image or []) + (monster_image or [] ) + (item_images or [])
    #print ("returning from talk_to_functions", player_image, npc_image, monster_image, room_image, item_images)

    return (image_gallery, map_image, 
        room_description.strip()+"\n"+room_riddles.strip(), # use room_description to show riddles
        room_contents.strip(), console.strip(), your_status, 
        rooms_visted.strip(), inventory1.strip(), 
        room_long_description, game_state_selector, room_video)


if not os.path.exists(model_path):
    print("Please select the path for  LLM & DIFF folders. ") 
    llm_model_path = ask_for_folder()

if not os.path.exists(data_path):
    print("Please select the path for saved games, art, and game data .")
    data_path = ask_for_folder()

# Get the current directory- not used
current_dir = os.getcwd()

# Define the directories
image_dir = os.path.join(data_path, "Adventure_Art")
game_dir = os.path.join(data_path, "Adventure_Game_Saved")
# llm_dir no longer needed with Ollama
diff_dir = os.path.join(model_path, "Diffusion_Models")

# Check if the directories exist, if not, create them
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

if not os.path.exists(game_dir):
    os.makedirs(game_dir)

help = """
ADVENTURE GAME COMMANDS:

MOVEMENT:
  go <direction> - Move north, south, east, or west
  
ITEMS:
  take <item> - Pick up an item from the room
  leave <item> - Drop an item from your inventory
  use <item> - Use a magic item to heal yourself
  study <item> - Examine an item closely
  
COMBAT:
  attack <monster> with <weapon> - Fight a monster
  
NPCS:
  talk <message> - Talk to an NPC (e.g., "talk hello")
  trade <your item> for <their item> - Trade with NPC
  hint - Ask NPC for trading suggestions
  teach <question> - Ask NPC about the game world
  
✨ MASTER BEAST (MB) - YOUR ULTIMATE HELPER! ✨
  MB is a mystical creature with glowing eyes who ALWAYS waits in Room 1!
  She knows EVERYTHING about the game and wants you to succeed!
  
  HOW TO TALK TO MB:
  Just address her directly by name! No need to type "talk":
  - "MB, I need help"
  - "Master Beast, where can I find magic items?"
  - "Beast, how do I solve this riddle?"
  - "MB: I'm stuck, what should I do?"
  - "Hey MB, tell me about all the riddles"
  - "Beast, where are the best weapons?"
  - "MB I'm confused about spells"
  
  You can also use the traditional way:
  - "talk I need help"
  - "talk where can I find magic items?"
  
  MB KNOWS EVERYTHING:
  - Exact solutions to every riddle
  - Which room has which items
  - Which monsters carry magic items
  - The best strategies for winning
  - All game mechanics and secrets
  - What you should do next
  
  💡 TIP: Just type "MB" or "Beast" followed by your question - like talking to a friend!
  
RIDDLES (IMPORTANT!):
  magic - Show all your magic items (riddle solvers)
  riddle status - Check riddle info for current room
  solve <magic item> - Use ONE magic item to solve a riddle
  solve <item1> and <item2> - Use TWO magic items if riddle needs both
  
  Example: If you're in a room with a riddle and have "Elixir of Life":
    Type: solve Elixir of Life
  
  Tips for riddles:
  - Type 'magic' to see what magic items you have
  - Read the riddle carefully
  - Check the hints (shown in easy mode)
  - Magic items are the KEY to solving riddles
  - Some riddles need 1 item, others need 2
  - The magic item will disappear when used correctly
  - ASK MB IN ROOM 1 FOR EXACT SOLUTIONS!
  
ROOM:
  details - Get detailed room description
  draw - Redraw room image
  
OTHER:
  help - Show this help message
  
COMMON QUESTIONS (just type them!):
  How do I solve riddles?
  What do I do with magic items?
  How do I fight monsters?
  Where do I find magic items?
  
HINTS:
  - Monsters often carry magic items needed for riddles
  - NPCs can trade valuable items
  - Explore all rooms to find magic items
  - In easy mode, riddle hints are shown automatically
  - MB IN ROOM 1 KNOWS EVERYTHING - USE HER!
  
QUICK START:
  1. Go to Room 1 and talk to MB for guidance
  2. Get magic items (kill monsters, trade, explore)
  3. Find rooms with riddles
  4. Use magic items to solve riddles
  5. Win the game!
  
🌟 REMEMBER: MB in Room 1 is your best friend! She'll never let you down! 🌟
"""

active_game = False
diff_dict = {"cheat": 1, "easy": 1, "medium": 2, "hard": 3}

playerOne = None
rooms = {} # dictionary of rooms room number is they key, room objecters
list_of_saved_games = saved_files()

# Check if Ollama is running before getting models
if check_ollama_running():
    list_of_llms = saved_llms()
else:
    list_of_llms = ["Please start Ollama first with 'ollama serve'"]
    print("\n*** WARNING: Ollama is not running! ***")
    print("Please start Ollama in another terminal with: ollama serve")
    print("Then refresh this page or restart the application.\n")

list_of_diffusers = saved_diffusers()

TEMP_VIDEO_DIR = setup_temp_directory()

with gr.Blocks(theme=gr.themes.Default(font=[gr.themes.GoogleFont("IBM Plex Mono")])) as web_interface:
    if not list_of_saved_games:
        list_of_saved_games = ["No saved games"]
    if not list_of_llms or list_of_llms == ["No Ollama models found"]:
        list_of_llms = ["No Ollama models found - please install models with 'ollama pull <model>'"]
    if not list_of_diffusers:
        list_of_diffusers = ["No saved diffusers"]

    with gr.Row():
        gr.Markdown("<h1 style='text-align: center;'>JMR's Colossal Cave Adventure</h1>")
    with gr.Row():
        with gr.Column():
            image_output = gr.Gallery(label="Room Image")
            video_output = gr.Video(label="Room Video", format="mp4")
            text_input = gr.Textbox(autofocus=True, label="What do you want to do>")
            difficulty_selector = gr.Radio(["Cheat", "Easy", "Medium", "Hard"], value = "Easy", label="Difficulty")
            game_state_selector = gr.Radio(["Play-Game","Load-Game", "New-Game", "Save-Game"], label="Game State")
            
            with gr.Row():
                game_selector = gr.Dropdown(list_of_saved_games, visible=False, label="Select a saved game to load" , value = list_of_saved_games[0])
                name_of_game = gr.Textbox(visible = False, label="Name of game")  
                number_of_rooms = gr.Textbox(visible = False, label="Number of rooms")
                art_style = gr.Textbox(visible = False, label="Art Style")
            with gr.Row():
                room_npc_desc_art = gr.Checkbox(visible = False, value= True, label="Room & NPC Art and Description")
                generate_videos = gr.Checkbox(label="Generate Videos", value=False, interactive=True)  # Now supported on Mac with MPS!
                monster_desc_art = gr.Checkbox(visible = False, value= True, label="Monster Art & Description")
                item_desc_art = gr.Checkbox(visible = False, value= True, label="Item Art and Description")
            with gr.Row():
                llm_select = gr.Dropdown(list_of_llms, visible=False, label="Select LLM", value = list_of_llms[0])
                diff_select = gr.Dropdown(list_of_diffusers, visible=False, label="Select a Diffuser", value = list_of_diffusers[0])
                
            submit_button = gr.Button("Submit", visible=False)
        with gr.Column():
            room_desc_output = gr.Textbox(label="Room Description")
            contents_output = gr.Textbox(label="Room Contents")
            console_output = gr.Textbox(label="Console")
            status_output = gr.Textbox(label="Health, Wealth, Points, Killed")
            # Use a Row here to place map_output and map_image_output side by side
            with gr.Row():
                map_image_output = gr.Image(label=" ")
                with gr.Column():
                    rooms_explored = gr.Textbox(label="Rooms Explored")
                    inventory_out = gr.Textbox(label="Inventory")
            long_description_output = gr.Textbox(label="Room Long Description")
    
    # Conditionally show the game_selector
    def update_game_selector(game_state):
        if game_state == "Load-Game":
            return (gr.Dropdown(visible=True), gr.Textbox(visible=False), gr.Textbox(visible=False), gr.Checkbox(visible=False), gr.Checkbox(visible=False), 
        gr.Checkbox(visible=False), gr.Button(visible=True), gr.Dropdown(visible=True), gr.Dropdown(visible=True), gr.Textbox(visible=True), gr.Checkbox(visible=False))
        elif game_state == "Save-Game":
            return (gr.Dropdown(visible=False), gr.Textbox(visible=True), gr.Textbox(visible=False), gr.Checkbox(visible=False), gr.Checkbox(visible=False), 
        gr.Checkbox(visible=False), gr.Button(visible=True), gr.Dropdown(visible=False), gr.Dropdown(visible=False), gr.Textbox(visible=False), gr.Checkbox(visible=False))
        elif game_state == "New-Game":
            return (gr.Dropdown(visible=False), gr.Textbox(visible=True), gr.Textbox(visible=True), gr.Checkbox(visible=True), gr.Checkbox(visible=True), 
                gr.Checkbox(visible=True), gr.Button(visible=True), gr.Dropdown(visible=True), gr.Dropdown(visible=True), gr.Textbox(visible=True), gr.Checkbox(visible=True, interactive=True))  # Mac MPS video support!
        elif game_state == "Play-Game":
            return (gr.Dropdown(visible=False), gr.Textbox(visible=False), gr.Textbox(visible=False), gr.Checkbox(visible=False), gr.Checkbox(visible=False), 
                gr.Checkbox(visible=False), gr.Button(visible=True), gr.Dropdown(visible=False), gr.Dropdown(visible=False), gr.Textbox(visible=False), gr.Checkbox(visible=False))
        else:
            return (gr.Dropdown(visible=False), gr.Textbox(visible=False), gr.Textbox(visible=False), gr.Checkbox(visible=False) , gr.Checkbox(visible=False), 
        gr.Checkbox(visible=False), gr.Button(visible=False), gr.Dropdown(visible=False), gr.Dropdown(visible=False), gr.Textbox(visible=False), gr.Checkbox(visible=False))

    game_state_selector.change(update_game_selector, inputs=[game_state_selector], 
                               outputs=[game_selector, name_of_game, number_of_rooms, room_npc_desc_art, monster_desc_art, item_desc_art, submit_button, llm_select, diff_select, art_style, generate_videos])
    # Now the function will be called when the submit button is clicked or Enter is pressed
    submit_inputs = [text_input, game_selector, difficulty_selector, game_state_selector, name_of_game, number_of_rooms, room_npc_desc_art, 
                     monster_desc_art, item_desc_art, llm_select, diff_select, art_style, generate_videos]
    submit_outputs = [image_output, map_image_output, room_desc_output, contents_output, console_output, status_output, rooms_explored, 
                      inventory_out, long_description_output, game_state_selector, video_output]
    
    text_input.submit(  # Set up the submit event
        fn=talk_to_functions,
        inputs=submit_inputs,
        outputs=submit_outputs
    )
    submit_button.click(
        fn=talk_to_functions,
        inputs=submit_inputs,
        outputs=submit_outputs
    )

# Ensure mflux is installed for Mac compatibility
ensure_mflux_installed()

try:
    web_interface.launch(share=True)
except:
    web_interface.launch(share=False)

"""
=== AREAS FOR LLM EDITING AND ENHANCEMENT ===

KEY MODIFICATION POINTS:

1. COMMAND PROCESSING (lines ~3600-4200):
   - llm_process_command(): Modify LLM prompts for better command understanding
   - execute_action(): Add new game actions or modify existing ones
   - process_command(): Adjust command routing logic

2. GAME CLASSES (lines ~700-3100):
   - Player class: Add new player abilities or modify existing actions
   - NPC class: Enhance dialogue system or trading mechanics
   - Room class: Add new room features or modify connections
   - Item classes: Create new item types or modify existing ones

3. GAME WORLD GENERATION (lines ~500-700):
   - new_game(): Modify game initialization parameters
   - populating_rooms_random(): Adjust item/monster distribution
   - connecting_rooms(): Change room connection algorithms

4. UI INTEGRATION (lines ~4300-4900):
   - talk_to_functions(): Main game loop - modify game state handling
   - Gradio interface: Adjust UI components or add new features

5. LLM INTEGRATION (lines ~3400-3600):
   - ollama_generate(): Modify LLM parameters or add new models
   - handle_question_with_llm(): Enhance question-answering system

GLOBAL VARIABLES (see initalize_game_varables() for details):
- rooms: Main game world dictionary
- playerOne: Player character object  
- active_game: Game state flag
- All other game state tracked in global scope

ITEM HANDLING PATTERN:
- All items inherit from AdventureGame base class
- Use _find_item_smart() for intelligent name matching
- LLM processes item commands and returns structured entities
- Game executes actions using returned entity lists
"""   