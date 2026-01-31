# Colossal Cave Adventure with Local LLM

An interactive text-based adventure game inspired by the classic "Colossal Cave Adventure", enhanced with local Language Models (LLMs) for natural language processing and diffusion models for dynamic AI-generated artwork.

**Author:** Jonathan M. Rothberg (@jmrothberg)

---

## Features

- **Two Adventure Games:**
  - `LMM_adventure_Dec_7_25.py` - Fully LLM-driven adventure where the AI acts as game master
  - `Colossal_Cave_Aug_2_25.py` - Classic adventure with AI-generated images and videos

- **AI-Generated Art:** Diffusion models create room artwork and item illustrations on demand
- **Video Generation:** Generate atmospheric videos for rooms using Pyramid Flow or Wan2.2
- **Natural Language Commands:** Talk naturally with NPCs and issue commands in plain English
- **Local LLM Integration:** Uses Ollama for fully private, local AI inference
- **Web Interface:** Beautiful Gradio-based UI for easy gameplay
- **Classic Gameplay:** Explore caverns, collect treasures, solve puzzles, battle monsters

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/jmrothberg/Collosol-Cave-with-local-LLM.git
cd Collosol-Cave-with-local-LLM
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama (Required for LLM)

See [Setting Up Local LLMs](#setting-up-local-llms-with-ollama) section below.

### 5. Run the Game

```bash
# LLM-driven adventure (recommended)
python LMM_adventure_Dec_7_25.py

# Classic adventure with video support
python Colossal_Cave_Aug_2_25.py
```

---

## System Requirements

### Minimum Requirements
- **Python:** 3.10 or 3.11 (3.12 has some compatibility issues)
- **RAM:** 16GB (32GB+ recommended for video generation)
- **Storage:** 10GB for code + dependencies, 20-100GB for models

### For Image Generation
- **GPU:** NVIDIA GPU with 8GB+ VRAM, or Apple Silicon Mac with MPS
- **CUDA:** 11.8+ for NVIDIA GPUs

### For Video Generation
- **GPU:** NVIDIA GPU with 24GB+ VRAM, or Apple Silicon Mac with 32GB+ unified memory
- **CUDA:** 12.4+ recommended for video generation

---

## Setting Up Local LLMs with Ollama

The games use [Ollama](https://ollama.ai) to run local LLMs. Ollama provides a simple way to run various open-source language models locally.

### Installing Ollama

#### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### macOS
```bash
# Using Homebrew
brew install ollama

# Or download from https://ollama.com/download
```

#### Windows
Download the installer from [ollama.com/download](https://ollama.com/download)

### Starting Ollama

After installation, start the Ollama service:

```bash
ollama serve
```

Or on macOS/Windows, the Ollama app runs in the background automatically.

### Downloading LLM Models

Download models that work well with the adventure games:

```bash
# Recommended models for gameplay (choose one or more)
ollama pull llama3.2:3b      # Fast, good for quick responses (3GB)
ollama pull llama3.1:8b      # Balanced quality and speed (5GB)
ollama pull qwen2.5:7b       # Good for world bible generation (4GB)
ollama pull mistral:7b       # Another good option (4GB)

# For world bible generation (needs more capacity)
ollama pull llama3.1:70b     # Best quality but requires 40GB+ VRAM
ollama pull qwen2.5:32b      # High quality alternative (20GB)
```

### Recommended Model Combinations

| Use Case | Fast Model (Gameplay) | Heavy Model (World Bible) |
|----------|----------------------|---------------------------|
| Low VRAM (8GB) | llama3.2:3b | qwen2.5:7b |
| Medium VRAM (16GB) | llama3.1:8b | qwen2.5:14b |
| High VRAM (24GB+) | llama3.1:8b | qwen2.5:32b |
| Apple Silicon (16GB) | llama3.2:3b | llama3.1:8b |
| Apple Silicon (32GB+) | llama3.1:8b | qwen2.5:14b |

### Verifying Ollama Setup

```bash
# Check available models
ollama list

# Test a model
ollama run llama3.2:3b "Hello, are you ready for an adventure?"
```

---

## Setting Up Diffusion Models for Image Generation

The games can generate AI artwork using various diffusion models. You have several options:

### Option 1: Z-Image-Turbo (Recommended for Speed)

Z-Image-Turbo generates high-quality images in just 9 inference steps.

```bash
# Download using Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Tongyi-MAI/Z-Image-Turbo', 
                  local_dir='./Z-Image-Turbo',
                  local_dir_use_symlinks=False)
"
```

Or place in your models directory:
```
/data/Diffusion_Models/Z-Image-Turbo/  (Linux)
~/Diffusion_Models/Z-Image-Turbo/       (Mac)
```

### Option 2: FLUX.1-dev (High Quality)

FLUX produces exceptional image quality but requires more VRAM.

```bash
# Requires Hugging Face account and accepting model license
# Set HF_TOKEN environment variable first
export HF_TOKEN=your_token_here

python -c "
from huggingface_hub import snapshot_download
snapshot_download('black-forest-labs/FLUX.1-dev',
                  local_dir='/data/Diffusion_Models/FLUX.1-dev',
                  local_dir_use_symlinks=False)
"
```

### Option 3: Stable Diffusion XL (SD XL)

A good balance of quality and speed.

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('stabilityai/stable-diffusion-xl-base-1.0',
                  local_dir='/data/Diffusion_Models/SDXL',
                  local_dir_use_symlinks=False)
"
```

### Where to Place Models

The games look for models in these locations (in order):

1. Environment variable: `DIFFUSION_MODELS_DIR`
2. `/data/Diffusion_Models/` (Linux)
3. `/Users/<username>/Diffusion_Models/` (macOS)
4. `./Diffusion_Models/` (relative to game directory)

---

## Setting Up Video Generation (Optional)

Video generation creates atmospheric clips for game rooms. This requires significant GPU resources.

### Pyramid Flow (Recommended for Mac)

```bash
# Download the miniflux version (smaller, works on Mac)
python miniflux_download.py

# Or the full SD3 version (higher quality, more VRAM needed)
python huggingfacesnapshotpyramid.py
```

Models will be downloaded to `/data/pyramid-flow-miniflux/` or `/data/pyramid-flow-sd3/`.

### Wan2.2 (Best Quality)

Wan2.2 models require manual download due to license requirements:

1. Visit [Alibaba Wan on Hugging Face](https://huggingface.co/alibaba-pai)
2. Accept the model license
3. Download to `/data/Wan2.2-T2V-A14B/` or similar

### Video Generation Requirements

| Model | VRAM Required | Video Length |
|-------|--------------|--------------|
| Pyramid-Flow-Miniflux | 16GB+ | 5-10 sec |
| Pyramid-Flow-SD3 | 24GB+ | 5-10 sec |
| Wan2.2-TI2V-5B | 24GB+ | 5-10 sec |
| Wan2.2-T2V-A14B | 40GB+ | 5-10 sec |

---

## Running the Games

### LMM Adventure (LLM-Driven)

This is the main game where the LLM acts as game master:

```bash
python LMM_adventure_Dec_7_25.py
```

Features:
- Generate a "World Bible" that defines your adventure's theme and story
- Choose from preset themes (Tolkien, Star Wars, Cyberpunk, etc.) or create your own
- The LLM dynamically creates NPCs, locations, puzzles, and narrative
- AI-generated artwork for every room and important item

### Colossal Cave (Classic)

The original-style adventure with enhanced AI features:

```bash
python Colossal_Cave_Aug_2_25.py
```

Features:
- Classic room-based exploration
- Pre-defined puzzles and riddles
- Video generation for immersive room experiences
- Compatible with Mac MPS for video generation

### Command Line Options

```bash
# Specify LLM model
python LMM_adventure_Dec_7_25.py --model llama3.1:8b

# Specify max tokens for responses
python LMM_adventure_Dec_7_25.py --max_tokens 800

# Enable debug output
python LMM_adventure_Dec_7_25.py --debug
```

---

## Game Commands

### Basic Commands

| Command | Description |
|---------|-------------|
| `go <direction>` | Move north, south, east, or west |
| `take <item>` | Pick up an item from the room |
| `leave <item>` | Drop an item in the room |
| `use <item>` | Use an item from your inventory |
| `study <item>` | Examine an item closely |
| `attack <monster> with <weapon>` | Combat a monster |
| `trade <item> for <item>` | Trade with NPCs |
| `talk <message>` | Speak to NPCs |

### Special Commands

| Command | Description |
|---------|-------------|
| `details` | Get LLM description of current room |
| `draw` | Generate new artwork for current room |
| `video` | Generate video for current room (if enabled) |
| `help` | Show all commands |
| `inventory` | Show your items |
| `map` | Display the game map |

### Puzzle Commands

| Command | Description |
|---------|-------------|
| `solve <item1> and <item2>` | Solve puzzles with magic items |

### Cheat Commands

| Command | Description |
|---------|-------------|
| `xyzzy` | Activate cheat mode |
| `magicword <room>` | Open magical passages |
| `health <value>` | Set health value |

---

## Configuration

### Environment Variables

Create a `.env` file (copy from `.env.example`):

```bash
# Hugging Face token for downloading gated models
HF_TOKEN=your_huggingface_token_here

# Custom model directories
DIFFUSION_MODELS_DIR=/path/to/your/models

# GPU selection for diffusion
DIFFUSION_GPU=0

# Ollama API key (only for remote servers)
OLLAMA_API_KEY=your_key_here
```

### In-Game Settings

The Gradio UI provides controls for:
- LLM model selection
- Image generation model selection
- Theme/style for generated artwork
- Max tokens for LLM responses
- Debug output toggle

---

## Project Structure

```
Colossal_Cave/
├── LMM_adventure_Dec_7_25.py    # Main LLM-driven adventure game
├── Colossal_Cave_Aug_2_25.py    # Classic adventure with video
├── diffusion_manager.py         # Unified diffusion model interface
├── complete_instruction.py      # Game help system
├── adventure_dataRA.json        # Game data (rooms, items, NPCs)
│
├── # Download helpers
├── downloader_ltx.py            # Download LTX-Video model
├── miniflux_download.py         # Download Pyramid-Flow-Miniflux
├── huggingfacesnapshotpyramid.py # Download Pyramid-Flow-SD3
│
├── # Test scripts
├── Z_image_TestCode.py          # Test Z-Image-Turbo
├── flux1_test_mini_dec_19_24.py # Test FLUX model
├── vidiofromtext_pyramid_Dec_25_24.py # Pyramid Flow video UI
├── generate_1_5B_gradio_lightning_8_19_25.py # Wan2.2 video UI
│
├── # Video generation libraries (do not modify)
├── Pyramid_Flow/                # Pyramid Flow model code
├── LTX-Video/                   # LTX Video model code
├── wan/                         # Wan2.2 model code
│
├── # Generated content
├── Generated_Art/               # AI-generated images
├── Adventure_Art/               # Pre-generated game art
├── Adventure_Game_Saved/        # Saved games
├── temp_videos/                 # Generated videos
│
├── # Configuration
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

---

## Troubleshooting

### Ollama Issues

**"Ollama not running"**
```bash
# Start Ollama service
ollama serve

# Or check if it's already running
curl http://localhost:11434/api/tags
```

**"Model not found"**
```bash
# List installed models
ollama list

# Pull the required model
ollama pull llama3.1:8b
```

### GPU/CUDA Issues

**"CUDA out of memory"**
- Use a smaller model (e.g., llama3.2:3b instead of llama3.1:8b)
- Close other GPU-using applications
- Reduce image resolution in settings
- Enable CPU offloading in the code

**"MPS not available" (Mac)**
- Update to macOS 12.3 or later
- Update PyTorch: `pip install --upgrade torch`

### Image Generation Issues

**"Model not found"**
- Check that models are in the correct directory
- Verify with: `ls /data/Diffusion_Models/` or equivalent

**"Low quality images"**
- Try Z-Image-Turbo or FLUX for better quality
- Increase inference steps in settings

### Import Errors

**"ModuleNotFoundError"**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

---

## Tips for Best Experience

1. **Start with a good LLM:** llama3.1:8b offers a great balance for gameplay
2. **Generate a World Bible first:** This creates consistent story, NPCs, and objectives
3. **Use themes:** The preset themes provide better artwork and narrative consistency
4. **Save often:** Use the save game feature before dangerous encounters
5. **Explore everything:** Talk to NPCs, examine items, try different commands

---

## License

MIT License

---

## Acknowledgments

- Inspired by the original "Colossal Cave Adventure" by Will Crowther and Don Woods
- Uses [Ollama](https://ollama.ai) for local LLM inference
- Image generation powered by [Diffusers](https://huggingface.co/docs/diffusers)
- Video generation using Pyramid-Flow and Wan2.2 models
- Web UI built with [Gradio](https://gradio.app)

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## Contact

- GitHub: [@jmrothberg](https://github.com/jmrothberg)
- Project: [Colossal-Cave-with-local-LLM](https://github.com/jmrothberg/Collosol-Cave-with-local-LLM)
