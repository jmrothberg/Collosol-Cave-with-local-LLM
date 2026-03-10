# Colossal Cave Adventure with Local LLMs

Two interactive text-based adventure games inspired by the classic "Colossal Cave Adventure", enhanced with local Language Models for natural language processing and AI-generated artwork.

**Author:** Jonathan M. Rothberg (@jmrothberg)

---

## Two Adventure Games

### 1. LMM Adventure (LLM-Driven) — `LMM_adventure_Feb_15_26.py`

A fully procedural adventure where the LLM acts as game master. The AI dynamically creates the entire world — rooms, NPCs, items, puzzles, and narrative — based on a "World Bible" theme you choose or create.

- **LLM:** [MLX-LM](https://github.com/ml-explore/mlx-lm) — runs local LLMs natively on Apple Silicon via MLX
- **Images:** [MFLUX](https://github.com/filipstrand/mflux) — FLUX image generation on Apple Silicon via MLX
- **Platform:** Apple Silicon Mac only (M1/M2/M3/M4)
- **UI:** Gradio web interface
- Preset themes: Tolkien, Star Wars, Cyberpunk, and more
- AI-generated artwork for every room and important item

### 2. Colossal Cave Classic — `Colossal_Cave_Aug_2_25.py`

A traditional adventure with pre-defined rooms, NPCs, monsters, riddles, and treasures loaded from `adventure_dataRA.json`. Uses LLMs for natural language command parsing and NPC conversations, plus AI-generated images and videos.

- **LLM:** [Ollama](https://ollama.com) — local LLM server for command parsing and NPC dialogue
- **Images:** [Diffusers](https://github.com/huggingface/diffusers) (PyTorch) — supports FLUX, Stable Diffusion 3.5, SDXL
- **Video:** [Pyramid Flow](https://github.com/jy0205/Pyramid-Flow) — AI-generated room videos (Mac MPS or Linux CUDA)
- **Command parsing:** [sentence-transformers](https://www.sbert.net/) for vector embeddings + fuzzy matching
- **Platform:** Apple Silicon Mac (MPS) or Linux with NVIDIA GPU (CUDA); supports multi-GPU configurations
- **UI:** Gradio web interface
- 25+ pre-defined cave rooms with monsters, treasures, and riddles
- NPC trading, combat, and puzzle-solving

---

## Quick Start

### Clone the Repository

```bash
git clone https://github.com/jmrothberg/Collosol-Cave-with-local-LLM.git
cd Collosol-Cave-with-local-LLM
python3 -m venv .venv
source .venv/bin/activate
```

### Option A: LMM Adventure (Apple Silicon Mac)

```bash
# Install dependencies
pip install mlx-lm mflux gradio

# Download an LLM (place in ~/MLX_Models/)
huggingface-cli download Qwen/Qwen3-30B-A3B-8bit --local-dir ~/MLX_Models/Qwen3-30B-A3B-8bit

# Download FLUX image model (place in ~/Diffusion_Models/)
huggingface-cli download AITRADER/FLUX2-klein-9B-mlx-8bit --local-dir ~/Diffusion_Models/FLUX2-klein-9B-mlx-8bit

# Run
python LMM_adventure_Feb_15_26.py
```

### Option B: Colossal Cave Classic (Mac or Linux)

```bash
# Install Ollama (https://ollama.com)
# Then pull a model:
ollama pull llama3.1

# Install Python dependencies
pip install torch diffusers sentence-transformers fuzzywuzzy gradio ollama Pillow

# Run
python Colossal_Cave_Aug_2_25.py
```

For video generation, also install Pyramid Flow (see Pyramid-Flow/ directory).

---

## System Requirements

### LMM Adventure (MLX-LM + MFLUX)
- **Hardware:** Apple Silicon Mac (M1/M2/M3/M4)
- **RAM:** 32GB+ unified memory recommended (16GB minimum)
- **Python:** 3.10 or 3.11
- **Storage:** ~20GB for FLUX.2-klein-9B-mlx-8bit, plus LLM models

### Colossal Cave Classic (Ollama + Diffusers)
- **Hardware:** Apple Silicon Mac (MPS) or Linux with NVIDIA GPU (CUDA)
- **RAM:** 16GB+ recommended
- **Python:** 3.10 or 3.11
- **Ollama:** Required for LLM inference
- **Video generation:** PyTorch 2.9.0+ on Mac; CUDA on Linux

---

## Setting Up LLMs with MLX-LM

The game uses [MLX-LM](https://github.com/ml-explore/mlx-lm) to run local LLMs natively on Apple Silicon.

### Downloading MLX Models

Place MLX-format models in `~/MLX_Models/`. The game auto-discovers any directory containing a `config.json`.

```bash
# Recommended models (choose based on your RAM)
# 16GB Mac:
huggingface-cli download Qwen/Qwen3-4B-MLX --local-dir ~/MLX_Models/Qwen3-4B-MLX

# 32GB+ Mac:
huggingface-cli download Qwen/Qwen3-30B-A3B-8bit --local-dir ~/MLX_Models/Qwen3-30B-A3B-8bit

# 64GB+ Mac:
huggingface-cli download mlx-community/Qwen2.5-32B-Instruct-8bit --local-dir ~/MLX_Models/Qwen2.5-32B-Instruct-8bit
```

### In-Game Model Selection

The Gradio UI provides a dropdown to select and load any MLX model at runtime. No restart needed.

---

## Setting Up Image Generation with MFLUX

The game uses [MFLUX](https://github.com/filipstrand/mflux) to run FLUX image generation natively on Apple Silicon via MLX.

### Recommended: FLUX.2-klein-9B (MLX 8-bit)

The distilled FLUX.2-klein-9B model generates high-quality images in just **4 inference steps** — fast enough for real-time gameplay.

```bash
huggingface-cli download AITRADER/FLUX2-klein-9B-mlx-8bit \
  --local-dir ~/Diffusion_Models/FLUX2-klein-9B-mlx-8bit
```

### Alternative: FLUX.1-dev

Higher quality but slower (20 steps):

```bash
# Requires accepting the license at huggingface.co/black-forest-labs/FLUX.1-dev
huggingface-cli download black-forest-labs/FLUX.1-dev \
  --local-dir ~/Diffusion_Models/FLUX.1-dev
```

### Where Models Are Found

The game looks for FLUX models in `~/Diffusion_Models/` and lists any directory starting with `FLUX` in the MFLUX Model dropdown.

---

## Running the Games

### LMM Adventure

```bash
python LMM_adventure_Feb_15_26.py
```

Features:
- Generate a "World Bible" that defines your adventure's theme and story
- Choose from preset themes (Tolkien, Star Wars, Cyberpunk, etc.) or create your own
- The LLM dynamically creates NPCs, locations, puzzles, and narrative
- AI-generated artwork for every room and important item

Command line options:

```bash
python LMM_adventure_Feb_15_26.py --model Qwen3-30B-A3B-8bit   # Specify LLM model
python LMM_adventure_Feb_15_26.py --no-images                    # Disable image generation
python LMM_adventure_Feb_15_26.py --cli                          # CLI mode (no Gradio UI)
python LMM_adventure_Feb_15_26.py --player "Gandalf"             # Set player name
```

### Colossal Cave Classic

```bash
python Colossal_Cave_Aug_2_25.py
```

Features:
- Pre-defined cave with 25+ rooms, monsters, treasures, and riddles (from `adventure_dataRA.json`)
- Ollama LLM parses natural language commands and drives NPC conversations
- Multiple diffuser models: FLUX, Stable Diffusion 3.5, SDXL
- Optional AI-generated video for each room via Pyramid Flow
- NPC trading, combat, riddle-solving, and magic items
- Supports Mac MPS and Linux CUDA (including multi-GPU setups)

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
| `help` | Show all commands |
| `inventory` | Show your items |
| `map` | Display the game map |

---

## Project Structure

```
Colossal_Cave/
├── LMM_adventure_Feb_15_26.py      # LLM-driven adventure (MLX-LM + MFLUX)
├── mflux_image_gen.py              # MFLUX image generator (FLUX.1 + FLUX.2)
├── Colossal_Cave_Aug_2_25.py       # Classic adventure (Ollama + Diffusers + Video)
├── adventure_dataRA.json           # Room/NPC/monster/item data for Classic game
├── data_generator_adventure_Aug_2_25.py  # Generates adventure_dataRA.json
├── diffusion_manager.py            # Diffusers image interface (for Classic game)
├── complete_instruction.py         # Game help system
│
├── Pyramid-Flow/                   # Pyramid Flow video generation module
├── pyramid-flow-miniflux/          # Pyramid Flow model weights (Mac)
├── Generated_Art/                  # AI-generated images
├── Adventure_Game_Saved/           # Saved games
│
├── .venv/                          # Python virtual environment
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Troubleshooting

### Import Errors

**"numpy.dtype size changed"** or **"_ARRAY_API not found"**
```bash
# Upgrade numpy-dependent packages
pip install --upgrade numpy scipy scikit-learn pandas
```

**"ModuleNotFoundError"**
```bash
source .venv/bin/activate
pip install mlx-lm mflux gradio
```

### Image Generation Issues

**"Model not found"**
- Check that FLUX models are in `~/Diffusion_Models/`
- Directory names should start with `FLUX` (e.g., `FLUX2-klein-9B-mlx-8bit`)

**Images are slow**
- Use FLUX.2-klein-9B (4 steps) instead of FLUX.1-dev (20 steps)
- Reduce resolution in `mflux_image_gen.py` (default: 768x768)

### LLM Issues

**"No MLX models found"**
- Place MLX model directories in `~/MLX_Models/`
- Each model directory must contain a `config.json`

---

## Tips for Best Experience

1. **Start with a good LLM:** Qwen3-30B-A3B-8bit is an excellent balance of quality and speed
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
- LMM Adventure: [MLX-LM](https://github.com/ml-explore/mlx-lm) for LLM inference, [MFLUX](https://github.com/filipstrand/mflux) for image generation
- Classic Game: [Ollama](https://ollama.com) for LLM inference, [Diffusers](https://github.com/huggingface/diffusers) for image generation, [Pyramid Flow](https://github.com/jy0205/Pyramid-Flow) for video generation
- Web UI built with [Gradio](https://gradio.app)

---

## Contact

- GitHub: [@jmrothberg](https://github.com/jmrothberg)
- Project: [Colossal-Cave-with-local-LLM](https://github.com/jmrothberg/Collosol-Cave-with-local-LLM)
