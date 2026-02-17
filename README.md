# Colossal Cave Adventure with Local LLM

An interactive text-based adventure game inspired by the classic "Colossal Cave Adventure", enhanced with local Language Models (LLMs) for natural language processing and FLUX image generation for dynamic AI-generated artwork — all running natively on Apple Silicon via MLX.

**Author:** Jonathan M. Rothberg (@jmrothberg)

---

## Features

- **Two Adventure Games:**
  - `LMM_adventure_Feb_15_26.py` - Fully LLM-driven adventure where the AI acts as game master (MLX-LM + MFLUX)
  - `Colossal_Cave_Aug_2_25.py` - Classic adventure with AI-generated images and videos

- **AI-Generated Art:** MFLUX (FLUX on Apple Silicon) creates room artwork and item illustrations on demand
- **Natural Language Commands:** Talk naturally with NPCs and issue commands in plain English
- **Local LLM Integration:** Uses MLX-LM for fully private, local AI inference on Apple Silicon
- **Web Interface:** Gradio-based UI for easy gameplay
- **Classic Gameplay:** Explore caverns, collect treasures, solve puzzles, battle monsters

---

## Quick Start (Apple Silicon Mac)

### 1. Clone the Repository

```bash
git clone https://github.com/jmrothberg/Collosol-Cave-with-local-LLM.git
cd Collosol-Cave-with-local-LLM
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install mlx-lm mflux gradio
```

### 4. Download Models

**LLM models** (place in `~/MLX_Models/`):
```bash
# Example: download a quantized Qwen model
huggingface-cli download Qwen/Qwen3-30B-A3B-8bit --local-dir ~/MLX_Models/Qwen3-30B-A3B-8bit
```

**Image generation model** (place in `~/Diffusion_Models/`):
```bash
# FLUX.2-klein-9B MLX 8-bit (recommended — fast, 4-step distilled)
huggingface-cli download AITRADER/FLUX2-klein-9B-mlx-8bit --local-dir ~/Diffusion_Models/FLUX2-klein-9B-mlx-8bit
```

### 5. Run the Game

```bash
python LMM_adventure_Feb_15_26.py
```

---

## System Requirements

- **Hardware:** Apple Silicon Mac (M1/M2/M3/M4)
- **RAM:** 32GB+ unified memory recommended (16GB minimum)
- **Python:** 3.10 or 3.11
- **Storage:** ~20GB for FLUX.2-klein-9B-mlx-8bit, plus LLM models

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

## Running the Game

### LMM Adventure (LLM-Driven) — Main Game

```bash
python LMM_adventure_Feb_15_26.py
```

Features:
- Generate a "World Bible" that defines your adventure's theme and story
- Choose from preset themes (Tolkien, Star Wars, Cyberpunk, etc.) or create your own
- The LLM dynamically creates NPCs, locations, puzzles, and narrative
- AI-generated artwork for every room and important item

### Command Line Options

```bash
# Specify LLM model
python LMM_adventure_Feb_15_26.py --model Qwen3-30B-A3B-8bit

# Disable image generation
python LMM_adventure_Feb_15_26.py --no-images

# Run in CLI mode (no Gradio UI)
python LMM_adventure_Feb_15_26.py --cli

# Set player name
python LMM_adventure_Feb_15_26.py --player "Gandalf"
```

### Colossal Cave (Classic)

The original-style adventure with enhanced AI features:

```bash
python Colossal_Cave_Aug_2_25.py
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
| `help` | Show all commands |
| `inventory` | Show your items |
| `map` | Display the game map |

---

## Project Structure

```
Colossal_Cave/
├── LMM_adventure_Feb_15_26.py  # Main LLM-driven adventure (MLX-LM + MFLUX)
├── mflux_image_gen.py          # MFLUX image generator (FLUX.1 + FLUX.2)
├── Colossal_Cave_Aug_2_25.py   # Classic adventure with video
├── diffusion_manager.py        # Legacy diffusion interface (CUDA-based)
├── complete_instruction.py     # Game help system
│
├── Generated_Art/              # AI-generated images
├── Adventure_Game_Saved/       # Saved games
│
├── .venv/                      # Python virtual environment
├── requirements.txt            # Python dependencies
└── README.md                   # This file
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
- LLM inference powered by [MLX-LM](https://github.com/ml-explore/mlx-lm)
- Image generation powered by [MFLUX](https://github.com/filipstrand/mflux) (FLUX on Apple Silicon)
- Web UI built with [Gradio](https://gradio.app)

---

## Contact

- GitHub: [@jmrothberg](https://github.com/jmrothberg)
- Project: [Colossal-Cave-with-local-LLM](https://github.com/jmrothberg/Collosol-Cave-with-local-LLM)
