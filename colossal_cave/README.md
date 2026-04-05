# Colossal Cave Classic

A traditional text adventure with 25+ pre-defined rooms, monsters, treasures, riddles, and NPCs. The gameplay is **code-driven**: all rooms, items, and encounters are defined in `adventure_dataRA.json`. LLMs support the experience by parsing natural-language commands and generating NPC dialogue, while Diffusers create AI artwork and Pyramid Flow generates room videos.

---

## How It Works

- **Game world** loaded from `adventure_dataRA.json` -- 25+ rooms with monsters, riddles, treasures, magic items, and NPCs
- **Ollama** runs a local LLM to parse player commands ("go north", "attack troll with sword") and drive NPC conversations
- **Sentence Transformers** + fuzzy matching map free-text input to game actions
- **Diffusers** (PyTorch) generate AI artwork for rooms using FLUX, Stable Diffusion 3.5, or SDXL
- **Pyramid Flow** generates AI video for each room (optional)
- **Gradio** web UI for gameplay

---

## Quick Start

### 1. Install Ollama

Download from [ollama.com](https://ollama.com), then pull a model:

```bash
ollama pull llama3.1
```

### 2. Install Python Dependencies

```bash
cd colossal_cave
pip install -r requirements.txt
# Or for full features (video gen, etc.):
pip install -r colossal_requirements.txt
```

### 3. Run

```bash
python Colossal_Cave_Aug_2_25.py
```

---

## System Requirements

- **Platform:** Apple Silicon Mac (MPS) or Linux with NVIDIA GPU (CUDA)
- **RAM:** 16GB+ recommended
- **Python:** 3.10 or 3.11
- **Ollama:** Required for LLM inference
- **Video generation:** PyTorch 2.9.0+ on Mac; CUDA on Linux
- **Multi-GPU:** Distributes LLM, diffusion, and video across GPUs when available

---

## Game Commands

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
| `details` | Get LLM description of current room |
| `draw` | Generate new artwork for current room |
| `help` | Show all commands |
| `inventory` | Show your items |
| `map` | Display the game map |

---

## Files

| File | Description |
|------|-------------|
| `Colossal_Cave_Aug_2_25.py` | Main game |
| `adventure_dataRA.json` | Room, NPC, monster, and item data |
| `data_generator_adventure_Aug_2_25.py` | Generates `adventure_dataRA.json` procedurally |
| `complete_instruction.py` | In-game help system |
| `diffusion_manager.py` | Unified diffusion model interface |
| `mini_SentenceTransformer_Feb_1_dict.py` | Sentence embeddings for command parsing |
| `map.png` | Game world map |
| `requirements.txt` | Core Python dependencies |
| `colossal_requirements.txt` | Extended dependencies (video, diffusion) |
| `README_pyramid_MPS_Video_Generation.md` | Pyramid Flow video setup guide |

### Image & Video Tools

| File | Description |
|------|-------------|
| `mini_flux_MTS_CUDA_8_1_25.py` | FLUX image gen (CUDA, multi-GPU) |
| `mini_pyramid_MTS_CUDA_8_1_25.py` | Pyramid Flow video gen (CUDA) |
| `flux1_test_mini_dec_19_24.py` | FLUX.1 test script |
| `vidiofromtext_pyramid_Dec_25_24.py` | Text-to-video via Pyramid Flow |
| `vidiofromtext_ltx_Jan_7_25.py` | Text-to-video via LTX-Video |
| `miniflux_download.py` | FLUX model downloader |
| `huggingfacesnapshotpyramid.py` | Pyramid Flow model downloader |
| `dowloadflow.py` | Pyramid Flow downloader |
| `downloader_ltx.py` | LTX-Video model downloader |
| `Z_image_TestCode.py` | Z-Image-Turbo test script |

---

## Troubleshooting

**"numpy.dtype size changed"** or **"_ARRAY_API not found"**
```bash
pip install --upgrade numpy scipy scikit-learn pandas
```

**"ModuleNotFoundError"**
```bash
pip install torch diffusers sentence-transformers fuzzywuzzy gradio ollama Pillow
```

---

## License

MIT License
