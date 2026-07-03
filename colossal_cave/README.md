# Colossal Cave Classic

A traditional text adventure with 25+ pre-defined rooms, monsters, treasures,
riddles, and NPCs. The gameplay is **code-driven**: all rooms, items, and
encounters are defined in `adventure_dataRA.json`. A local LLM (via Ollama)
parses natural-language commands and drives NPC dialogue, while Diffusers create
AI artwork and (optionally) Pyramid Flow generates per-room videos.

---

## How It Works

- **Game world** loaded from `adventure_dataRA.json` -- 25+ rooms with monsters,
  riddles, treasures, magic items, and NPCs (including the special helper NPC
  **Master Beast / "MB"** who knows the entire game state).
- **Ollama** runs a local LLM to parse player commands ("go north", "attack
  troll with sword") and to drive in-character NPC conversations.
- **Command parsing**: Ollama in JSON mode (`format="json"`) returns
  `{chosen_action, entity_categories, confidence}`; **fuzzywuzzy** then snaps
  partial item names to real inventory / room items in `Player._find_item_smart`.
- **Diffusers** (PyTorch) generate AI artwork for rooms / NPCs / items / monsters
  using FLUX, Stable Diffusion 3.5, or SDXL. Auto-detects MPS (Apple Silicon)
  or CUDA.
- **Pyramid Flow** generates per-room AI video (optional; works on CUDA, and
  on Mac MPS with PyTorch 2.9.0+).
- **Gradio** web UI for gameplay, with image gallery and map.

> Note: `sentence-transformers` is currently imported and a model is loaded at
> startup (`vector_model = SentenceTransformer('all-mpnet-base-v2', ...)`),
> but it is **not used** by the current command pipeline -- Ollama JSON-mode
> parsing replaced it. The legacy parser lives in
> `mini_SentenceTransformer_Feb_1_dict.py` and is **not imported** by the main
> game; it is kept for reference.

---

## Quick Start

### 1. Install Ollama and pull a model

Download from [ollama.com](https://ollama.com), then pull a model:

```bash
ollama pull llama3.1
```

In a separate terminal, keep Ollama running: `ollama serve`.

### 2. Install Python dependencies

```bash
cd colossal_cave
pip install -r requirements.txt          # core runtime (game + image gen)
# optional, full features (video generation, extra utilities, MFLUX):
pip install -r colossal_requirements.txt
```

### 3. Run

```bash
python Colossal_Cave_Aug_2_25.py
```

On first launch you may be prompted to pick:

- A **models folder** (where Ollama / diffusion / sentence-transformer caches go).
- A **data folder** (for `adventure_dataRA.json`, generated art, and saved games).

---

## System Requirements

- **Platform:** Apple Silicon Mac (MPS) or Linux with NVIDIA GPU (CUDA).
- **RAM:** 16 GB+ recommended; more for larger diffusion models.
- **Python:** 3.10 or 3.11 (Pyramid Flow has matrix-shape issues on 3.12).
- **Ollama:** required for LLM inference; must be running (`ollama serve`).
- **Video generation:** PyTorch 2.9.0+ on Mac (Conv3D MPS support) or CUDA on
  Linux. See `README_pyramid_MPS_Video_Generation.md`.
- **Multi-GPU:** distributes LLM / diffusion / video across GPUs when ≥2 are
  available; falls back to a single device otherwise.

---

## Game Commands

### Core verbs

| Command | Description |
|---------|-------------|
| `go <direction>` / `n` `s` `e` `w` | Move (north / south / east / west) |
| `take <item>` / `take all` | Pick up one item, or every item in the room |
| `leave <item>` / `leave all` | Drop an item (or everything) |
| `use <magic item>` | Use a magic item to heal / boost yourself |
| `wear <armor>` / `remove <armor>` | Put on / take off armor |
| `study <item>` / `look around` | Examine an item or describe the room |
| `attack <monster> with <weapon>` | Combat a monster |
| `trade <my item> for <their item>` | Trade with the NPC in the current room |
| `talk <message>` | Speak to the NPC in the current room |

### Riddles & magic

| Command | Description |
|---------|-------------|
| `magic` | Show the magic items in your inventory (these solve riddles) |
| `riddle status` | Show this room's riddle, the required magic item(s), and whether you have them |
| `solve <magic item>` | Use a magic item to solve the riddle in this room |
| `solve <item1> and <item2>` | Some riddles need two magic items |

### Talking to MB (Master Beast) — your in-game helper

MB lives in **Room 1** and knows the entire game state. Just address her by name:

- `MB what should I do?`
- `Master Beast, where can I find magic items?`
- `Beast, how do I solve this riddle?`
- `MB I need help with the troll`

She will reference your current health, inventory, riddle progress, and item /
monster locations in her answer.

### Status & navigation

| Command | Description |
|---------|-------------|
| `inventory` / `inv` / `i` | Show your inventory |
| `status` | Show health, wealth, points, kills |
| `map` | Quick text reminder; the visual map is on the right side of the UI |
| `help` / `?` | Show full help (commands + complete instructions) |

### Cheat mode (optional)

- `xyzzy` enables cheat mode. Each new game also has a **random magic word**
  (e.g. `abracadabra`, `shazam`, `presto`, ...) printed in the console.
- In cheat mode you can:
  - See the full map (including unexplored rooms).
  - `<magicword> <room number>` -- create new magical exits from a room.
  - `health <number>` -- set your health directly.
  - Use `cheat` difficulty to surface exact riddle solutions in MB's hints.

---

## Files in this folder

### Core game (used at runtime)

| File | Description |
|------|-------------|
| `Colossal_Cave_Aug_2_25.py` | Main game (Gradio UI + Ollama + Diffusers + optional Pyramid Flow). Run this. |
| `adventure_dataRA.json` | Room descriptions, NPC names, monsters, weapons, armor, treasures, magic items, riddles, hints. Loaded at startup. |
| `complete_instruction.py` | `get_complete_instructions()` -- in-game help text bundled with `help`. Imported by the main game. |
| `requirements.txt` | Minimal core dependencies (game runtime + image generation). |
| `colossal_requirements.txt` | Extended dependencies (video generation, MFLUX, spaCy, dev tools). |
| `map.png` | Last generated map image (overwritten on every turn by `draw_map_png`). |
| `Adventure_Art/` | Generated images (rooms, NPCs, monsters, items, player). |
| `Adventure_Game_Saved/` | Pickled save files (`Adv_*.pkl`). |
| `temp_videos/` | Temporary `.mp4` files for room videos passed to the Gradio video player. |
| `PLAN_BETTER_ADVENTURE.md` | Working notes for upcoming engine + UX improvements. |
| `README.md` | This file. |
| `README_pyramid_MPS_Video_Generation.md` | Mac MPS + Pyramid Flow setup guide (video generation). |

### Content & data tools (used as needed)

| File | Description |
|------|-------------|
| `data_generator_adventure_Aug_2_25.py` | Standalone script that procedurally generates `adventure_dataRA.json` (rooms, NPCs, items, riddles, hints). Run this if you want to regenerate the world data. |
| `mini_SentenceTransformer_Feb_1_dict.py` | **Legacy** sentence-embedding command parser used in early versions. **Not imported** by the current main game; kept for reference and historical comparison with the Ollama-JSON path. |

### Image generation: standalone test / utility scripts

These are **not imported by the main game**. They are independent test scripts
useful for debugging a model on your machine, experimenting with new pipelines,
or running image generation outside the adventure UI.

| File | Description |
|------|-------------|
| `flux1_test_mini_dec_19_24.py` | FLUX.1-dev sanity-check script. Loads a local FLUX model from `/data/Diffusion_Models/FLUX.1-dev`, generates `flux-dev.png`. |
| `mini_flux_MTS_CUDA_8_1_25.py` | FLUX with multi-GPU support tuned for 2x 24 GB RTX 4090; also includes Mac MPS path. Demonstrates the "balanced" device-map pattern used by the main game. |
| `Z_image_TestCode.py` | Z-Image-Turbo test harness (interactive prompt loop, ~9-step inference). Saves output to `Generated_Art/`. |
| `diffusion_manager.py` | Experimental unified `ImageGenerator` interface (FLUX / SD3 / SDXL / Z-Image). **Not imported** by the current main game; useful as a reference for the planned media-layer split (see `PLAN_BETTER_ADVENTURE.md`). |
| `generate_1_5B_gradio_lightning_8_19_25.py` | Wan 2.2 video generation Gradio UI (Wan T2V / I2V / TI2V) with Lightning LoRA support. Independent of Pyramid Flow. |

### Video generation: standalone test scripts

These are **not imported by the main game** -- they are separate apps you can
run directly. The main game uses Pyramid Flow internally via `sys.path` injection
(see top of `Colossal_Cave_Aug_2_25.py`).

| File | Description |
|------|-------------|
| `mini_pyramid_MTS_CUDA_8_1_25.py` | Pyramid Flow video generation tuned for CUDA with 2x 24 GB GPUs (no CPU offload, larger resolutions, 5-second clips). |
| `vidiofromtext_pyramid_Dec_25_24.py` | Pyramid Flow text-to-video and image-to-video Gradio UI. Opens at `http://127.0.0.1:7860`. |
| `vidiofromtext_ltx_Jan_7_25.py` | LTX-Video text-to-video and image-to-video Gradio UI (Lightricks LTX-Video model). |

### Model downloaders (run once)

Each downloader saves to `/data/...` by default; edit `model_path` in the file
to change the destination. They use `huggingface_hub.snapshot_download`.

| File | Downloads |
|------|-----------|
| `miniflux_download.py` | `rain1011/pyramid-flow-miniflux` (~8 GB) -- recommended for Mac and 16-24 GB GPUs. |
| `huggingfacesnapshotpyramid.py` | Same as above, plus a commented-out option for the larger `pyramid-flow-sd3`. |
| `dowloadflow.py` | `rain1011/pyramid-flow-sd3` (the SD3-based larger Pyramid Flow model). |
| `downloader_ltx.py` | `Lightricks/LTX-Video` (~15 GB). |

---

## Troubleshooting

**"`numpy.dtype size changed`" or "`_ARRAY_API not found`"**

```bash
pip install --upgrade numpy scipy scikit-learn pandas
```

**"`ModuleNotFoundError`"**

```bash
pip install torch diffusers sentence-transformers fuzzywuzzy gradio ollama Pillow
```

**"`cannot import name 'FLAX_WEIGHTS_NAME' from 'transformers.utils'`"**

This means `transformers >= 5.x` is installed against an older `diffusers`.
Upgrade `diffusers`:

```bash
pip install -U "diffusers>=0.36.0"
```

**Ollama not responding**

In a separate terminal:

```bash
ollama serve
```

The game checks Ollama at startup; if it isn't running you'll see a clear
warning and the LLM dropdown will be empty.

**Mac video generation prerequisites** -- see
`README_pyramid_MPS_Video_Generation.md` for the PyTorch 2.9.0+ install steps,
Conv3D MPS verification, and FFmpeg requirement.

---

## License

MIT License
