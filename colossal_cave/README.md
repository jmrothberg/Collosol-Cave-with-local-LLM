# Colossal Cave Classic

Text adventure: the game picks **25+ rooms** from an **87-room pool** in `adventure_dataRA.json`. **MLX folders under `~/MLX_Models`** (or optional **Ollama**) parse commands and drive NPC/MB chat. **Diffusers** generate room/item art. **Pyramid Flow** (optional) generates room videos.

**Run:** `cd colossal_cave && python colossal_cave_July_5_26.py` → Gradio at `http://127.0.0.1:7860` (or next free port).

---

## Quick Start

### Automated (macOS / Linux)

```bash
# from repo root
chmod +x scripts/startup.sh
./scripts/startup.sh --setup    # create .venv, install deps, mkdir ~/MLX_Models on Apple Silicon
./scripts/startup.sh            # launch game
```

### Manual

```bash
# from repo root
python3 -m venv .venv && source .venv/bin/activate
pip install -r colossal_cave/requirements.txt
# Apple Silicon — required for gemma4_unified and other VLM MLX folders:
pip install --upgrade "mlx-vlm>=0.6.1" "mlx-lm>=0.19.0"
mkdir -p ~/MLX_Models
# optional: video + MFLUX
pip install -r colossal_cave/colossal_requirements.txt

cd colossal_cave && python colossal_cave_July_5_26.py
```

**MLX weights:** put converted models in `~/MLX_Models/<folder-name>/` (`config.json` + `*.safetensors`). Pick `<folder-name>` in the UI (no prefix).

**Optional Ollama:** `ollama serve` in another terminal; pick `ollama:<tag>` in the dropdown.

On first launch, pick a **models folder** (parent of `Diffusion_Models/`) and a **data folder** (defaults to this directory).

**Load-Game** restores save state only — no LLM/diffuser setup or MB intro until your first Play turn.

---

## MLX text LLMs vs Ollama vs diffusion

There are **three separate things**:

| What | Where it lives | What the game does |
|------|----------------|-------------------|
| **MLX_Models folders** | `~/MLX_Models/<name>/` (e.g. `gemma-4-12B-it-8bit/`) | **Primary text LLM path.** Pick any folder in the dropdown — the game loads `config.json` + `*.safetensors` directly via `mlx-vlm` or `mlx-lm`. **Ollama not required.** |
| **Ollama (optional)** | Background server (`ollama serve`) | Pick `ollama:<tag>` in the dropdown (e.g. `ollama:qwen3.6:27b-mlx-bf16`) to use models from `ollama list` instead of a local folder. |
| **Diffusion MLX** | `Diffusion_Models/*mlx*` (e.g. `FLUX2-klein-9B-mlx-8bit/`) | **Room art only** — loaded via **mflux**, not related to chat LLMs. |

### Run any folder under `~/MLX_Models`

```
~/MLX_Models/gemma-4-12B-it-8bit/
  config.json, model-00001-of-00003.safetensors, tokenizer.json, ...
```

1. Put converted MLX weights in `~/MLX_Models/<folder-name>/` (must include `config.json`).
2. In the game UI, set **Play LLM** and/or **Creation LLM** to `<folder-name>` (no prefix).
3. Inference libs (installed by `scripts/startup.sh` on Apple Silicon):
   - **VLM** (`gemma4_unified`, `qwen3_5`, etc.): `mlx-vlm>=0.6.1` (gemma4 needs 0.6.1+)
   - **Text-only** (Llama, Mistral, etc.): `mlx-lm>=0.19.0`

Override the models directory: `COLOSSAL_CAVE_MLX_MODELS_DIR=/path/to/models`

Default play model: `COLOSSAL_CAVE_PLAY_LLM=gemma-4-12B-it-8bit`

### Optional: use Ollama instead

If you prefer Ollama-managed models, pick entries prefixed with `ollama:` in the dropdown (requires `ollama serve` and `ollama list` showing that tag).

### Master Beast (MB)

MB lives in **Room 1** but answers from **any room** when you ask for help, hints, or riddles — you do not need to say "MB" or travel back first. Examples: `help me with the riddle`, `MB what should I do?`, `how do I solve this?`

### Debug log

After load or new game, check `Adventure_Game_Saved/Debug_Logs/<save>_play_debug.log`:

- `LLM_SETUP` — `play_backend=mlx` or `play_backend=ollama`, plus `mlx_folders_on_disk=[...]`
- `route=npc_mb_remote` — help/riddle routed to MB from another room
- `MLX_ERROR` — direct MLX load/generate failed (upgrade mlx-vlm, bad weights, etc.)
- `LLM_MISSING` — selected name not found on disk and not in Ollama

---

## World data

`data_generator_adventure_Aug_2_25.py` builds `adventure_dataRA.json`:

```bash
cd colossal_cave && python data_generator_adventure_Aug_2_25.py
```

Lists of room text, riddles, monsters, weapons, NPC names/descriptions → JSON. The game loads and validates it via `engine/data.py` (≥25 rooms; riddles need `magic_items` + `action_words`).

---

## Local models (images)

The diffusion dropdown lists **only folders** under `Diffusion_Models/` — no remote Hugging Face IDs.

| Model | Mac speed | Notes |
|-------|-----------|--------|
| **FLUX2-klein-9B-mlx-8bit** | Fastest | MLX via mflux (`mlx` in folder name) |
| **Z-Image-Turbo** | Medium | 9 steps, guidance 0. Needs `diffusers>=0.36` |
| **FLUX.1-dev** | Slow | Full diffusers FLUX on MPS; 4 steps |

**Mac paths:** `Diffusion_Models/` under your home directory, or the folder you choose at launch.

---

## Video (Pyramid only)

Main game uses **Pyramid Flow** (`pyramid-flow-miniflux`), not LTX. Enable video in the UI; needs PyTorch **2.9.1+**, FFmpeg, and the model at `../pyramid-flow-miniflux`. See `README_pyramid_MPS_Video_Generation.md`.

Install extras: `INSTALL_VIDEO=1 ./scripts/startup.sh --setup`

---

## Architecture

```
command → preprocess → try_local_command (fast-path: move, take, solve, help, …)
       → help/riddle/MB phrasing → MB in room 1 (engine/mlx_llm.py or ollama:)
       → if needed: LLM JSON parser (engine/llm_commands.py) → execute_action()
       → Gradio UI (ui/gradio_app.py) + Diffusers/mflux images + optional Pyramid video
```

Packages: `engine/` (incl. `mlx_llm.py`, `fast_path.py`, `debug_log.py`), `ui/`, `media/`, `config/`, `tests/`

---

## Testing

```bash
./scripts/startup.sh --test
# or:
cd colossal_cave
COLOSSAL_CAVE_SKIP_VECTOR_MODEL=1 COLOSSAL_CAVE_HEADLESS=1 pytest -m "not gpu and not ollama and not mlx"
```

**188+ CI-safe tests**: fast-path, MB routing, MLX engine mocks, dual LLM, organic map, puzzles.

```bash
# Local Mac: live MLX / Ollama
COLOSSAL_CAVE_PLAY_LLM=gemma-4-12B-it-8bit pytest -m "ollama or mlx"
```

**Env vars:**
- `COLOSSAL_CAVE_PLAY_LLM` — default `gemma-4-12B-it-8bit` (fast play/command model)
- `COLOSSAL_CAVE_CREATION_LLM` — optional override for new-game generation
- `COLOSSAL_CAVE_MLX_MODELS_DIR` — default `~/MLX_Models`
- `COLOSSAL_CAVE_SKIP_VECTOR_MODEL=1` — skip legacy SentenceTransformer load in tests/CI
- `COLOSSAL_CAVE_HEADLESS=1` — headless test mode

Optional: `pytest -m gpu` (diffusion smoke).

---

## Game commands

| Command | Action |
|---------|--------|
| `go n/s/e/w` | Move (you can enter monster rooms; attack there) |
| `take` / `leave` / `use` / `wear` | Items |
| `attack <monster> with <weapon>` | Combat |
| `help me with the riddle` / `MB, …` | Master Beast helper (MB in room 1; works from any room) |
| `talk <msg>` | NPC in current room |
| `solve <magic item>` | Riddles |
| `inventory` / `status` / `map` / `help` | Info |

MB (Master Beast) knows full game state — ask for hints, item locations, or what to do next.

---

## Organic cave map

- **Load-Game:** Map relayouts from your saved connections (jagged chambers + brown tunnels). Topology unchanged.
- **New-Game:** Cave graph topology (branches, dead ends, loops) — not a square grid.
- Room art is **clipped** (not stretched) into unique **jagged polygons** per room; current room gets a gold outline.
- **Tight layout:** connected rooms sit close together on the map.
- **Navigation:** Type movement commands — the map is your compass:
  - `go east` / `go e` — follow the **E** label on a tunnel from your current room
  - `go north` / `n`, `go south` / `s`, `go east` / `e`, `go west` / `w` — cardinals when the tunnel is straight
  - **8-way diagonals** when tunnels are angled: `go northeast` / `ne`, `go southeast` / `se`, `go southwest` / `sw`, `go northwest` / `nw` (hyphens ok: `south-east`)
  - **Exits on entry:** When you enter a room, `Exits from here: ...` prints at the top of the console and in the room description
- Preview: `python scripts/preview_organic_map.py`

**2D vs 3D:** In-game map is a 2D PNG in Gradio. Click-to-move on the map is not implemented yet; a 3D fly-through would be a separate WebGL project.

---

## Key files

| File | Purpose |
|------|---------|
| `colossal_cave_July_5_26.py` | Main game |
| `engine/mlx_llm.py` | Direct MLX load from `~/MLX_Models` |
| `data_generator_adventure_Aug_2_25.py` | Regenerate `adventure_dataRA.json` |
| `adventure_dataRA.json` | World data (rooms, riddles, monsters, NPCs) |
| `engine/fast_path.py` | Local commands without LLM |
| `engine/debug_log.py` | Per-save play debug logs |
| `requirements.txt` / `colossal_requirements.txt` | Dependencies |
| `../scripts/startup.sh` | macOS/Linux setup + launch |

---

## Troubleshooting

**Ollama empty / not responding** — only matters if you picked `ollama:<tag>` in the dropdown. MLX folders under `~/MLX_Models` work without Ollama.

**`Error generating response` with `gemma-4-12B-it-8bit`** — that model uses `gemma4_unified`, which needs **mlx-vlm ≥ 0.6.1**:

```bash
pip install --upgrade "mlx-vlm>=0.6.1"
# or re-run: ./scripts/startup.sh --setup
```

Until upgraded, MB still answers via offline guidance (no LLM prose), but game-command parsing will fail too.

**`Error generating response` with other MLX folders** — install `mlx-vlm` (VLM types) or `mlx-lm` (text-only). Check debug log for `MLX_ERROR`.

**`model 'gemma-4-12B-it-8bit' not found` (Ollama 404)** — you selected an Ollama tag, not the MLX folder. Pick `gemma-4-12B-it-8bit` (no `ollama:` prefix) to load from `~/MLX_Models/` directly.

**MB says nothing useful** — check debug log for `MLX_ERROR` or `MB_EMPTY_LLM`; upgrade mlx-vlm; offline fallback still gives riddle hints and exits.

**Movement / combat issues** — debug log at `Adventure_Game_Saved/Debug_Logs/<save_name>_play_debug.log` (exits, monsters, routing, responses).

**Video on Mac** — PyTorch 2.9.1+ stable, FFmpeg, `pyramid-flow-miniflux` present. Use `INSTALL_VIDEO=1 ./scripts/startup.sh --setup`.

---

## License

MIT License
