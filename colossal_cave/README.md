# Colossal Cave Classic

Text adventure: the game picks **25+ rooms** from an **87-room pool** in `adventure_dataRA.json`. **Ollama** parses commands and drives NPC chat. **Diffusers** generate room/item art. **Pyramid Flow** (optional) generates room videos.

**Run:** `cd colossal_cave && python colossal_cave_July_5_26.py` → Gradio at `http://127.0.0.1:7860` (or next free port).

---

## Quick Start

```bash
# from repo root
python3 -m venv .venv && source .venv/bin/activate
pip install -r colossal_cave/requirements.txt
# optional: video + MFLUX
pip install -r colossal_cave/colossal_requirements.txt

ollama pull gemma3:4b-it-q8_0   # or any model you prefer
ollama serve                   # separate terminal
cd colossal_cave && python colossal_cave_July_5_26.py
```

On first launch, pick a **models folder** (parent of `Diffusion_Models/`) and a **data folder** (defaults to this directory).

**Load-Game** restores save state only — no LLM/diffuser setup or MB intro until your first Play turn.

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

**Mac paths:** `Diffusion_Models/` under `/Users/jonathanrothberg` or `/Users/jonathanrothberg_1`, or the folder you choose at startup.

---

## Video (Pyramid only)

Main game uses **Pyramid Flow** (`pyramid-flow-miniflux`), not LTX. Enable video in the UI; needs PyTorch **2.9.1+**, FFmpeg, and the model at `../pyramid-flow-miniflux`. See `README_pyramid_MPS_Video_Generation.md`.

---

## Architecture

```
command → preprocess → try_local_command (fast-path: move, take, solve, help, …)
       → if needed: Ollama JSON (engine/llm_commands.py) → execute_action()
       → Gradio UI (ui/gradio_app.py) + Diffusers/mflux images + optional Pyramid video
```

Packages: `engine/`, `ui/`, `media/`, `config/`, `tests/`

---

## Testing

```bash
cd colossal_cave
COLOSSAL_CAVE_SKIP_VECTOR_MODEL=1 COLOSSAL_CAVE_HEADLESS=1 pytest -m "not gpu and not ollama and not mlx"
```

**110+ CI-safe tests**: fast-path (wear/remove/use/solve), dual LLM, category take, organic map, puzzles.

```bash
# Local: live play LLM (gemma)
COLOSSAL_CAVE_PLAY_LLM=gemma-4-12B-it-8bit pytest -m "ollama or mlx"
```

**Dual LLM env vars:**
- `COLOSSAL_CAVE_PLAY_LLM` — default `gemma-4-12B-it-8bit` (fast play/command model)
- `COLOSSAL_CAVE_CREATION_LLM` — optional override for new-game generation
- `COLOSSAL_CAVE_MLX_MODELS_DIR` — default `~/MLX_Models`

Optional: `pytest -m gpu` (diffusion smoke).

---

## Game commands

| Command | Action |
|---------|--------|
| `go n/s/e/w` | Move (you can enter monster rooms; attack there) |
| `take` / `leave` / `use` / `wear` | Items |
| `attack <monster> with <weapon>` | Combat |
| `talk <msg>` / `MB, …` | NPC / Master Beast helper |
| `solve <magic item>` | Riddles |
| `inventory` / `status` / `map` / `help` | Info |

MB (Master Beast) in Room 1 knows game state — ask for hints or what to do next.

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
| `data_generator_adventure_Aug_2_25.py` | Regenerate `adventure_dataRA.json` |
| `adventure_dataRA.json` | World data (rooms, riddles, monsters, NPCs) |
| `engine/fast_path.py` | Local commands without Ollama |
| `engine/debug_log.py` | Per-save play debug logs |
| `requirements.txt` / `colossal_requirements.txt` | Dependencies |

---

## Troubleshooting

**Ollama empty / not responding** — run `ollama serve` in another terminal.

**MB says nothing** — restart after updates; Qwen models may need thinking stripped; offline fallback applies when LLM returns empty.

**Movement / combat issues** — debug log at `Adventure_Game_Saved/Debug_Logs/<save_name>_play_debug.log` (exits, monsters, routing, responses).

**Video on Mac** — PyTorch 2.9.1+ stable, FFmpeg, `pyramid-flow-miniflux` present.

---

## License

MIT License
