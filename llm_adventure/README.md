# LLM Adventure

An LLM-driven, turn-based text adventure where **the AI is the game master**. You type actions; a local LLM writes the story and emits JSON directives that the engine executes to update game state, generate images, and keep the world consistent.

As local LLMs keep getting better -- cleaner JSON, smarter tool use, richer narrative -- the game automatically improves with no code changes.

---

## Quick Start (Apple Silicon Mac)

```bash
cd llm_adventure

# Install dependencies
pip install mlx-lm mflux gradio

# Download an LLM (place in ~/MLX_Models/)
huggingface-cli download Qwen/Qwen3-30B-A3B-8bit --local-dir ~/MLX_Models/Qwen3-30B-A3B-8bit

# Download FLUX image model (place in ~/Diffusion_Models/)
huggingface-cli download AITRADER/FLUX2-klein-9B-mlx-8bit --local-dir ~/Diffusion_Models/FLUX2-klein-9B-mlx-8bit

# Run
python LMM_adventure_Feb_15_26.py
```

Command-line options:

```bash
python LMM_adventure_Feb_15_26.py --model Qwen3-30B-A3B-8bit   # Specify LLM model
python LMM_adventure_Feb_15_26.py --no-images                    # Disable image generation
python LMM_adventure_Feb_15_26.py --cli                          # CLI mode (no Gradio UI)
python LMM_adventure_Feb_15_26.py --player "Gandalf"             # Set player name
```

---

## System Requirements

- **Hardware:** Apple Silicon Mac (M1/M2/M3/M4)
- **RAM:** 32GB+ unified memory recommended (16GB minimum)
- **Python:** 3.10 or 3.11
- **Storage:** ~20GB for FLUX.2-klein-9B-mlx-8bit, plus LLM models

---

## How It Works

### How a turn works
1. You enter an action (e.g., "open chest", "go north", "examine crystal key").
2. The turn LLM returns narration plus a small JSON block of directives.
3. The engine parses the JSON and executes tools (state updates and image requests).
4. Narration displays immediately; images are reused from cache or generated fresh.
5. Game state (map, inventory, notes, images) persists across turns.

### The LLM's central role
- The LLM decides the story beats and calls tools by emitting JSON directives appended after the narration.
- A static "World Bible" provides setting, items, NPCs, and overall constraints for consistency.
- Better local models improve reliability: cleaner JSON, fewer truncations, smarter tool use.
- The LLM follows storytelling principles: tension building, multiple solutions, fail-forward design, and narrative callbacks.

---

## Architecture: GameState + World Bible

### What the Turn-LLM sees each turn

The Turn-LLM receives a **carefully curated subset** of information, not the entire World Bible or full game state. This keeps the LLM fast and focused.

**From GameState (dynamic, changes every turn):**
- Player location, health, inventory
- Known map (trimmed to current + adjacent rooms)
- Recent conversation (last 2 turns)
- Story context (LLM's narrative memory)
- Game flags (puzzle states, conditions)
- Current room items and exits

**From World Bible (static blueprint, only current-location excerpts):**
- NPCs, monsters, and riddles at this location
- Available items and mechanics
- Current objective and hints

**What the LLM NEVER sees:** other rooms' content, the full item catalog, undiscovered locations, engine internals.

### GameState = What IS

Current reality: where you are, what you have, what you've done. Dynamic and changes every turn.

### World Bible = What COULD BE

The game's potential: NPCs to meet, puzzles to solve, items to find. Static blueprint created once at game start.

### Dynamic world expansion

The turn LLM has **complete creative authority** to expand beyond the World Bible:
- Can create ANY new room via `move_to` + `connect`
- Can place ANY new item via `place_items`
- Can invent new NPCs, puzzles, secrets on the fly

Every playthrough can be unique. The World Bible provides narrative spine; the LLM provides magic and surprise.

---

## JSON Directives (Tool Calls)

The LLM appends JSON after its narration:

```json
{
  "state_updates": {
    "move_to": "Hidden Chamber",
    "connect": [["Hidden Chamber", "Gloomy Cavern"]],
    "place_items": ["Crystal Key"],
    "room_take": ["Crystal Key"],
    "add_note": "Dust cloud hides a small key.",
    "set_context": "Player discovered the key puzzle.",
    "change_health": -5
  },
  "images": ["Crystal Key close-up", "Hidden Chamber overview"]
}
```

### Available Directives

| Directive | Description |
|-----------|-------------|
| `move_to` | Move player to a room |
| `connect` | Link rooms bidirectionally |
| `place_items` | Put items in current room |
| `room_take` | Player picks up item from room |
| `add_items` | Give item to player directly |
| `remove_items` | Consume/destroy item from inventory |
| `change_health` | Damage or heal player |
| `set_context` | LLM's narrative memory |
| `set_flag` | Track puzzle/event states |
| `add_note` | Add to quest log |
| `images` | Generate scene artwork |

Advanced directives (for larger models): `timer_event`, `conditional_action`, `chain_reaction`.

---

## Model Selection

### LLM Models (~/MLX_Models/)

The Gradio UI auto-discovers MLX models. Recommended choices by RAM:

| RAM | Model | Notes |
|-----|-------|-------|
| 16GB | Qwen3-4B-MLX | Basic gameplay |
| 32GB | Qwen3-30B-A3B-8bit | Good balance of quality and speed |
| 64GB | Qwen2.5-32B-Instruct-8bit | Rich narrative |

### Image Models (~/Diffusion_Models/)

| Model | Steps | Quality | Speed |
|-------|-------|---------|-------|
| FLUX.2-klein-9B-mlx-8bit | 4 | Good | Fast |
| FLUX.1-dev | 20 | Best | Slower |

---

## How the Game Improves with Better LLMs

The game automatically gets better as local LLMs improve, without any code changes:

**Current LLMs (~35B):** Basic room connections, item management, simple NPCs

**Better LLMs (~70B+):** Complex multi-room puzzles, NPCs with personality, foreshadowing

**Future LLMs:** Dynamic story arcs, NPC relationships, emergent mechanics

The key: all improvements come from the LLM better using the same simple tools.

---

## Files

| File | Description |
|------|-------------|
| `LMM_adventure_Feb_15_26.py` | Main game engine (MLX-LM + MFLUX + Gradio) |
| [`../browser_adventure/`](../browser_adventure/) | **Browser edition** — game + detailed docs ([`adventure.html`](../browser_adventure/adventure.html), [`README.md`](../browser_adventure/README.md)) |
| `llm_adventure/adventure.html` | Redirect to `../browser_adventure/adventure.html` (keep old bookmarks working) |
| Root `adventure.html` | Redirect stub when `http.server` runs from repo root (→ `browser_adventure/adventure.html`) |
| `mflux_image_gen.py` | FLUX image generation wrapper (Apple Silicon) |
| `diffusers-webgpu-compare-test.html` | Browser-only T2I compare (`web-txt2img` + ONNX Runtime Web / WebGPU; not Python diffusers) |
| `vendor/web-txt2img/` | Vendored `web-txt2img@0.3.1` `dist/` (same-origin workers for localhost) |
| `transformersjs-compare-test.html` | Browser ONNX LLM compare (Transformers.js) |
| `deprecated_diffuser_server/` | Old diffusion server approach (no longer used) |

---

## Browser Edition (`browser_adventure/`)

**Full documentation:** [`../browser_adventure/README.md`](../browser_adventure/README.md) (how narration is generated, world bible vs. LLM, JSON tools, customizing stories, serving requirements).

**[Play on GitHub Pages](https://jmrothberg.github.io/Collosol-Cave-with-local-LLM/browser_adventure/adventure.html)** — no install; first visit downloads models (~4 GB total, then cached). [Short URL (root stub)](https://jmrothberg.github.io/Collosol-Cave-with-local-LLM/adventure.html) redirects into `browser_adventure/`.

The game lives in **`browser_adventure/adventure.html`** (repo root folder). It loads `web-txt2img` from **`llm_adventure/vendor/`**, so you must serve from the **repository root**, not from `llm_adventure/` alone:

```bash
cd Colossal_Cave    # repo root
python3 -m http.server 8080
# http://localhost:8080/browser_adventure/adventure.html
```

Legacy paths **`/llm_adventure/adventure.html`** and **`/adventure.html`** redirect to the new location when using a root server.

**Models:**
- **LLM:** Gemma 4 E4B (`onnx-community/gemma-4-E4B-it-ONNX`) via Transformers.js + ONNX Runtime Web / WebGPU
- **Images:** Stable Diffusion 1.5 (`sd-1.5`) via `web-txt2img` worker + ONNX Runtime Web / WebGPU

**Requirements:**
- **WebGPU** (Chrome/Edge 113+) strongly recommended; WASM fallback available but slower
- **First run** downloads ~4 GB of model weights (browser-cached after first load)
- **VRAM:** Gemma 4B + SD 1.5 run concurrently (LLM on main thread, image gen in Web Worker)

**Built-in cave adventure:** Default world bible in `adventure.html` (classic cave: hermit, troll, temple, treasure). Click "Start Adventure" after models load.

---

## Browser WebGPU text-to-image compare (test harness)

**[Try it live on GitHub Pages](https://jmrothberg.github.io/Collosol-Cave-with-local-LLM/llm_adventure/diffusers-webgpu-compare-test.html)** — no install, no server, just click and run.

The file `diffusers-webgpu-compare-test.html` runs **two models side by side in the browser** using [`web-txt2img`](https://www.npmjs.com/package/web-txt2img) (workers + **ONNX Runtime Web**, WebGPU when available). The library’s `dist/` is **vendored** under `llm_adventure/vendor/web-txt2img/` so the worker script is **same-origin** as the page (browsers reject `Worker` scripts on a CDN when the page is `localhost`). It is **not** the Python Hugging Face `diffusers` stack; it is a separate, client-only ONNX path aimed at fast browser models (`sd-turbo`, `janus-pro-1b`, `sd-1.5` multi-step).

**How to open it**

- Serve over **http/https** (not `file://`). **Option A:** `cd llm_adventure` then `python3 -m http.server 8080` and open `http://localhost:8080/diffusers-webgpu-compare-test.html`. **Option B:** run `python3 -m http.server 8080` from the **repo root** and open `http://localhost:8080/llm_adventure/diffusers-webgpu-compare-test.html`, or `http://localhost:8080/diffusers-webgpu-compare-test.html` (root stub redirects into `llm_adventure/`). Workers, WASM, and Hub downloads need a normal origin.

**Requirements**

- **WebGPU** strongly recommended (Janus-Pro is WebGPU-only; SD-Turbo and SD 1.5 can try WASM fallback). Chrome/Edge 113+ is a practical baseline.
- **First run** downloads large weights per model (~2–2.3GB each, browser-cached). Two **separate** workers are used so A and B can differ; holding both loaded may exceed VRAM — use **Unload** on one side if needed.
- **SD 1.5 multi-step** (25 steps, CFG 7.5) produces much higher quality images than the single-step turbo models (~2GB, FP16 ONNX from `microsoft/stable-diffusion-v1.5-webnn`).

**Extending models**

- The page lists ids from the `web-txt2img` registry. When the library adds entries, append them to the `MODEL_CATALOG` array in the HTML.

**Updating the vendored library**

- To bump `web-txt2img`, from `llm_adventure`: `rm -rf vendor/web-txt2img && mkdir -p vendor && (cd vendor && npm pack web-txt2img@X.Y.Z && tar -xzf web-txt2img-X.Y.Z.tgz package/dist && mv package/dist web-txt2img && rm -rf package web-txt2img-X.Y.Z.tgz)`.

---

## Tips

1. **Start with a good LLM:** Qwen3-30B-A3B-8bit is an excellent balance of quality and speed
2. **Generate a World Bible first:** This creates consistent story, NPCs, and objectives
3. **Use themes:** Preset themes (Tolkien, Star Wars, Cyberpunk, etc.) provide better artwork and narrative
4. **Save often:** Use the save game feature before dangerous encounters
5. **Keep commands specific:** Short, clear actions get better LLM responses

---

## Troubleshooting

**"No MLX models found"**
- Place MLX model directories in `~/MLX_Models/`
- Each model directory must contain a `config.json`

**"Model not found" (images)**
- Check that FLUX models are in `~/Diffusion_Models/`
- Directory names should start with `FLUX`

**Images are slow**
- Use FLUX.2-klein-9B (4 steps) instead of FLUX.1-dev (20 steps)

---

## License

MIT License
