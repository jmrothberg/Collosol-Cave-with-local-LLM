# JMR's BitLife — local-LLM life simulator

A **BitLife-style life simulator** that runs entirely in your browser, built on the same
local-AI stack as this repo's `browser_adventure`:

- **Premade, deterministic game engine** — you live a life by clicking **Age Up** and menu
  buttons (Activities, Careers, Crime, Casino, Relationships, Assets/Investments). This works
  **instantly, with no AI models loaded at all**.
- **Type anything** — free-typed actions are interpreted by a small **local LLM**
  (Gemma 4 E4B via Transformers.js / WebGPU) into bounded, sanitized game effects + narration.
- **Local diffuser graphics** — **Stable Diffusion 1.5** (the vendored `web-txt2img` WebGPU
  worker) paints your **character avatar** (regenerated as you age) and **life-event scenes**
  (birth, graduation, wedding, prison, death, …).
- **Stock market + insider trading** — buy/sell stocks, crypto and bonds whose prices random-walk
  each year; act on insider tips for big gains, but the SEC may investigate and send you to prison
  (the "Martha" achievement). Plus real estate, pets, fame, achievements, and multiple save slots.

## Run it

The image worker is loaded by relative path from `../llm_adventure/vendor/web-txt2img/`, and the
ONNX worker only resolves when the page is **served from the repo root**. Use the included server
(it also sends the COOP/COEP headers WebGPU/threaded-WASM need):

```bash
python3 scripts/serve-threaded.py 8080
# then open:
http://localhost:8080/bitlife/bitlife.html
```

- **Browser:** Chrome/Edge 113+ with **WebGPU** strongly recommended (WASM fallback works but is slow).
- **First load:** downloads ~5 GB of models (Gemma + SD 1.5), cached afterwards. This cache is
  **shared with `browser_adventure`** — if you've played that, the models are already cached.
- **No-AI mode:** tick *"Skip loading AI models"* on the start screen to play the deterministic
  game only (no typed AI, no images) — great for a fast first look.
- **Ollama option:** tick *"Use local Ollama"* and enter a model name to route typed actions
  through a local Ollama server instead of in-browser Gemma. (Set `OLLAMA_ORIGINS` so the browser
  can reach it.)

## Files

| File | What it is |
|------|------------|
| `bitlife.html` | The entire self-contained game (engine + UI + LLM/image integration). |
| `bitlife_data.json` | Premade content: yearly events, activities, careers, market catalog, insider tips, names, achievements. A minimal copy is embedded in the HTML as `FALLBACK_DATA` so the game still runs if this file can't be fetched. |
| `pregen_art.py` | **Optional** GPU batch script that pre-bakes life-event scene art into `assets/` + `manifest.json` ahead of time. |
| `assets/` + `manifest.json` | Optional pre-generated PNGs the game uses **instantly** when present. |

## How the art is "made in advance"

The game resolves every image with this priority — **static asset → IndexedDB cache → live generate**:

1. **Static assets (instant):** if `assets/manifest.json` exists, matching scenes load immediately
   with no generation. Create it with `python3 bitlife/pregen_art.py` on a machine with a GPU.
2. **Persistent cache:** everything generated in-browser is stored in **IndexedDB**, so replays and
   later sessions reuse art (second run is instant).
3. **Idle background pre-render:** once SD 1.5 finishes loading, the game pre-renders the current +
   next life-stage avatar and the common life-event scenes during idle time, so they're ready before
   you reach them — without blocking gameplay.

## How typed actions stay safe

Typed actions go to the LLM, which must return a small JSON directive
(`{narration, effects, relationship_changes, set_flags, grant_achievement, event_image}`). All
proposed effects are clamped (`sanitizeLlmEffects` → `applyEffects` → 0–100 stat clamp), so the AI
can flavor the story but can't break the game. A parse failure just shows narration; gameplay never
blocks on the model.

## Reproducible lives

All engine randomness flows through a seeded `mulberry32` RNG. Enter a **seed** on the start screen
(or read it from ☰ Menu) to replay the exact same life.

---
*v1.0 — Jonathan Rothberg, 2026. Engine & vendor image worker reused from `browser_adventure` / `llm_adventure`.*
