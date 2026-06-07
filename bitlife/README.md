# BitLife clone — local-AI life simulator

A faithful clone of **BitLife** (Candywriter, LLC) that runs entirely in the browser. The game engine
is **premade and deterministic** — you live a life by clicking **Age Up** and menu buttons — and it
works **instantly with no AI loaded**. The only additions are local, in-browser AI for three things:

1. **Understanding** — a local LLM (Gemma 4 E4B) interprets anything you **type** into game effects.
2. **Greater description** — the same LLM writes richer narration.
3. **Illustration** — a local diffuser (Stable Diffusion 1.5) paints your avatar and life-event scenes.

Everything else mirrors the real game: stats, yearly random events, Occupation (school + jobs),
Activities (mind & body, doctor, crime, casino), relationships, an investments market with **insider
trading** (the "Martha" ribbon), real estate, pets, fame, achievements, save slots, and seeded
reproducible lives.

## Run

```bash
python3 scripts/serve-threaded.py 8080
# open http://localhost:8080/bitlife/bitlife.html   (Chrome/Edge 113+ with WebGPU)
```

- Tick **"Skip loading AI models"** to play instantly with no downloads (deterministic game only).
- First AI load is ~5 GB (Gemma + SD 1.5), cached afterward.
- Must be served over http (not `file://`) and from the repo root so the image worker resolves.
- Optional: tick **"Use local Ollama"** + a model name to route typed actions through Ollama.

## Files

```
bitlife.html        # the entire game (engine + UI + LLM + image manager)
bitlife_data.json   # premade content (events, activities, careers, market, tips, achievements)
pregen_art.py       # OPTIONAL GPU baker -> assets/ + manifest.json (pre-bakes scene art)
assets/             # optional pre-baked PNGs (manifest.json maps scene keys -> files)
background.md       # developer guide: how to add features, debug, and reach BitLife parity
```

## Developing

Read **[background.md](./background.md)** — it covers how real BitLife works (with sources), the code
architecture, **how to add features** (events/activities/careers/assets/achievements/images), how to
debug, and the roadmap to feature parity. Most content growth is just adding data to
`bitlife_data.json`.

---
*v1.0 — Jonathan Rothberg, 2026. Local-AI stack reused from this repo's `browser_adventure` / `llm_adventure`.*
