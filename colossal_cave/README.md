# Colossal Cave Classic

Text adventure with 25+ rooms (from `adventure_dataRA.json`). **Ollama** parses commands and drives NPC chat. **Diffusers** generate room/item art from local models. **Pyramid Flow** (optional) generates room videos.

**Run:** `python Colossal_Cave_Aug_2_25.py` → Gradio UI at `http://127.0.0.1:7860`

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
cd colossal_cave && python Colossal_Cave_Aug_2_25.py
```

On first launch, pick a **models folder** (parent of `Diffusion_Models/`) and a **data folder** (defaults to this directory).

---

## Local models (images)

The diffusion dropdown lists **only folders** under `Diffusion_Models/` — no remote Hugging Face IDs.

| Model | Mac speed | Notes |
|-------|-----------|--------|
| **FLUX2-klein-9B-mlx-8bit** | Fastest | MLX via mflux (`mlx` in folder name). Download: `huggingface-cli download AITRADER/FLUX2-klein-9B-mlx-8bit --local-dir ~/Diffusion_Models/FLUX2-klein-9B-mlx-8bit` |
| **Z-Image-Turbo** | Medium | 9 steps, guidance 0. Needs `diffusers>=0.36` |
| **FLUX.1-dev** | Slow | Full diffusers FLUX on MPS; 4 steps |
| Qwen-Image / Qwen-Image-Edit / FLUX.1-Fill-dev | Slow / special | Edit/inpaint models; not tuned for adventure art |

**Mac paths:** looks for `Diffusion_Models/` under `/Users/jonathanrothberg` or `/Users/jonathanrothberg_1`, or the folder you choose at startup.

**HF token:** not required for local models. Set `HF_TOKEN` only if downloading new weights.

---

## Video (Pyramid only)

The main game uses **Pyramid Flow** (`pyramid-flow-miniflux`), not LTX. LTX is a separate CUDA test script (`vidiofromtext_ltx_Jan_7_25.py`).

**Mac:** enable video in the UI; needs PyTorch **2.9.1+ stable** (Conv3D MPS), FFmpeg (`brew install ffmpeg`), and the model at `./pyramid-flow-miniflux` (repo root). Download once:

```bash
python miniflux_download.py   # or huggingface-cli download rain1011/pyramid-flow-miniflux --local-dir ../pyramid-flow-miniflux
```

Details: `README_pyramid_MPS_Video_Generation.md`

---

## Dependencies

Core pins (`requirements.txt`):

- `torch>=2.9.1` + `torchvision>=0.24.1` — install **together** from stable PyPI (not `--pre` nightly)
- `diffusers>=0.36.0` (Z-Image support)
- `transformers>=4.44,<5.0` and `huggingface-hub>=0.34,<1.0` (mflux / sentence-transformers compatibility)

Upgrade ML stack:

```bash
pip install -U torch torchvision diffusers "transformers>=4.44,<5.0" accelerate safetensors "huggingface-hub>=0.34,<1.0"
```

---

## Requirements

- **Python** 3.10 or 3.11 (Pyramid has issues on 3.12)
- **Mac:** Apple Silicon + 16 GB+ RAM; **Linux:** NVIDIA GPU for best performance
- **Ollama** running (`ollama serve`)

---

## Architecture (brief)

```
command → engine/preprocess.py → engine/llm_commands.py (Ollama JSON)
       → execute_action() → Gradio UI
       → Diffusers / mflux (images) → Pyramid Flow (optional video)
```

Extracted packages: `engine/`, `ui/`, `media/`, `config/`, `tests/`

---

## Testing

```bash
cd colossal_cave
COLOSSAL_CAVE_SKIP_VECTOR_MODEL=1 COLOSSAL_CAVE_HEADLESS=1 pytest -m "not gpu and not ollama"
```

- **CI-safe suite:** engine helpers, fast-path, execute_action, save/load, walkthrough simulator (~65+ tests)
- **Optional local:** `pytest -m ollama` (live Ollama), `pytest -m gpu` (CUDA/MPS smoke)
- Env vars: `COLOSSAL_CAVE_SKIP_VECTOR_MODEL=1` skips unused SentenceTransformer load; `COLOSSAL_CAVE_HEADLESS=1` skips interactive video prompts on load

CI runs on push via `.github/workflows/colossal_cave_tests.yml`.

---

## Game commands (summary)

| Command | Action |
|---------|--------|
| `go n/s/e/w` | Move |
| `take` / `leave` / `use` / `wear` | Items |
| `attack <monster> with <weapon>` | Combat |
| `talk <msg>` / `MB, …` | NPC / Master Beast helper |
| `solve <magic item>` | Riddles |
| `inventory` / `status` / `map` / `help` | Info |

MB (Master Beast) in Room 1 knows the full game state — ask her anything.

---

## Key files

| File | Purpose |
|------|---------|
| `Colossal_Cave_Aug_2_25.py` | Main game |
| `adventure_dataRA.json` | World data |
| `requirements.txt` / `colossal_requirements.txt` | Dependencies |
| `miniflux_download.py` | Pyramid video model downloader |
| `Z_image_TestCode.py`, `mini_flux_*.py`, `vidiofromtext_*.py` | Standalone model tests (not imported by main game) |

---

## Troubleshooting

**Ollama empty / not responding** — run `ollama serve` in another terminal.

**`ZImagePipeline` missing** — `pip install -U "diffusers>=0.36.0"`

**`torchvision::nms does not exist`** — torch/torchvision mismatch; upgrade both from stable PyPI (see Dependencies).

**Z-Image very slow** — confirm log shows `Using Z-Image-Turbo settings: 9 steps, 0.0 guidance`. For fastest Mac images, use **FLUX2-klein-9B-mlx-8bit**.

**Hugging Face token prompt** — not needed for local models; ignore or set `HF_TOKEN` only for downloads.

**Video fails on Mac** — PyTorch 2.9.1+ stable, FFmpeg installed, `pyramid-flow-miniflux` present, see pyramid README.

---

## License

MIT License
