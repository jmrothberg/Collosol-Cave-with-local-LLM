# Adventure Games with Local LLMs

Two text-adventure games inspired by the classic "Colossal Cave Adventure," each taking a fundamentally different approach to using AI.

**Author:** Jonathan M. Rothberg ([@jmrothberg](https://github.com/jmrothberg))

---

## Browser AI Demo (no install needed)

**[Run WebGPU Text-to-Image Compare in your browser](https://jmrothberg.github.io/Collosol-Cave-with-local-LLM/llm_adventure/diffusers-webgpu-compare-test.html)** — compare SD-Turbo, Janus-Pro-1B, and SD 1.5 multi-step side by side. Runs entirely in-browser via ONNX Runtime Web + WebGPU. Requires Chrome/Edge 113+ with WebGPU. First run downloads ~2 GB per model (cached).

---

## Two Games, Two Philosophies

### [`colossal_cave/`](colossal_cave/) — Code-Driven Adventure

A traditional text adventure where **the code drives the gameplay**. The game has 25+ hand-designed rooms, monsters, treasures, riddles, and NPCs loaded from a JSON data file. LLMs play a supporting role: Ollama parses natural-language commands and generates NPC dialogue, while Diffusers (FLUX, Stable Diffusion 3.5, SDXL) create artwork and Pyramid Flow generates room videos.

- Structured, predictable gameplay with rich pre-built content
- Works on Apple Silicon Mac (MPS) or Linux with NVIDIA GPU (CUDA)
- Multi-GPU support for distributing LLM, diffusion, and video across GPUs
- See [`colossal_cave/README.md`](colossal_cave/README.md) for setup and details

### [`llm_adventure/`](llm_adventure/) — LLM-Driven Adventure

A procedural adventure where **the LLM is the game master**. Instead of pre-defined rooms, the AI dynamically creates the entire world -- rooms, NPCs, items, puzzles, and narrative -- guided by a "World Bible" theme. The engine sends the LLM a curated game state each turn; the LLM responds with narration plus JSON directives (move_to, room_take, place_items, etc.) that the engine executes to update the game.

- Every playthrough is unique -- the LLM creates content on the fly
- As local LLMs improve (better JSON, smarter tool use), the game automatically gets better with no code changes
- Runs natively on Apple Silicon via MLX-LM (language) and MFLUX (images)
- See [`llm_adventure/README.md`](llm_adventure/README.md) for setup and details

---

## Quick Start

```bash
git clone https://github.com/jmrothberg/Collosol-Cave-with-local-LLM.git
cd Collosol-Cave-with-local-LLM
python3 -m venv .venv
source .venv/bin/activate
```

Then follow the README in whichever game folder you want to play.

---

## Project Structure

```
.
├── README.md                          # This file
├── colossal_cave/                     # Game 1: code-driven + LLM support
│   ├── README.md
│   ├── Colossal_Cave_Aug_2_25.py      # Main game (Ollama + Diffusers + Pyramid Flow)
│   ├── adventure_dataRA.json          # 25+ rooms, NPCs, monsters, riddles
│   ├── diffusion_manager.py           # Image generation interface
│   ├── complete_instruction.py        # Help system
│   └── ...                            # Video gen tools, downloaders, utilities
│
├── llm_adventure/                     # Game 2: LLM as game master
│   ├── README.md
│   ├── LMM_adventure_Feb_15_26.py     # Main game (MLX-LM + MFLUX)
│   ├── mflux_image_gen.py             # FLUX image generation (Apple Silicon)
│   └── deprecated_diffuser_server/    # Old diffusion server approach
│
├── .env.example
└── .gitignore
```

---

## License

MIT License

---

## Acknowledgments

- Inspired by the original "Colossal Cave Adventure" by Will Crowther and Don Woods
- [MLX-LM](https://github.com/ml-explore/mlx-lm), [MFLUX](https://github.com/filipstrand/mflux), [Ollama](https://ollama.com), [Diffusers](https://github.com/huggingface/diffusers), [Pyramid Flow](https://github.com/jy0205/Pyramid-Flow), [Gradio](https://gradio.app)

---

## Contact

- GitHub: [@jmrothberg](https://github.com/jmrothberg)
- Project: [Colossal-Cave-with-local-LLM](https://github.com/jmrothberg/Collosol-Cave-with-local-LLM)
