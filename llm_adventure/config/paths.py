"""Filesystem paths for llm_adventure."""

import os

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(SCRIPT_DIR, "Adventure_Game_Saved")
ART_DIR = os.path.join(SCRIPT_DIR, "Adventure_Art")
IN_GAME_IMG_DIR = os.path.join(SAVE_DIR, "in_game_images")

# MLX model discovery (override via env if needed)
MLX_MODELS_DIR = os.environ.get(
    "MLX_MODELS_DIR",
    os.path.expanduser("~/MLX_Models"),
)
