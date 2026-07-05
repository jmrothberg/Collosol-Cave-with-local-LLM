#!/usr/bin/env bash
# Colossal Cave — first-time setup and launch (macOS / Linux)
# Usage:
#   ./scripts/startup.sh          # install deps + run game
#   ./scripts/startup.sh --setup  # install only
#   ./scripts/startup.sh --test   # install + run CI-safe tests

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
GAME_DIR="$REPO_ROOT/colossal_cave"
MLX_DIR="${COLOSSAL_CAVE_MLX_MODELS_DIR:-$HOME/MLX_Models}"

SETUP_ONLY=0
RUN_TESTS=0
for arg in "$@"; do
  case "$arg" in
    --setup) SETUP_ONLY=1 ;;
    --test) RUN_TESTS=1 ;;
    -h|--help)
      echo "Usage: $0 [--setup | --test]"
      echo "  (no args)  install/update deps and start Colossal Cave"
      echo "  --setup    install/update deps only"
      echo "  --test     install/update deps and run pytest (CI-safe)"
      exit 0
      ;;
  esac
done

echo "=== Colossal Cave setup ==="
echo "Repo:      $REPO_ROOT"
echo "Platform:  $(uname -s) $(uname -m)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3.11+ first."
  exit 1
fi

PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo "Python:    $PY_VER"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

echo "Installing core requirements..."
pip install -r "$GAME_DIR/requirements.txt"

# Apple Silicon: MLX text LLMs from ~/MLX_Models (primary path; Ollama optional)
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  echo "Apple Silicon detected — ensuring mlx-vlm>=0.6.1 (gemma4_unified) and mlx-lm..."
  pip install --upgrade "mlx-vlm>=0.6.1" "mlx-lm>=0.19.0"
  mkdir -p "$MLX_DIR"
  echo "MLX models dir: $MLX_DIR"
  echo "  Put converted weights here, e.g. $MLX_DIR/gemma-4-12B-it-8bit/"
else
  echo "Non-Apple-Silicon host — MLX text LLMs need Apple Silicon."
  echo "  Use ollama:<tag> in the game dropdown, or run on a Mac with ~/MLX_Models."
fi

if [[ "${INSTALL_VIDEO:-0}" == "1" ]]; then
  echo "INSTALL_VIDEO=1 — installing optional video/MFLUX stack..."
  pip install -r "$GAME_DIR/colossal_requirements.txt"
fi

if [[ "$RUN_TESTS" == "1" ]]; then
  echo "Running CI-safe tests..."
  cd "$GAME_DIR"
  COLOSSAL_CAVE_SKIP_VECTOR_MODEL=1 COLOSSAL_CAVE_HEADLESS=1 \
    pytest -m "not gpu and not ollama and not mlx" -q
  echo "Tests passed."
  exit 0
fi

if [[ "$SETUP_ONLY" == "1" ]]; then
  cat <<EOF

Setup complete.

Next steps:
  1. Put MLX weights in: $MLX_DIR/<folder-name>/  (config.json + safetensors)
     Or (optional): ollama serve + pick ollama:<tag> in the UI
  2. Put diffusion models in: ~/Diffusion_Models/  (or choose folder at launch)
  3. Run:  source $VENV_DIR/bin/activate && cd $GAME_DIR && python colossal_cave_July_5_26.py

Env overrides (optional):
  COLOSSAL_CAVE_MLX_MODELS_DIR   — MLX weights folder (default ~/MLX_Models)
  COLOSSAL_CAVE_PLAY_LLM         — default play model name (default gemma-4-12B-it-8bit)
  INSTALL_VIDEO=1 ./scripts/startup.sh --setup  — add Pyramid/MFLUX extras

EOF
  exit 0
fi

echo "Starting Colossal Cave..."
cd "$GAME_DIR"
exec python colossal_cave_July_5_26.py
