#!/bin/bash
# Start DiffusionD server

echo "🎨 Starting DiffusionD Server..."
echo "   Make sure your Diffusion_Models directory exists with models"
echo "   Server will be available at http://127.0.0.1:8000"
echo ""

# Parse GPU argument
GPU_ARG=""
if [ "$1" = "--gpu" ] && [ -n "$2" ]; then
    GPU_ARG="--gpu $2"
    echo "   Using GPU: $2"
fi

# Activate virtual environment
source .venv/bin/activate

# Set environment variable for models directory
export DIFFUSION_MODELS_DIR="/data/Diffusion_Models"

# Set PyTorch CUDA memory configuration to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Start the server on GPU 1 to avoid memory fragmentation
python3 diffusiond.py --host 127.0.0.1 --port 8000 --gpu 1
