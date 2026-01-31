# DiffusionD - Ollama-like Diffusion Model Server

DiffusionD is a centralized server for serving multiple diffusion models, similar to how Ollama serves LLMs. This allows any application to use diffusion models without handling the complexity of model loading, GPU management, and different architectures.

## Architecture

- **Server**: `diffusiond.py` - FastAPI-based REST server
- **Client**: `diffusion_client.py` - Python client library for applications
- **Models**: Automatically discovered from `Diffusion_Models/` directory
- **API**: Simple REST endpoints for model management and image generation

## Supported Model Types

- **FLUX** models (Black Forest Labs)
- **Stable Diffusion XL** (SDXL)
- **Stable Diffusion 3** (SD3)
- **Z-Image-Turbo** (special optimized model)
- **Generic Diffusion** pipelines

## Quick Start

### 1. Start the Server

```bash
# Using the startup script
./start_diffusion_server.sh

# Or directly
python3 diffusiond.py --host 127.0.0.1 --port 8000
```

### 2. Use in Your Application

```python
from diffusion_client import DiffusionClient

# Create client
client = DiffusionClient(model="Z-Image-Turbo")

# Generate image
image_path = client.generate("a beautiful mountain landscape")
if image_path:
    print(f"Image saved to: {image_path}")
```

### 3. List Available Models

```python
from diffusion_client import list_available_models

models = list_available_models()
print("Available models:", models)
```

## API Endpoints

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-12-03T10:30:00"
}
```

### GET /models
List available models.

**Response:**
```json
{
  "models": [
    {
      "name": "Z-Image-Turbo",
      "type": "zimage",
      "loaded": true,
      "path": "/data/Diffusion_Models/Z-Image-Turbo"
    }
  ]
}
```

### POST /generate
Generate an image.

**Request:**
```json
{
  "model": "Z-Image-Turbo",
  "prompt": "a beautiful sunset",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 9,
  "guidance_scale": 0.0
}
```

**Response:**
```json
{
  "image_path": "Generated_Art/a_beautiful_sunset_20241203_103000.png"
}
```

### POST /unload/{model_name}
Unload a model from memory.

## Model Directory Structure

Place your models in the `Diffusion_Models` directory:

```
/data/Diffusion_Models/
├── Z-Image-Turbo/
├── flux-dev/
├── stable-diffusion-xl/
└── ...
```

The server automatically detects model types based on:
- Directory names (FLUX, SDXL, SD3)
- Model configuration files
- Special handling for known models

## Environment Variables

- `DIFFUSION_MODELS_DIR`: Path to models directory (default: `/data/Diffusion_Models`)

## Systemd Service

To run as a system service:

```bash
# Copy service file
sudo cp diffusiond.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start
sudo systemctl enable diffusiond
sudo systemctl start diffusiond

# Check status
sudo systemctl status diffusiond
```

## Benefits

1. **Centralized Management**: All model loading logic in one place
2. **Memory Efficiency**: Models cached and managed centrally
3. **Simple API**: Applications just call `generate(prompt)`
4. **Multiple Architectures**: Supports FLUX, SDXL, SD3, Z-Image, etc.
5. **Device Agnostic**: Automatically handles CUDA/MPS/CPU
6. **Backwards Compatible**: Existing code works with minimal changes

## Migration from Local Generation

Your existing adventure game automatically uses the server if available. If the server is down, it falls back to local generation.

```python
# Old way (still works)
image_gen = ImageGenerator(model_id="Z-Image-Turbo")

# New way (automatic)
# Code detects server and uses DiffusionClient if available
```

## Troubleshooting

### Server Won't Start
- Check that `Diffusion_Models` directory exists
- Verify model directories contain valid diffusion models
- Check GPU memory availability for large models

### Client Connection Failed
- Ensure server is running: `curl http://127.0.0.1:8000/health`
- Check firewall settings
- Verify client and server are on same network

### Model Not Found
- Check model name spelling
- Verify model directory exists in `Diffusion_Models`
- Run `curl http://127.0.0.1:8000/models` to list available models

## Performance Tips

- Models are loaded on-demand and cached in memory
- Large models (FLUX, SDXL) may take time to load initially
- Use `/unload/{model}` to free GPU memory when switching models
- Server automatically cleans up old generated images (keeps last 100)
