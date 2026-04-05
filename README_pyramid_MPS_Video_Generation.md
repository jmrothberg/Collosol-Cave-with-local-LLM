# Mac MPS Video Generation with Pyramid-Flow

This guide documents how to successfully run Pyramid-Flow video generation on Mac with MPS (Metal Performance Shaders) acceleration, including all the compatibility fixes needed.

## 🚀 Quick Start

```bash
git clone https://github.com/jy0205/Pyramid-Flow.git
pip install -r Pyramid-Flow/requirements.txt
pip install huggingface_hub psutil
brew install ffmpeg
python mac_pyramid.py
```

## 📋 System Requirements

- **macOS**: 13.2+ (Ventura or later)
- **Apple Silicon**: M1/M2/M3 Mac with MPS support
- **Memory**: 16GB+ RAM (96GB+ recommended for high quality)
- **PyTorch**: 2.9.0+ with Conv3D MPS support

## 🔧 Critical PyTorch Setup

### 1. Install Latest PyTorch with Conv3D Support

```bash
# CRITICAL: Conv3D support for MPS was only added in late 2023
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

**Why this matters**: Earlier PyTorch versions (< 2.9.0) don't support Conv3D operations on MPS, which are essential for video generation models.

### 2. Verify MPS and Conv3D Support

```python
import torch
print("MPS Available:", torch.backends.mps.is_available())
print("MPS Built:", torch.backends.mps.is_built())
print("PyTorch Version:", torch.__version__)
```

## 🛠️ Key Compatibility Fixes

### 1. Float64 → Float32 Conversion for MPS

**Problem**: MPS doesn't support float64 operations, but many PyTorch functions default to float64.

**Solution**: Monkey patch key functions:

```python
if torch.backends.mps.is_available():
    # Set default dtype
    torch.set_default_dtype(torch.float32)
    
    # Patch torch.from_numpy()
    original_from_numpy = torch.from_numpy
    def mps_safe_from_numpy(ndarray):
        tensor = original_from_numpy(ndarray)
        if tensor.dtype == torch.float64:
            tensor = tensor.float()
        return tensor
    torch.from_numpy = mps_safe_from_numpy
    
    # Patch torch.arange()
    original_arange = torch.arange
    def mps_safe_arange(*args, **kwargs):
        if 'dtype' in kwargs and kwargs['dtype'] == torch.float64:
            kwargs['dtype'] = torch.float32
        return original_arange(*args, **kwargs)
    torch.arange = mps_safe_arange
```

### 2. MPS Memory Management

```python
# For systems with large shared memory (64GB+)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# For systems with limited memory, try:
# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.8"
```

### 3. Autocast Compatibility

**Problem**: MPS doesn't support `torch.autocast()` like CUDA.

**Solution**: Use conditional context managers:

```python
if torch.backends.mps.is_available():
    context_manager = torch.no_grad()  # MPS: no autocast
else:
    context_manager = torch.autocast(device_type="cpu", enabled=True, dtype=torch_dtype)
```

## 📦 Dependencies Installation

### Core Dependencies

```bash
pip install timm>=0.9.0  # CRITICAL: 0.6.12 has Python 3.11+ compatibility issues
pip install ipython      # Required by Pyramid-Flow debugging
pip install huggingface_hub
pip install psutil       # For memory monitoring
brew install ffmpeg      # For MP4 video export
```

### Python Path Configuration

```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Pyramid-Flow'))
```

## 🎬 Model Configuration

### Resolution and Model Variants

```python
# High Quality (768p) - Requires more memory
width, height = 1280, 768
model_variant = 'diffusion_transformer_768p'

# Medium Quality (384p) - More memory efficient  
width, height = 640, 384
model_variant = 'diffusion_transformer_384p'
```

### Generation Parameters

```python
# High Quality Settings (for 64GB+ systems)
num_inference_steps = [10, 10, 10]  # Higher quality
temp = 16                           # Longer video (16 frames)
save_memory = False                 # Faster VAE decoding

# Fast/Testing Settings
num_inference_steps = [5, 5, 5]     # Faster generation
temp = 8                            # Shorter video (8 frames)  
save_memory = True                  # Memory efficient
```

## 🎯 Complete Working Example

```python
import sys
import os
import torch
from PIL import Image
from datetime import datetime

# Add Pyramid-Flow to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Pyramid-Flow'))

# Auto-download model
from huggingface_hub import snapshot_download
model_path = './pyramid-flow-miniflux'
if not os.path.exists(model_path):
    snapshot_download("rain1011/pyramid-flow-miniflux", local_dir=model_path)

from pyramid_dit import PyramidDiTForVideoGeneration
from diffusers.utils import export_to_video

# Setup MPS with compatibility fixes
if torch.backends.mps.is_available():
    device = "mps"
    torch.set_default_dtype(torch.float32)
    # Apply float64 patches (see above)
    
# Load model
model = PyramidDiTForVideoGeneration(
    model_path,
    model_name="pyramid_flux", 
    model_dtype='fp16' if device == "mps" else 'fp32',
    model_variant='diffusion_transformer_384p'
)

# Generate video
with torch.no_grad():
    frames = model.generate_i2v(
        prompt="A mystical scene with magical energy",
        input_image=your_image,
        num_inference_steps=[5, 5, 5],
        temp=8,
        video_guidance_scale=4.0,
        output_type="pil"
    )

# Save with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
export_to_video(frames, f"./video_{timestamp}.mp4", fps=24)
```

## ⚡ Performance Optimization Tips

### Memory Usage (96GB+ Systems)
- Use `save_memory=False` for faster VAE decoding
- Higher resolution: `1280x768` with `diffusion_transformer_768p`
- More inference steps: `[10, 10, 10]` for better quality

### Speed vs Quality Trade-offs
- **Fastest**: 384p, 5 steps, 8 frames, `save_memory=True`
- **Balanced**: 768p, 7 steps, 12 frames, `save_memory=False`  
- **Highest Quality**: 768p, 10 steps, 16 frames, `save_memory=False`

## 🚨 Common Issues & Solutions

### "MPS device does not support float64"
- **Cause**: Default PyTorch operations creating float64 tensors
- **Fix**: Apply float64→float32 monkey patches (see above)

### "Conv3d is not currently supported on MPS"
- **Cause**: Outdated PyTorch version
- **Fix**: Update to PyTorch 2.9.0+ nightly build

### "No module named 'timm'" or dataclass errors
- **Cause**: Wrong timm version (0.6.12 has Python 3.11+ issues)
- **Fix**: `pip install "timm>=0.9.0"`

### Video generation hangs/OOM
- **Cause**: Resolution too high for available memory
- **Fix**: Reduce resolution, use `save_memory=True`, fewer frames

### "ffmpeg not found" during video export
- **Fix**: `brew install ffmpeg`

## 🔄 Adapting for Other Video Models

These patterns apply to other PyTorch video generation models on Mac:

1. **Always check Conv3D support**: Most video models need 3D convolutions
2. **Float64→Float32 patches**: Universal MPS compatibility requirement  
3. **Memory management**: Set appropriate `PYTORCH_MPS_HIGH_WATERMARK_RATIO`
4. **No autocast on MPS**: Use conditional context managers
5. **Path configuration**: Ensure proper module imports

## 📚 References

- [Pyramid-Flow Repository](https://github.com/jy0205/Pyramid-Flow)
- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [MPS Conv3D Support PR](https://github.com/pytorch/pytorch/pull/114183)

## 🎉 Success Indicators

When everything works correctly, you should see:
```
Using MPS (Metal Performance Shaders) with Conv3D support!
MPS memory before generation:
RAM usage: 45%
Starting video generation with 640x384 resolution...
Generated 8 frames successfully!
Saving video as: ./image_to_video_sample_20250123_143025.mp4
```

Happy video generation! 🎬✨ 