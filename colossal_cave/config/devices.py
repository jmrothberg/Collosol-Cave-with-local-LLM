"""Device selection — MPS/CUDA/CPU without Gradio dependencies."""

import os
import platform

import torch


def detect_devices() -> dict:
    """Return llm/diffuser/video device strings based on available hardware."""
    if platform.system() == 'Darwin' and torch.backends.mps.is_available():
        llm_device = "mps"
        diffuser_device = "mps"
        video_device = "mps"
        dtype = torch.float32
    elif torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 3:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
            llm_device = "cuda:0"
            diffuser_device = "cuda:1"
            video_device = "cuda:2"
        elif num_gpus == 2:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
            llm_device = "cuda:0"
            diffuser_device = "cuda:1"
            video_device = None
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            llm_device = "cuda:0"
            diffuser_device = "cuda:0"
            video_device = None
        dtype = torch.float16
    else:
        llm_device = "cpu"
        diffuser_device = "cpu"
        video_device = None
        dtype = torch.float32

    return {
        "llm_device": llm_device,
        "diffuser_device": diffuser_device,
        "video_device": video_device,
        "dtype": dtype,
    }
