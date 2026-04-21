"""NVIDIA CUDA-specific setup and environment variable configuration.

Configures Ollama environment variables for optimal performance on NVIDIA GPUs.
Handles CUDA Compute Capability detection, flash attention, tensor cores,
and memory management for both consumer (GeForce) and workstation (Quadro/A-series) GPUs.

Usage::

    from anvil.hardware.cuda import configure_cuda_env, get_cuda_status

    env = configure_cuda_env(gpu)
    status = get_cuda_status()
"""

from __future__ import annotations

import os
import shutil
import subprocess

from anvil.hardware.detect import GPUInfo
from anvil.utils.logging import get_logger

log = get_logger("hardware.cuda")

__all__ = [
    "configure_cuda_env",
    "get_cuda_status",
    "CUDA_COMPUTE_CAPABILITIES",
    "NVIDIA_SMI_TIMEOUT",
    "DGPU_MAX_LOADED_MODELS",
    "WORKSTATION_MAX_LOADED_MODELS",
]

# Timeout for nvidia-smi queries
NVIDIA_SMI_TIMEOUT = 10  # seconds

# Max loaded models by GPU class
DGPU_MAX_LOADED_MODELS = "2"
WORKSTATION_MAX_LOADED_MODELS = "3"

# CUDA Compute Capability thresholds for feature support
# Flash attention requires compute capability >= 7.0 (Volta+)
# Tensor cores available at >= 7.0
# FP8 support at >= 8.9 (Ada Lovelace)
CUDA_COMPUTE_CAPABILITIES: dict[str, float] = {
    "flash_attention": 7.0,   # Volta (V100), Turing (RTX 2000), Ampere (RTX 3000)+
    "tensor_cores": 7.0,      # V100 and newer
    "fp8": 8.9,               # Ada Lovelace (RTX 4000) and newer
    "int8_matmul": 7.5,       # Turing (RTX 2000) and newer
}

# Known GPU series → approximate compute capability
# Used when nvidia-smi doesn't report compute capability directly
_GPU_SERIES_COMPUTE: list[tuple[str, float]] = [
    # Ada Lovelace (RTX 40xx, L4, L40)
    ("rtx 40", 8.9), ("l40", 8.9), ("l4 ", 8.9),
    # Ampere (RTX 30xx, A100, A6000)
    ("rtx 30", 8.6), ("a100", 8.0), ("a6000", 8.6), ("a5000", 8.6),
    ("a40", 8.6), ("a30", 8.0), ("a16", 8.6), ("a10", 8.6), ("a2 ", 8.6),
    # Turing (RTX 20xx, T4)
    ("rtx 20", 7.5), ("t4", 7.5),
    # Volta (V100)
    ("v100", 7.0),
    # Pascal (GTX 10xx, P100) — no tensor cores
    ("gtx 10", 6.1), ("p100", 6.0),
    # Blackwell (RTX 50xx) — next gen
    ("rtx 50", 10.0),
    # Hopper (H100, H200)
    ("h100", 9.0), ("h200", 9.0),
]


def configure_cuda_env(gpu: GPUInfo) -> dict[str, str]:
    """Set CUDA environment variables for optimal Ollama performance.

    Args:
        gpu: Detected GPU info (must be vendor="nvidia", driver="cuda").

    Returns:
        Dict of environment variables that were set.
    """
    env_vars: dict[str, str] = {}

    if gpu.vendor != "nvidia" or gpu.driver != "cuda":
        return env_vars

    # Detect compute capability for feature gating
    compute_cap = _estimate_compute_capability(gpu.name)

    # Flash attention — major speedup on Volta+ (compute >= 7.0)
    if compute_cap >= CUDA_COMPUTE_CAPABILITIES["flash_attention"]:
        os.environ.setdefault("OLLAMA_FLASH_ATTENTION", "1")
        env_vars["OLLAMA_FLASH_ATTENTION"] = "1"
        log.info("CUDA flash attention enabled (compute %.1f)", compute_cap)
    else:
        log.info("CUDA flash attention skipped (compute %.1f < 7.0)", compute_cap)

    # Max loaded models — based on VRAM
    if gpu.total_gb >= 48:
        # Workstation GPU (A6000, A100 80GB, H100, etc.)
        os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", WORKSTATION_MAX_LOADED_MODELS)
        env_vars["OLLAMA_MAX_LOADED_MODELS"] = WORKSTATION_MAX_LOADED_MODELS
    elif gpu.total_gb >= 8:
        os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", DGPU_MAX_LOADED_MODELS)
        env_vars["OLLAMA_MAX_LOADED_MODELS"] = DGPU_MAX_LOADED_MODELS
    else:
        os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")
        env_vars["OLLAMA_MAX_LOADED_MODELS"] = "1"

    # CUDA memory pool — let Ollama manage CUDA memory allocation
    # This reduces fragmentation on consumer GPUs
    if gpu.total_gb <= 12:
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
        env_vars["CUDA_LAUNCH_BLOCKING"] = "0"

    return env_vars


def get_cuda_status() -> dict[str, str]:
    """Get CUDA runtime status from nvidia-smi.

    Returns dict with driver_version, cuda_version, gpu_utilization, memory_used,
    memory_total, temperature, and power_draw.
    """
    if not shutil.which("nvidia-smi"):
        return {"error": "nvidia-smi not found"}

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT,
        )
        if result.returncode != 0:
            return {"error": f"nvidia-smi returned {result.returncode}"}

        line = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            return {"error": "Unexpected nvidia-smi output format"}

        return {
            "driver_version": parts[0],
            "gpu_name": parts[1],
            "gpu_utilization_pct": parts[2],
            "memory_used_mb": parts[3],
            "memory_total_mb": parts[4],
            "temperature_c": parts[5],
            "power_draw_w": parts[6],
        }
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError, IndexError) as e:
        return {"error": str(e)}


def _estimate_compute_capability(gpu_name: str) -> float:
    """Estimate CUDA compute capability from GPU name.

    Uses known GPU series patterns when nvidia-smi doesn't report
    compute capability directly.
    """
    name_lower = gpu_name.lower()
    for pattern, cc in _GPU_SERIES_COMPUTE:
        if pattern in name_lower:
            return cc
    # Conservative default: assume Pascal-era (no tensor cores)
    return 6.1
