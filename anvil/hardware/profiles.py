"""Hardware profiles: map detected hardware to optimal Ollama settings.

Supports CPU-only systems — Ollama runs models on CPU when no GPU is available.
CPU inference is slower but fully functional. The framework auto-detects this
and selects appropriately sized models with optimized settings.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from anvil.hardware.detect import HardwareInfo
from anvil.utils.logging import get_logger

log = get_logger("hardware.profiles")

__all__ = [
    "HardwareProfile",
    "PROFILES",
    "select_profile",
    "recommend_models",
    "VALID_PROFILE_NAMES",
    "OS_MEMORY_RESERVATION_GB",
    "LOW_RAM_THRESHOLD_GB",
    "OS_THREAD_RESERVATION",
    "CPU_MAX_BATCH_SIZE",
    "LOW_RAM_CONTEXT",
]

# System resource reservations
OS_MEMORY_RESERVATION_GB = 4.0  # GB reserved for OS + other apps
OS_THREAD_RESERVATION = 2  # threads reserved for OS
CPU_MAX_BATCH_SIZE = 512  # smaller batches are faster on CPU
LOW_RAM_THRESHOLD_GB = 8  # below this, force smallest models
LOW_RAM_CONTEXT = 2048  # context size for very low RAM systems

# Valid profile names
VALID_PROFILE_NAMES = {"compact", "standard", "workstation", "high_memory", "ultra_memory"}


@dataclass(slots=True)
class HardwareProfile:
    """Optimal settings for a hardware tier."""

    name: str
    min_gpu_gb: float
    recommended_model: str
    fallback_model: str
    larger_model: str  # for complex tasks
    num_ctx: int
    num_batch: int
    max_threads: int
    description: str
    is_cpu_only: bool = False  # True when running without GPU acceleration

    def __post_init__(self) -> None:
        """Validate profile fields."""
        self.min_gpu_gb = max(0.0, self.min_gpu_gb)
        self.max_threads = max(1, self.max_threads)
        self.num_ctx = max(128, self.num_ctx)
        self.num_batch = max(1, self.num_batch)

    def __repr__(self) -> str:
        cpu = " (CPU-only)" if self.is_cpu_only else ""
        return f"HardwareProfile({self.name!r}, {self.recommended_model}, ctx={self.num_ctx}{cpu})"


# Hardware profiles from compact to high-end
PROFILES = [
    HardwareProfile(
        name="compact",
        min_gpu_gb=0,
        recommended_model="qwen2.5-coder:3b",
        fallback_model="qwen2.5-coder:1.5b",
        larger_model="qwen2.5-coder:7b",
        num_ctx=4096,
        num_batch=1024,
        max_threads=8,
        description="Low-memory systems (iGPU, <8 GB usable). Small models, short context.",
    ),
    HardwareProfile(
        name="standard",
        min_gpu_gb=8,
        recommended_model="qwen2.5-coder:7b",
        fallback_model="qwen2.5-coder:3b",
        larger_model="qwen2.5-coder:14b",
        num_ctx=8192,
        num_batch=2048,
        max_threads=14,
        description="Mid-range systems (8-20 GB GPU). 7B models with good context.",
    ),
    HardwareProfile(
        name="workstation",
        min_gpu_gb=20,
        recommended_model="qwen2.5-coder:14b",
        fallback_model="qwen2.5-coder:7b",
        larger_model="qwen2.5-coder:32b",
        num_ctx=32768,
        num_batch=4096,
        max_threads=14,
        description="Workstation systems (20-60 GB GPU). 14B models with large context.",
    ),
    HardwareProfile(
        name="high_memory",
        min_gpu_gb=60,
        recommended_model="qwen2.5-coder:32b",
        fallback_model="qwen2.5-coder:14b",
        larger_model="qwen2.5:72b",
        num_ctx=65536,
        num_batch=8192,
        max_threads=16,
        description="High-memory systems (60-90 GB GPU). 32B+ models with maximum context.",
    ),
    HardwareProfile(
        name="ultra_memory",
        min_gpu_gb=90,
        recommended_model="qwen2.5:72b",
        fallback_model="qwen2.5-coder:32b",
        larger_model="llama3.3:70b",
        num_ctx=131072,
        num_batch=16384,
        max_threads=16,
        description="Ultra-high memory unified systems (90+ GB). 70B+ models with 128K context.",
    ),
]

# Model recommendation tables (static — avoid recreating per call)
_GENERAL_MODELS = {
    "compact": "llama3.2:3b",
    "standard": "qwen3:8b",
    "workstation": "qwen3:30b",
    "high_memory": "llama3.3:70b",
    "ultra_memory": "llama3.3:70b",
}

_REASONING_MODELS = {
    "compact": "deepseek-r1:1.5b",
    "standard": "deepseek-r1:8b",
    "workstation": "deepseek-r1:14b",
    "high_memory": "deepseek-r1:32b",
    "ultra_memory": "deepseek-r1:70b",
}

_CREATIVE_MODELS = {
    "compact": "gemma3:4b",
    "standard": "gemma3:12b",
    "workstation": "gemma3:27b",
    "high_memory": "gemma3:27b",
    "ultra_memory": "gemma3:27b",
}


def select_profile(hw: HardwareInfo) -> HardwareProfile:
    """Select the best hardware profile based on detected hardware.

    For CPU-only systems, uses available RAM to determine model size.
    For iGPUs, uses usable_gb (total minus OS reservation).
    For dGPUs, uses total VRAM.
    """
    is_cpu_only = hw.gpu.vendor == "none" or hw.gpu.driver == "cpu"

    if is_cpu_only:
        # CPU-only: use RAM to determine model size
        # Ollama loads models into RAM when no GPU is available
        # Leave ~4 GB for OS + other apps
        available_ram = max(0, hw.ram_gb - OS_MEMORY_RESERVATION_GB)
        gpu_gb = available_ram
        log.info("CPU-only mode: using %.1f GB RAM for models (%.1f GB total)", available_ram, hw.ram_gb)
    elif hw.gpu.is_unified_memory:
        # Unified memory APU (Strix Halo): use usable_gb (VRAM minus OS headroom)
        gpu_gb = hw.gpu.usable_gb
    elif hw.gpu.is_igpu:
        gpu_gb = hw.gpu.usable_gb
    else:
        gpu_gb = hw.gpu.total_gb

    # Select highest profile that fits
    selected = PROFILES[0]  # default to compact
    for profile in PROFILES:
        if gpu_gb >= profile.min_gpu_gb:
            selected = profile

    # Copy to avoid mutating the shared PROFILES list elements
    selected = copy.copy(selected)

    # Adjust thread count to actual hardware
    selected.max_threads = min(selected.max_threads, hw.cpu.threads)

    # Mark CPU-only mode and optimize settings
    if is_cpu_only:
        selected.is_cpu_only = True
        # CPU inference benefits from more threads and smaller batches
        selected.max_threads = max(1, hw.cpu.threads - OS_THREAD_RESERVATION)
        selected.num_batch = min(selected.num_batch, CPU_MAX_BATCH_SIZE)

        # For very low RAM systems, force the smallest model
        if hw.ram_gb < LOW_RAM_THRESHOLD_GB:
            selected.recommended_model = "qwen2.5-coder:1.5b"
            selected.fallback_model = "qwen2.5-coder:0.5b"
            selected.num_ctx = LOW_RAM_CONTEXT

    log.info("Selected profile: %s (%.1f GB %s, %d threads%s)",
             selected.name, gpu_gb,
             "RAM" if is_cpu_only else "GPU",
             selected.max_threads,
             ", CPU-only" if is_cpu_only else "")
    return selected


def recommend_models(hw: HardwareInfo) -> list[dict[str, str]]:
    """Recommend models based on hardware, across different categories."""
    profile = select_profile(hw)
    is_cpu_only = profile.is_cpu_only

    if is_cpu_only:
        memory_str = f"{hw.ram_gb:.0f} GB RAM, CPU-only"
    elif hw.gpu.is_unified_memory:
        memory_str = f"{hw.gpu.usable_gb:.0f} GB unified memory"
    elif hw.gpu.is_igpu:
        memory_str = f"{hw.gpu.usable_gb:.0f} GB usable"
    else:
        memory_str = f"{hw.gpu.total_gb:.0f} GB GPU"

    recommendations = []

    # Coding models
    reason = f"Best coding model for your {profile.name} hardware ({memory_str})"
    if is_cpu_only:
        reason += ". Slower on CPU but fully functional"
    recommendations.append({
        "category": "Coding",
        "model": profile.recommended_model,
        "reason": reason,
    })

    # General chat
    recommendations.append({
        "category": "General chat",
        "model": _GENERAL_MODELS.get(profile.name, "llama3.2:3b"),
        "reason": "Good all-around conversational model",
    })

    # Research / reasoning
    recommendations.append({
        "category": "Reasoning",
        "model": _REASONING_MODELS.get(profile.name, "deepseek-r1:8b"),
        "reason": "Chain-of-thought reasoning for complex problems",
    })

    # Creative writing
    recommendations.append({
        "category": "Creative writing",
        "model": _CREATIVE_MODELS.get(profile.name, "gemma3:4b"),
        "reason": "Strong creative and instructional capabilities",
    })

    return recommendations
