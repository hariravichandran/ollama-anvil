"""ROCm-specific setup and environment variable configuration."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from anvil.hardware.detect import GPUInfo
from anvil.utils.logging import get_logger

log = get_logger("hardware.rocm")

__all__ = [
    "configure_rocm_env",
    "validate_gfx_override",
    "get_rocm_status",
    "generate_ollama_service_env",
    "GFX_OVERRIDES",
    "ROCM_REQUIRED_GROUPS",
    "ROCM_OPTIONAL_GROUPS",
    "ROCM_VERSION_FILE",
    "ROCMINFO_TIMEOUT",
    "GROUPS_TIMEOUT",
    "IGPU_MAX_LOADED_MODELS",
    "DGPU_MAX_LOADED_MODELS",
]

# Subprocess timeouts
ROCMINFO_TIMEOUT = 10  # seconds for rocminfo query
GROUPS_TIMEOUT = 5  # seconds for groups query

# ROCm version file
ROCM_VERSION_FILE = Path("/opt/rocm/.info/version")

# Pre-compiled regex patterns for ROCm info parsing
_GFX_NAME_RE = re.compile(r"Name:\s+(gfx\d+)")
_GFX_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")

# iGPU vs discrete model limits
IGPU_MAX_LOADED_MODELS = "1"
DGPU_MAX_LOADED_MODELS = "2"

# Known GFX version overrides for AMD APUs/GPUs that need HSA_OVERRIDE_GFX_VERSION.
# Maps GFX architecture IDs to the required override version string.
#
# RDNA2 (GFX 10.3.x) — Radeon RX 6000 series & Rembrandt/Barcelo APUs
#   Needs HSA_OVERRIDE_GFX_VERSION="10.3.0"
#
# RDNA3 (GFX 11.0.x) — Radeon RX 7000 series & Phoenix/Hawk Point APUs
#   Needs HSA_OVERRIDE_GFX_VERSION="11.0.0"
#
# RDNA3.5 (GFX 11.5.x) — Strix Point APUs (Ryzen AI 300 series) & Strix Halo (Ryzen AI MAX series)
#   Strix Point: up to 33 GB shared memory (iGPU with GTT)
#   Strix Halo: 64-128 GB unified memory (large VRAM pool, similar to Apple Silicon)
#   Needs HSA_OVERRIDE_GFX_VERSION="11.0.0" (maps to nearest supported target)
#
# RDNA4 (GFX 12.0.x) — Radeon RX 9070 series (Navi 48/44)
#   Needs HSA_OVERRIDE_GFX_VERSION="12.0.0"
GFX_OVERRIDES: dict[str, str] = {
    # RDNA2 APUs (Rembrandt, Barcelo, etc.)
    "gfx1035": "10.3.0",
    "gfx1036": "10.3.0",
    # RDNA2 discrete (Radeon RX 6800/6900/6700/6600 series)
    "gfx1030": "10.3.0",
    "gfx1031": "10.3.0",
    "gfx1032": "10.3.0",
    # RDNA3 APUs (Phoenix, Hawk Point)
    "gfx1103": "11.0.0",
    # RDNA3 discrete (Radeon RX 7900/7800/7700/7600 series)
    "gfx1100": "11.0.0",
    "gfx1101": "11.0.0",
    "gfx1102": "11.0.0",
    # RDNA3.5 APUs (Strix Point — Ryzen AI 300, Strix Halo — Ryzen AI MAX)
    "gfx1150": "11.0.0",
    "gfx1151": "11.0.0",
    # RDNA4 discrete (Radeon RX 9070 series — Navi 48/44)
    "gfx1200": "12.0.0",
    "gfx1201": "12.0.0",
}


def configure_rocm_env(gpu: GPUInfo) -> dict[str, str]:
    """Set ROCm environment variables for optimal Ollama performance.

    Returns a dict of environment variables that were set.
    """
    env_vars: dict[str, str] = {}

    if gpu.vendor != "amd" or gpu.driver != "rocm":
        return env_vars

    # HSA_OVERRIDE_GFX_VERSION — needed for many AMD GPUs
    gfx_version = _detect_gfx_override(gpu)
    if gfx_version:
        os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", gfx_version)
        env_vars["HSA_OVERRIDE_GFX_VERSION"] = gfx_version
        log.info("Set HSA_OVERRIDE_GFX_VERSION=%s", gfx_version)

    # Flash attention — significant speedup on supported hardware
    os.environ.setdefault("OLLAMA_FLASH_ATTENTION", "1")
    env_vars["OLLAMA_FLASH_ATTENTION"] = "1"

    # Max loaded models — prevent OOM on iGPUs.
    # Unified memory APUs (Strix Halo) have enough memory for multiple models.
    if gpu.is_igpu and not gpu.is_unified_memory:
        os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", IGPU_MAX_LOADED_MODELS)
        env_vars["OLLAMA_MAX_LOADED_MODELS"] = IGPU_MAX_LOADED_MODELS
    else:
        os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", DGPU_MAX_LOADED_MODELS)
        env_vars["OLLAMA_MAX_LOADED_MODELS"] = DGPU_MAX_LOADED_MODELS

    # Reduce GPU overhead for unified memory — all memory is GPU-accessible
    if gpu.is_unified_memory:
        os.environ.setdefault("OLLAMA_GPU_OVERHEAD", "128")
        env_vars["OLLAMA_GPU_OVERHEAD"] = "128"
        log.info("Set OLLAMA_GPU_OVERHEAD=128 (unified memory APU)")

    return env_vars


def _detect_gfx_override(gpu: GPUInfo) -> str:
    """Detect the required HSA_OVERRIDE_GFX_VERSION for this GPU."""
    # Check if already set by user
    existing = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if existing:
        return existing

    # Try to detect GFX version from rocminfo
    gfx = _get_gfx_from_rocminfo()
    if gfx and gfx in GFX_OVERRIDES:
        return GFX_OVERRIDES[gfx]

    # Try architecture field from detection
    if gpu.architecture:
        for prefix, override in GFX_OVERRIDES.items():
            if gpu.architecture.startswith(prefix):
                return override

    return ""


def _get_gfx_from_rocminfo() -> str:
    """Extract GFX target from rocminfo output."""
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True, text=True, timeout=ROCMINFO_TIMEOUT,
        )
        if result.returncode == 0:
            match = _GFX_NAME_RE.search(result.stdout)
            if match:
                return match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.debug("Could not get GFX from rocminfo: %s", e)
    return ""


def validate_gfx_override(gfx_version: str) -> str:
    """Validate a user-provided HSA_OVERRIDE_GFX_VERSION.

    Returns error message or empty string if valid.
    """
    if not gfx_version:
        return "GFX version cannot be empty"

    # Must match format like "10.3.0", "11.0.0", "12.0.0"
    if not _GFX_VERSION_RE.match(gfx_version):
        return f"Invalid format: '{gfx_version}' (expected: X.Y.Z, e.g., '10.3.0')"

    # Check if it's a known valid override target
    valid_targets = set(GFX_OVERRIDES.values())
    if gfx_version not in valid_targets:
        suggestion = ", ".join(sorted(valid_targets))
        return f"Unknown GFX target '{gfx_version}'. Known targets: {suggestion}"

    return ""


# Required Linux groups for ROCm GPU access
ROCM_REQUIRED_GROUPS = {"render", "video"}

# Optional groups that may be needed for some configurations
ROCM_OPTIONAL_GROUPS = {"compute"}


def get_rocm_status() -> dict[str, str]:
    """Get current ROCm status for display."""
    status: dict[str, str] = {}

    # ROCm version
    version_file = ROCM_VERSION_FILE
    if version_file.exists():
        status["rocm_version"] = version_file.read_text(encoding="utf-8", errors="replace").strip()
    else:
        status["rocm_version"] = "not installed"

    # Current env vars
    for var in ["HSA_OVERRIDE_GFX_VERSION", "OLLAMA_FLASH_ATTENTION", "OLLAMA_MAX_LOADED_MODELS"]:
        val = os.environ.get(var)
        if val:
            status[var] = val

    # Validate GFX override if set
    gfx = os.environ.get("HSA_OVERRIDE_GFX_VERSION", "")
    if gfx:
        validation = validate_gfx_override(gfx)
        if validation:
            status["gfx_override_warning"] = validation

    # User groups (check required + optional)
    try:
        result = subprocess.run(["groups"], capture_output=True, text=True, timeout=GROUPS_TIMEOUT)
        groups_str = result.stdout.strip() if result.returncode == 0 else ""
        if not groups_str:
            log.debug("groups command returned empty output or failed (rc=%d)", result.returncode)
        else:
            groups = set(filter(None, groups_str.split()))
            for g in ROCM_REQUIRED_GROUPS:
                status[f"{g}_group"] = "yes" if g in groups else "no"
            for g in ROCM_OPTIONAL_GROUPS:
                if g in groups:
                    status[f"{g}_group"] = "yes"

            # Warn about missing required groups
            missing = ROCM_REQUIRED_GROUPS - groups
            if missing:
                status["group_warning"] = f"Missing required groups: {', '.join(sorted(missing))}. Fix with: sudo usermod -aG {','.join(sorted(missing))} $USER"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        log.debug("Could not check user groups for ROCm: %s", e)

    return status


def generate_ollama_service_env(gpu: GPUInfo) -> str:
    """Generate environment variables for Ollama systemd service override."""
    lines = ["# Generated by ollama-anvil for optimal ROCm performance"]

    gfx = _detect_gfx_override(gpu)
    if gfx:
        lines.append(f'Environment="HSA_OVERRIDE_GFX_VERSION={gfx}"')

    lines.append('Environment="OLLAMA_FLASH_ATTENTION=1"')

    max_models = IGPU_MAX_LOADED_MODELS if (gpu.is_igpu and not gpu.is_unified_memory) else DGPU_MAX_LOADED_MODELS
    lines.append(f'Environment="OLLAMA_MAX_LOADED_MODELS={max_models}"')

    if gpu.is_unified_memory:
        lines.append('Environment="OLLAMA_GPU_OVERHEAD=128"')

    return "\n".join(lines)
