"""Vulkan GPU detection and environment configuration.

Vulkan is a cross-platform GPU compute backend that Ollama supports as an
alternative to ROCm (AMD), CUDA (NVIDIA), and Metal (Apple). It's especially
useful for:

- AMD GPUs where ROCm is not installed or not working
- Intel GPUs (iGPU and Arc) without dedicated compute drivers
- Systems where the primary driver doesn't support LLM compute
- Cross-platform fallback when vendor-specific drivers fail

Ollama uses OLLAMA_GPU_LIBRARY=vulkan to enable the Vulkan backend.

Usage::

    from anvil.hardware.vulkan import detect_vulkan, configure_vulkan_env

    vk = detect_vulkan()
    if vk.available:
        env = configure_vulkan_env(vk)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

from anvil.utils.logging import get_logger

log = get_logger("hardware.vulkan")

__all__ = [
    "VulkanInfo",
    "detect_vulkan",
    "configure_vulkan_env",
    "VULKAN_ENV_KEY",
    "VULKANINFO_TIMEOUT",
]

# Ollama environment variable to force Vulkan backend
VULKAN_ENV_KEY = "OLLAMA_GPU_LIBRARY"

# Timeout for vulkaninfo subprocess
VULKANINFO_TIMEOUT = 10  # seconds

# Regex for parsing vulkaninfo output
_VK_DEVICE_NAME_RE = re.compile(r"deviceName\s*=\s*(.+)")
_VK_API_VERSION_RE = re.compile(r"apiVersion\s*=\s*([\d.]+)")
_VK_DEVICE_TYPE_RE = re.compile(r"deviceType\s*=\s*(\S+)")
_VK_HEAP_SIZE_RE = re.compile(r"size\s*=\s*(\d+)")
_VK_DRIVER_VERSION_RE = re.compile(r"driverVersion\s*=\s*([\d.]+)")


class VulkanInfo:
    """Detected Vulkan GPU information."""

    __slots__ = (
        "available", "device_name", "device_type", "api_version",
        "driver_version", "heap_size_mb", "device_count",
    )

    def __init__(
        self,
        available: bool = False,
        device_name: str = "",
        device_type: str = "",
        api_version: str = "",
        driver_version: str = "",
        heap_size_mb: int = 0,
        device_count: int = 0,
    ) -> None:
        self.available = available
        self.device_name = device_name
        self.device_type = device_type
        self.api_version = api_version
        self.driver_version = driver_version
        self.heap_size_mb = heap_size_mb
        self.device_count = device_count

    def __repr__(self) -> str:
        if not self.available:
            return "VulkanInfo(not available)"
        mem = f"{self.heap_size_mb}MB" if self.heap_size_mb else "unknown"
        return (
            f"VulkanInfo({self.device_name!r}, type={self.device_type}, "
            f"api={self.api_version}, mem={mem})"
        )

    @property
    def heap_size_gb(self) -> float:
        """Heap size in GB."""
        return self.heap_size_mb / 1024


def detect_vulkan() -> VulkanInfo:
    """Detect Vulkan GPU support via vulkaninfo.

    Returns VulkanInfo with available=False if Vulkan is not detected.
    """
    if not shutil.which("vulkaninfo"):
        log.debug("vulkaninfo not found — Vulkan detection skipped")
        return VulkanInfo()

    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True, text=True, timeout=VULKANINFO_TIMEOUT,
        )
        if result.returncode != 0:
            log.debug("vulkaninfo failed with code %d", result.returncode)
            return VulkanInfo()

        return _parse_vulkaninfo(result.stdout)

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        log.debug("Could not run vulkaninfo: %s", e)
        return VulkanInfo()


def _parse_vulkaninfo(output: str) -> VulkanInfo:
    """Parse vulkaninfo --summary output."""
    device_name = ""
    device_type = ""
    api_version = ""
    driver_version = ""
    heap_size_mb = 0
    device_count = 0

    for line in output.splitlines():
        line = line.strip()

        match = _VK_DEVICE_NAME_RE.search(line)
        if match and not device_name:
            device_name = match.group(1).strip()
            device_count += 1
            continue

        match = _VK_API_VERSION_RE.search(line)
        if match and not api_version:
            api_version = match.group(1)
            continue

        match = _VK_DEVICE_TYPE_RE.search(line)
        if match and not device_type:
            device_type = match.group(1).strip()
            continue

        match = _VK_DRIVER_VERSION_RE.search(line)
        if match and not driver_version:
            driver_version = match.group(1)
            continue

        # Count additional GPU devices
        if "deviceName" in line and device_name:
            device_count += 1

    if not device_name:
        return VulkanInfo()

    return VulkanInfo(
        available=True,
        device_name=device_name,
        device_type=device_type,
        api_version=api_version,
        driver_version=driver_version,
        heap_size_mb=heap_size_mb,
        device_count=max(1, device_count),
    )


def configure_vulkan_env(vk: VulkanInfo | None = None) -> dict[str, str]:
    """Set environment variables to use Vulkan as Ollama's GPU backend.

    This should be called when:
    - ROCm is not available but an AMD GPU is present
    - Intel GPU detected but xe/i915 compute not working
    - User explicitly requests Vulkan via --vulkan flag

    Args:
        vk: Optional VulkanInfo. If None, will detect automatically.

    Returns:
        Dict of environment variables that were set.
    """
    if vk is None:
        vk = detect_vulkan()

    env_vars: dict[str, str] = {}

    if not vk.available:
        log.info("Vulkan not available — cannot configure")
        return env_vars

    # Tell Ollama to use Vulkan backend
    os.environ.setdefault(VULKAN_ENV_KEY, "vulkan")
    env_vars[VULKAN_ENV_KEY] = "vulkan"
    log.info("Set %s=vulkan (device: %s)", VULKAN_ENV_KEY, vk.device_name)

    # Flash attention works with Vulkan on supported hardware
    os.environ.setdefault("OLLAMA_FLASH_ATTENTION", "1")
    env_vars["OLLAMA_FLASH_ATTENTION"] = "1"

    # Limit loaded models on Vulkan (more conservative than CUDA/ROCm)
    os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")
    env_vars["OLLAMA_MAX_LOADED_MODELS"] = "1"

    return env_vars
