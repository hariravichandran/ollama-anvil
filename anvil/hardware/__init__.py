"""Hardware detection and optimization for Ollama."""

from anvil.hardware.cuda import configure_cuda_env, get_cuda_status
from anvil.hardware.detect import HardwareInfo, detect_hardware
from anvil.hardware.profiles import HardwareProfile, select_profile
from anvil.hardware.vulkan import VulkanInfo, configure_vulkan_env, detect_vulkan

__all__ = [
    "detect_hardware", "HardwareInfo", "select_profile", "HardwareProfile",
    "configure_cuda_env", "get_cuda_status",
    "VulkanInfo", "detect_vulkan", "configure_vulkan_env",
]
