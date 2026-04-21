"""Hardware detection: GPU, CPU, and RAM — cross-platform.

Supports:
- Linux: sysfs (AMD), nvidia-smi (NVIDIA), lspci (Intel), vulkaninfo (fallback)
- macOS: system_profiler, sysctl (Apple Silicon, Intel Macs)
- Windows: nvidia-smi (NVIDIA), WMI/PowerShell (AMD, Intel)
- Vulkan: cross-platform fallback via vulkaninfo when native drivers unavailable
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from anvil.utils.logging import get_logger

log = get_logger("hardware.detect")

__all__ = [
    "GPUInfo",
    "CPUInfo",
    "HardwareInfo",
    "detect_hardware",
    "SYSTEM",
    "OS_MEMORY_HEADROOM_GB",
    "IGPU_VRAM_THRESHOLD_GB",
    "UMA_LARGE_VRAM_THRESHOLD_GB",
    "TIMEOUT_FAST",
    "TIMEOUT_NORMAL",
    "TIMEOUT_SLOW",
    "AMD_VENDOR_ID",
    "INTEL_VENDOR_ID",
]

SYSTEM = platform.system()  # "Linux", "Darwin", "Windows"

# Subprocess timeouts (seconds)
TIMEOUT_FAST = 5  # quick queries like sysctl
TIMEOUT_NORMAL = 10  # standard queries like nvidia-smi, lspci
TIMEOUT_SLOW = 15  # heavier queries like system_profiler

# Vendor IDs (PCI)
AMD_VENDOR_ID = "0x1002"
INTEL_VENDOR_ID = "0x8086"

# Memory thresholds
OS_MEMORY_HEADROOM_GB = 4.0  # GB reserved for OS in shared-memory configs
IGPU_VRAM_THRESHOLD_GB = 8.0  # below this VRAM + GTT > 0 = iGPU
UMA_LARGE_VRAM_THRESHOLD_GB = 32.0  # VRAM above this with GTT present = unified memory APU (e.g. Strix Halo)

# Pre-compiled regex patterns for hardware info parsing
_LSPCI_NAME_RE = re.compile(r"\] (.+)$")
_VERSION_NUM_RE = re.compile(r"(\d+\.\d+[\.\d]*)")
_ROCM_MAJOR_RE = re.compile(r"ROCM_VERSION_MAJOR\s+(\d+)")
_ROCM_MINOR_RE = re.compile(r"ROCM_VERSION_MINOR\s+(\d+)")
_ROCM_PATCH_RE = re.compile(r"ROCM_VERSION_PATCH\s+(\d+)")
_PCI_ID_SUFFIX_RE = re.compile(r"\s*\[[\da-fA-F]{4}:[\da-fA-F]{4}\]\s*$")
_ARC_GPU_RE = re.compile(r"\bArc\b", re.IGNORECASE)
_CPU_MODEL_RE = re.compile(r"model name\s*:\s*(.+)")
_CORE_ID_RE = re.compile(r"core id\s*:\s*(\d+)")
_PHYS_ID_RE = re.compile(r"physical id\s*:\s*(\d+)")
_MEMTOTAL_RE = re.compile(r"MemTotal:\s+(\d+)\s+kB")


@dataclass(slots=True)
class GPUInfo:
    """Detected GPU information."""

    vendor: str = "none"  # "amd", "nvidia", "intel", "apple", "none"
    name: str = "Unknown"
    vram_gb: float = 0.0
    gtt_gb: float = 0.0  # AMD UMA shared memory
    total_gb: float = 0.0
    driver: str = ""  # "rocm", "cuda", "metal", "vulkan", "xe", "i915", "directx", "cpu"
    driver_version: str = ""
    architecture: str = ""  # e.g., "gfx1035" for RDNA2, "arm64" for Apple Silicon
    is_igpu: bool = False
    is_unified_memory: bool = False  # Large unified memory APU (e.g. Strix Halo, Apple Silicon)

    def __post_init__(self) -> None:
        """Clamp memory values to non-negative."""
        self.vram_gb = max(0.0, self.vram_gb)
        self.gtt_gb = max(0.0, self.gtt_gb)
        self.total_gb = max(0.0, self.total_gb)

    def __repr__(self) -> str:
        mem = f"{self.total_gb:.1f}GB"
        igpu = " iGPU" if self.is_igpu else ""
        return f"GPUInfo({self.vendor}, {self.name!r}, {mem}{igpu}, {self.driver or 'no driver'})"

    @property
    def usable_gb(self) -> float:
        """Estimated usable memory for LLM inference."""
        if self.is_igpu:
            # iGPU shares system RAM — leave ~4GB for OS
            return max(0, self.total_gb - OS_MEMORY_HEADROOM_GB)
        if self.is_unified_memory:
            # Unified memory APU (Strix Halo, Apple Silicon) — use VRAM as the
            # usable pool. GTT is overflow/slower and shouldn't be counted for
            # model fitting. Leave headroom for OS and KV cache overhead.
            return max(0, self.vram_gb - OS_MEMORY_HEADROOM_GB)
        if self.vendor == "apple":
            # Apple Silicon unified memory — leave headroom for OS
            return max(0, self.total_gb - OS_MEMORY_HEADROOM_GB)
        return self.total_gb


@dataclass(slots=True)
class CPUInfo:
    """Detected CPU information."""

    model: str = "Unknown"
    threads: int = 1
    cores: int = 1


@dataclass(slots=True)
class HardwareInfo:
    """Complete hardware detection result."""

    gpu: GPUInfo = field(default_factory=GPUInfo)
    cpu: CPUInfo = field(default_factory=CPUInfo)
    ram_gb: float = 0.0
    platform: str = SYSTEM

    def summary(self) -> str:
        """Human-readable hardware summary."""
        lines = [
            f"GPU:  {self.gpu.name} ({self.gpu.vendor})",
            f"      {self.gpu.total_gb:.1f} GB ({self.gpu.driver or 'no driver'})",
        ]
        if self.gpu.is_unified_memory:
            lines.append(f"      Unified memory: {self.gpu.vram_gb:.1f} GB VRAM (usable: {self.gpu.usable_gb:.1f} GB)")
        elif self.gpu.is_igpu:
            lines.append(f"      iGPU: {self.gpu.vram_gb:.1f} GB VRAM + {self.gpu.gtt_gb:.1f} GB GTT (shared)")
        elif self.gpu.vendor == "apple":
            lines.append("      Unified memory (shared with CPU)")
        lines.extend([
            f"CPU:  {self.cpu.model}",
            f"      {self.cpu.threads} threads ({self.cpu.cores} cores)",
            f"RAM:  {self.ram_gb:.1f} GB",
            f"OS:   {self.platform}",
        ])
        return "\n".join(lines)


# In-process cache — hardware doesn't change within a single process lifetime
_hardware_cache: HardwareInfo | None = None


def detect_hardware(use_cache: bool = True) -> HardwareInfo:
    """Detect GPU, CPU, and RAM. Returns HardwareInfo with all findings.

    Results are cached in-process since hardware doesn't change at runtime.
    Pass use_cache=False to force re-detection.
    """
    global _hardware_cache
    if use_cache and _hardware_cache is not None:
        return _hardware_cache

    info = HardwareInfo()
    info.gpu = _detect_gpu()
    info.cpu = _detect_cpu()
    info.ram_gb = _detect_ram()
    log.debug("Hardware detected: GPU=%s (%.1f GB), CPU=%s (%d threads), RAM=%.1f GB",
              info.gpu.name, info.gpu.total_gb, info.cpu.model, info.cpu.threads, info.ram_gb)
    _hardware_cache = info
    return info


# ─── GPU Detection ───────────────────────────────────────────────────────────


def _detect_gpu() -> GPUInfo:
    """Detect GPU — dispatches to platform-specific detection."""
    if SYSTEM == "Darwin":
        apple = _detect_apple_gpu()
        if apple:
            return apple

    # NVIDIA works on all platforms
    nvidia = _detect_nvidia_gpu()
    if nvidia and nvidia.total_gb > 0:
        return nvidia

    if SYSTEM == "Linux":
        amd = _detect_amd_gpu_linux()
        if amd and amd.total_gb > 0:
            return amd

        intel = _detect_intel_gpu_linux()
        if intel and intel.total_gb > 0:
            return intel

    elif SYSTEM == "Windows":
        wmi = _detect_gpu_windows()
        if wmi and wmi.total_gb > 0:
            return wmi

    # Vulkan fallback — cross-platform GPU compute when native drivers unavailable
    vulkan_gpu = _detect_vulkan_fallback()
    if vulkan_gpu:
        return vulkan_gpu

    log.info("No GPU detected — CPU-only mode")
    return GPUInfo(vendor="none", name="CPU only", driver="cpu")


def _detect_nvidia_gpu() -> GPUInfo | None:
    """Detect NVIDIA GPU via nvidia-smi (works on Linux, Windows, macOS)."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=TIMEOUT_NORMAL,
        )
        if result.returncode != 0:
            return None

        lines = result.stdout.strip().splitlines()
        if not lines:
            return None
        line = lines[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            return None

        name = parts[0]
        vram_mb = float(parts[1])
        driver_version = parts[2]

        return GPUInfo(
            vendor="nvidia",
            name=name,
            vram_gb=vram_mb / 1024,
            total_gb=vram_mb / 1024,
            driver="cuda",
            driver_version=driver_version,
            is_igpu=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


def _detect_apple_gpu() -> GPUInfo | None:
    """Detect Apple Silicon GPU on macOS via sysctl and system_profiler."""
    if platform.machine() != "arm64":
        return None  # Intel Mac — no Metal GPU compute for LLMs

    try:
        # Get chip name
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=TIMEOUT_FAST,
        )
        chip_name = result.stdout.strip() if result.returncode == 0 else "Apple Silicon"

        # Total unified memory (this IS the GPU memory on Apple Silicon)
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=TIMEOUT_FAST,
        )
        total_bytes = int(result.stdout.strip()) if result.returncode == 0 else 0
        total_gb = total_bytes / (1024 ** 3)

        # Get GPU core count from system_profiler
        gpu_cores = ""
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True, text=True, timeout=TIMEOUT_NORMAL,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if isinstance(data, dict):
                    displays = data.get("SPDisplaysDataType", [])
                    if displays:
                        gpu_cores = displays[0].get("sppci_cores", "")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            log.debug("Could not parse GPU cores from system_profiler: %s", e)

        name = chip_name
        if gpu_cores:
            name += f" ({gpu_cores}-core GPU)"

        return GPUInfo(
            vendor="apple",
            name=name,
            total_gb=total_gb,
            driver="metal",
            driver_version="Metal 3",
            architecture="arm64",
            is_igpu=False,  # Apple Silicon unified memory is not "iGPU" in the traditional sense
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def _detect_amd_gpu_linux() -> GPUInfo | None:
    """Detect AMD GPU via Linux sysfs."""
    drm_path = Path("/sys/class/drm")
    if not drm_path.exists():
        return None

    best_gpu = None
    best_total = 0.0

    for card_dir in sorted(drm_path.glob("card[0-9]*")):
        device_dir = card_dir / "device"
        if not device_dir.exists():
            continue

        # Check if AMD
        vendor_path = device_dir / "vendor"
        if vendor_path.exists():
            vendor_id = vendor_path.read_text(encoding="utf-8", errors="replace").strip()
            if vendor_id != AMD_VENDOR_ID:
                continue

        vram_path = device_dir / "mem_info_vram_total"
        gtt_path = device_dir / "mem_info_gtt_total"

        if not vram_path.exists():
            continue

        vram_bytes = int(vram_path.read_text(encoding="utf-8", errors="replace").strip())
        gtt_bytes = int(gtt_path.read_text(encoding="utf-8", errors="replace").strip()) if gtt_path.exists() else 0

        vram_gb = vram_bytes / (1024 ** 3)
        gtt_gb = gtt_bytes / (1024 ** 3)

        # Classify memory architecture:
        # - Traditional iGPU (Rembrandt, Phoenix, Strix Point): small VRAM (<8GB) + large GTT
        # - Unified memory APU (Strix Halo): large VRAM (>=32GB) + GTT present
        #   VRAM already represents the full unified memory pool; adding GTT double-counts.
        # - Discrete GPU: VRAM only, no GTT
        is_igpu = vram_gb < IGPU_VRAM_THRESHOLD_GB and gtt_gb > 0
        is_unified_memory = vram_gb >= UMA_LARGE_VRAM_THRESHOLD_GB and gtt_gb > 0

        if is_unified_memory:
            # Strix Halo and similar: VRAM is the unified pool, GTT is overflow mapping.
            # Use VRAM as total to avoid inflating memory estimates.
            total_gb = vram_gb
        else:
            # Traditional iGPU: VRAM (carved out) + GTT (shared system RAM)
            # Discrete GPU: VRAM only (gtt_gb will be 0)
            total_gb = vram_gb + gtt_gb

        if total_gb > best_total:
            best_total = total_gb
            name = _read_amd_gpu_name(device_dir)

            arch = ""
            rev_path = device_dir / "gpu_id"
            if rev_path.exists():
                arch = rev_path.read_text(encoding="utf-8", errors="replace").strip()

            driver = "cpu"
            driver_version = ""
            rocm_ver = _detect_rocm_version()
            if rocm_ver:
                driver = "rocm"
                driver_version = rocm_ver

            best_gpu = GPUInfo(
                vendor="amd",
                name=name,
                vram_gb=vram_gb,
                gtt_gb=gtt_gb,
                total_gb=total_gb,
                driver=driver,
                driver_version=driver_version,
                architecture=arch,
                is_igpu=is_igpu,
                is_unified_memory=is_unified_memory,
            )

    return best_gpu


def _read_amd_gpu_name(device_dir: Path) -> str:
    """Read AMD GPU product name from sysfs or lspci."""
    for name_file in ["product_name", "pp_dpm_sclk"]:
        path = device_dir / name_file
        if path.exists() and name_file == "product_name":
            name = path.read_text(encoding="utf-8", errors="replace").strip()
            if name:
                return name

    try:
        result = subprocess.run(
            ["lspci", "-d", "1002:", "-nn"],
            capture_output=True, text=True, timeout=TIMEOUT_FAST,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                if "VGA" in line or "Display" in line:
                    match = _LSPCI_NAME_RE.search(line)
                    if match:
                        return match.group(1).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.debug("Could not read AMD GPU name via lspci: %s", e)

    return "AMD GPU"


def _detect_rocm_version() -> str:
    """Detect ROCm version if installed."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showversion"],
            capture_output=True, text=True, timeout=TIMEOUT_FAST,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "ROCm" in line or "version" in line.lower():
                    match = _VERSION_NUM_RE.search(line)
                    if match:
                        return match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.debug("Could not detect ROCm version via rocm-smi: %s", e)

    version_file = Path("/opt/rocm/.info/version")
    if version_file.exists():
        return version_file.read_text(encoding="utf-8", errors="replace").strip()

    ver_header = Path("/opt/rocm/include/rocm-core/rocm_version.h")
    if ver_header.exists():
        content = ver_header.read_text(encoding="utf-8", errors="replace")
        major = _ROCM_MAJOR_RE.search(content)
        minor = _ROCM_MINOR_RE.search(content)
        if major and minor:
            patch = _ROCM_PATCH_RE.search(content)
            return f"{major.group(1)}.{minor.group(1)}.{patch.group(1) if patch else '0'}"

    return ""


def _detect_intel_gpu_linux() -> GPUInfo | None:
    """Detect Intel GPU via lspci and sysfs (Linux only).

    Supports Intel Arc discrete GPUs (A-series) with dedicated VRAM,
    as well as integrated Intel UHD/Iris GPUs.
    """
    try:
        result = subprocess.run(
            ["lspci", "-d", "8086:", "-nn"],
            capture_output=True, text=True, timeout=TIMEOUT_FAST,
        )
        if result.returncode != 0:
            return None

        for line in result.stdout.splitlines():
            if "VGA" not in line and "Display" not in line:
                continue

            # Parse GPU name from lspci output
            match = _LSPCI_NAME_RE.search(line)
            raw_name = match.group(1).strip() if match else "Intel GPU"

            # Clean up: strip trailing PCI IDs like [8086:56a0]
            name = _PCI_ID_SUFFIX_RE.sub("", raw_name).strip()
            if not name:
                name = raw_name

            # Determine if this is a discrete Arc GPU
            is_arc = bool(_ARC_GPU_RE.search(name))

            # Detect driver: prefer xe (modern), fall back to i915
            driver = "cpu"
            driver_version = ""
            for drv_name in ("xe", "i915"):
                drv_path = Path(f"/sys/module/{drv_name}")
                if drv_path.exists():
                    driver = drv_name
                    ver_path = drv_path / "version"
                    if ver_path.exists():
                        try:
                            driver_version = ver_path.read_text(encoding="utf-8", errors="replace").strip()
                        except OSError as e:
                            log.debug("Could not read driver version from %s: %s", ver_path, e)
                    break

            # Detect VRAM from sysfs
            vram_gb = _detect_intel_vram_linux()

            return GPUInfo(
                vendor="intel",
                name=name,
                vram_gb=vram_gb,
                total_gb=vram_gb,
                driver=driver,
                driver_version=driver_version,
                is_igpu=not is_arc,
            )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        log.debug("Could not detect Intel GPU via lspci: %s", e)
    return None


def _detect_intel_vram_linux() -> float:
    """Detect Intel GPU VRAM from sysfs.

    Intel Arc discrete GPUs expose VRAM through /sys/class/drm/card*/device/.
    """
    drm_path = Path("/sys/class/drm")
    if not drm_path.exists():
        return 0.0

    for card_dir in sorted(drm_path.glob("card[0-9]*")):
        device_dir = card_dir / "device"
        if not device_dir.exists():
            continue

        # Check if Intel (vendor ID 0x8086)
        vendor_path = device_dir / "vendor"
        if vendor_path.exists():
            try:
                vendor_id = vendor_path.read_text(encoding="utf-8", errors="replace").strip()
                if vendor_id != INTEL_VENDOR_ID:
                    continue
            except OSError:
                continue

        # Dedicated VRAM (Intel Arc discrete GPUs)
        vram_path = device_dir / "mem_info_vram_total"
        if vram_path.exists():
            try:
                vram_bytes = int(vram_path.read_text(encoding="utf-8", errors="replace").strip())
                return vram_bytes / (1024 ** 3)
            except (ValueError, OSError) as e:
                log.debug("Could not read Intel VRAM from %s: %s", vram_path, e)

        # Alternative: lmem_total_bytes
        lmem_path = device_dir / "lmem_total_bytes"
        if lmem_path.exists():
            try:
                lmem_bytes = int(lmem_path.read_text(encoding="utf-8", errors="replace").strip())
                return lmem_bytes / (1024 ** 3)
            except (ValueError, OSError) as e:
                log.debug("Could not read Intel lmem from %s: %s", lmem_path, e)

    return 0.0


def _detect_gpu_windows() -> GPUInfo | None:
    """Detect GPU on Windows via PowerShell WMI (AMD, Intel fallback)."""
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-CimInstance Win32_VideoController | "
             "Select-Object Name, AdapterRAM, DriverVersion | "
             "ConvertTo-Json"],
            capture_output=True, text=True, timeout=TIMEOUT_NORMAL,
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return None

        for gpu in data:
            name = gpu.get("Name", "")
            adapter_ram = gpu.get("AdapterRAM", 0) or 0
            driver_ver = gpu.get("DriverVersion", "")
            total_gb = adapter_ram / (1024 ** 3) if adapter_ram else 0.0

            vendor = "none"
            if "AMD" in name or "Radeon" in name:
                vendor = "amd"
            elif "Intel" in name:
                vendor = "intel"

            if vendor != "none":
                return GPUInfo(
                    vendor=vendor,
                    name=name,
                    total_gb=total_gb,
                    driver="directx",
                    driver_version=driver_ver,
                    is_igpu=total_gb < 4.0,
                )
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError) as e:
        log.debug("Could not detect GPU via Windows WMI: %s", e)
    return None


def _detect_vulkan_fallback() -> GPUInfo | None:
    """Detect GPU via Vulkan as a fallback when native drivers are unavailable.

    This is tried after AMD/NVIDIA/Intel/Apple-specific detection fails.
    Vulkan provides GPU compute for systems where ROCm/CUDA are not installed.
    """
    try:
        from anvil.hardware.vulkan import detect_vulkan
        vk = detect_vulkan()
        if not vk.available:
            return None

        # Determine vendor from device name
        name_lower = vk.device_name.lower()
        if "amd" in name_lower or "radeon" in name_lower:
            vendor = "amd"
        elif "nvidia" in name_lower or "geforce" in name_lower or "rtx" in name_lower:
            vendor = "nvidia"
        elif "intel" in name_lower or "arc" in name_lower or "iris" in name_lower:
            vendor = "intel"
        else:
            vendor = "none"

        is_igpu = vk.device_type in ("INTEGRATED_GPU", "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU")
        total_gb = vk.heap_size_gb if vk.heap_size_mb > 0 else 0.0

        log.info("Vulkan fallback: %s (%s, %.1f GB)", vk.device_name, vendor, total_gb)
        return GPUInfo(
            vendor=vendor,
            name=vk.device_name,
            total_gb=total_gb,
            driver="vulkan",
            driver_version=vk.api_version,
            is_igpu=is_igpu,
        )
    except ImportError:
        log.debug("Vulkan module not available")
        return None


# ─── CPU Detection ───────────────────────────────────────────────────────────


def _detect_cpu() -> CPUInfo:
    """Detect CPU model and thread count — cross-platform."""
    if SYSTEM == "Linux":
        return _detect_cpu_linux()
    if SYSTEM == "Darwin":
        return _detect_cpu_macos()
    if SYSTEM == "Windows":
        return _detect_cpu_windows()
    return CPUInfo(threads=os.cpu_count() or 1, cores=(os.cpu_count() or 2) // 2)


def _detect_cpu_linux() -> CPUInfo:
    """Detect CPU from /proc/cpuinfo."""
    info = CPUInfo()
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        content = cpuinfo_path.read_text(encoding="utf-8", errors="replace")
        match = _CPU_MODEL_RE.search(content)
        if match:
            info.model = match.group(1).strip()
        info.threads = content.count("processor\t:")
        if info.threads == 0:
            info.threads = os.cpu_count() or 1
        core_ids = set(_CORE_ID_RE.findall(content))
        phys_ids = set(_PHYS_ID_RE.findall(content))
        if core_ids and phys_ids:
            info.cores = len(core_ids) * len(phys_ids)
        else:
            info.cores = info.threads // 2 or 1
    else:
        info.threads = os.cpu_count() or 1
        info.cores = info.threads // 2 or 1
    return info


def _detect_cpu_macos() -> CPUInfo:
    """Detect CPU on macOS via sysctl."""
    info = CPUInfo()
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=TIMEOUT_FAST,
        )
        if result.returncode == 0:
            info.model = result.stdout.strip()

        result = subprocess.run(
            ["sysctl", "-n", "hw.ncpu"],
            capture_output=True, text=True, timeout=TIMEOUT_FAST,
        )
        if result.returncode == 0:
            info.threads = int(result.stdout.strip())

        result = subprocess.run(
            ["sysctl", "-n", "hw.physicalcpu"],
            capture_output=True, text=True, timeout=TIMEOUT_FAST,
        )
        if result.returncode == 0:
            info.cores = int(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        info.threads = os.cpu_count() or 1
        info.cores = info.threads // 2 or 1
    return info


def _detect_cpu_windows() -> CPUInfo:
    """Detect CPU on Windows via PowerShell."""
    info = CPUInfo()
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-CimInstance Win32_Processor | "
             "Select-Object Name, NumberOfCores, NumberOfLogicalProcessors | "
             "ConvertTo-Json"],
            capture_output=True, text=True, timeout=TIMEOUT_NORMAL,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if isinstance(data, list):
                data = data[0] if data else {}
            if isinstance(data, dict):
                info.model = data.get("Name", "Unknown").strip()
                info.cores = data.get("NumberOfCores", 1)
                info.threads = data.get("NumberOfLogicalProcessors", 1)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        info.threads = os.cpu_count() or 1
        info.cores = info.threads // 2 or 1
    return info


# ─── RAM Detection ───────────────────────────────────────────────────────────


def _detect_ram() -> float:
    """Detect total system RAM in GB — cross-platform."""
    if SYSTEM == "Linux":
        return _detect_ram_linux()
    if SYSTEM == "Darwin":
        return _detect_ram_macos()
    if SYSTEM == "Windows":
        return _detect_ram_windows()

    # Fallback: psutil
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        return 0.0


def _detect_ram_linux() -> float:
    """Detect RAM from /proc/meminfo."""
    meminfo_path = Path("/proc/meminfo")
    if meminfo_path.exists():
        content = meminfo_path.read_text(encoding="utf-8", errors="replace")
        match = _MEMTOTAL_RE.search(content)
        if match:
            return int(match.group(1)) / (1024 * 1024)
    return 0.0


def _detect_ram_macos() -> float:
    """Detect RAM on macOS via sysctl."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=TIMEOUT_FAST,
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / (1024 ** 3)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError) as e:
        log.debug("Could not detect RAM on macOS: %s", e)
    return 0.0


def _detect_ram_windows() -> float:
    """Detect RAM on Windows via PowerShell."""
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"],
            capture_output=True, text=True, timeout=TIMEOUT_NORMAL,
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / (1024 ** 3)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError) as e:
        log.debug("Could not detect RAM on Windows: %s", e)
    return 0.0
