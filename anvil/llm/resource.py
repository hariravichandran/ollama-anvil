"""Resource pressure manager — adaptive inference for low-resource systems.

Monitors system memory in real-time and provides recommendations for
running larger models on constrained hardware. Key strategies:

1. **Memory pressure detection** — triggers when available RAM drops
2. **Adaptive context sizing** — shrinks context window under pressure
3. **Model fit estimation** — predicts whether a model will fit before loading
4. **GPU layer splitting** — recommends partial GPU offloading for iGPU/hybrid
5. **Aggressive mode** — extra-small context + low batch for very tight systems

Usage::

    from anvil.llm.resource import ResourceManager
    mgr = ResourceManager()
    status = mgr.check_pressure()
    if status.level == "critical":
        # Reduce context, switch to smaller model, etc.
        recommended_ctx = status.recommended_context
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from anvil.utils.logging import get_logger

log = get_logger("llm.resource")

__all__ = [
    "ResourceManager",
    "ResourceStatus",
    "ModelFitResult",
    "PRESSURE_THRESHOLDS",
    "MIN_FREE_MB",
    "BYTES_PER_TOKEN",
    "OVERHEAD_MB",
]

# Pressure levels: percentage of total RAM that is free
PRESSURE_THRESHOLDS = {
    "low": 30,       # >30% free — no pressure
    "moderate": 15,   # 15-30% free — start conserving
    "high": 8,        # 8-15% free — aggressive conservation
    "critical": 3,    # <3% free — emergency mode
}

# Minimum free memory to maintain (MB)
MIN_FREE_MB = 512

# Approximate bytes per token for context window estimation
# Based on Ollama's KV cache: ~2 bytes per token per layer for fp16,
# ~1 byte for q8, ~0.5 for q4. Using 1.5 as conservative average.
BYTES_PER_TOKEN = 1.5

# Estimated overhead for model runtime (MB)
OVERHEAD_MB = 500


@dataclass(frozen=True, slots=True)
class ResourceStatus:
    """Current resource pressure status."""

    level: str           # "low", "moderate", "high", "critical"
    free_ram_mb: int
    total_ram_mb: int
    free_pct: float
    recommended_context: int
    recommended_batch: int
    notes: list[str]

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Pressure: {self.level} ({self.free_pct:.1f}% free)",
            f"Free RAM: {self.free_ram_mb:,} MB / {self.total_ram_mb:,} MB",
            f"Recommended context: {self.recommended_context:,} tokens",
            f"Recommended batch: {self.recommended_batch}",
        ]
        if self.notes:
            for n in self.notes:
                lines.append(f"  - {n}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ModelFitResult:
    """Result of checking whether a model fits in available resources."""

    fits: bool
    model_size_mb: int
    available_mb: int
    headroom_mb: int
    max_context_if_loaded: int
    recommendation: str

    def summary(self) -> str:
        """Human-readable summary."""
        status = "FITS" if self.fits else "DOES NOT FIT"
        return (
            f"{status}: model ~{self.model_size_mb:,} MB, "
            f"available {self.available_mb:,} MB, "
            f"headroom {self.headroom_mb:,} MB\n"
            f"Max context if loaded: {self.max_context_if_loaded:,} tokens\n"
            f"{self.recommendation}"
        )


class ResourceManager:
    """Monitor system resources and provide adaptive recommendations.

    Reads /proc/meminfo on Linux (falls back to psutil or estimates).
    For unified memory systems (Strix Halo), also reads GPU VRAM usage
    from sysfs to avoid false pressure warnings when system RAM is low
    but GPU memory is abundant.
    """

    def __init__(self, gpu_vram_gb: float = 0.0, is_unified_memory: bool = False) -> None:
        self._total_ram_mb = self._detect_total_ram()
        # Auto-detect unified memory if not explicitly provided
        if gpu_vram_gb == 0.0:
            detected = self._detect_gpu_vram_total()
            if detected is not None:
                gpu_vram_gb = detected / 1024  # MB to GB
                # Heuristic: if VRAM > 32 GB and GTT exists, it's unified memory
                from pathlib import Path
                gtt_path = None
                for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
                    gtt = card / "device" / "mem_info_gtt_total"
                    if gtt.exists():
                        gtt_path = gtt
                        break
                if gtt_path and gpu_vram_gb > 32:
                    is_unified_memory = True
        self._gpu_vram_gb = gpu_vram_gb
        self._is_unified_memory = is_unified_memory

    @staticmethod
    def _detect_gpu_vram_total() -> int | None:
        """Detect total GPU VRAM in MB from sysfs (AMD only)."""
        try:
            from pathlib import Path
            for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
                total_path = card / "device" / "mem_info_vram_total"
                if total_path.exists():
                    total = int(total_path.read_text().strip())
                    return total // (1024 * 1024)
        except (OSError, ValueError):
            pass
        return None

    def __repr__(self) -> str:
        return f"ResourceManager(total_ram={self._total_ram_mb} MB)"

    @staticmethod
    def _detect_total_ram() -> int:
        """Detect total system RAM in MB."""
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) // 1024
        except (OSError, ValueError, IndexError):
            log.debug("Could not read /proc/meminfo for total RAM")
        # Fallback: os.sysconf
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (pages * page_size) // (1024 * 1024)
        except (ValueError, OSError):
            return 8192  # fallback: assume 8GB

    @staticmethod
    def _detect_free_ram() -> int:
        """Detect available system RAM in MB (includes cached/buffers)."""
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                mem_info: dict[str, int] = {}
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        mem_info[key] = int(parts[1])
                # MemAvailable is the best metric (accounts for cache/buffers)
                if "MemAvailable" in mem_info:
                    return mem_info["MemAvailable"] // 1024
                # Fallback: Free + Buffers + Cached
                free = mem_info.get("MemFree", 0)
                buffers = mem_info.get("Buffers", 0)
                cached = mem_info.get("Cached", 0)
                return (free + buffers + cached) // 1024
        except (OSError, ValueError):
            log.debug("Could not read /proc/meminfo for free RAM")
        return 4096  # fallback: assume 4GB free

    @staticmethod
    def _detect_gpu_vram_free_mb() -> int | None:
        """Read free GPU VRAM from sysfs (AMD only). Returns MB or None."""
        try:
            from pathlib import Path
            drm = Path("/sys/class/drm")
            for card in sorted(drm.glob("card[0-9]*")):
                total_path = card / "device" / "mem_info_vram_total"
                used_path = card / "device" / "mem_info_vram_used"
                if total_path.exists() and used_path.exists():
                    total = int(total_path.read_text().strip())
                    used = int(used_path.read_text().strip())
                    return (total - used) // (1024 * 1024)
        except (OSError, ValueError):
            pass
        return None

    def check_pressure(self, target_context: int = 8192) -> ResourceStatus:
        """Check current resource pressure and recommend settings.

        For unified memory systems, uses GPU VRAM free space instead of
        system RAM to determine pressure level, since models load into
        GPU memory while system RAM remains mostly unchanged.

        Args:
            target_context: Desired context window size.

        Returns:
            ResourceStatus with pressure level and recommendations.
        """
        if self._is_unified_memory:
            # Unified memory: check GPU VRAM usage instead of system RAM
            gpu_free_mb = self._detect_gpu_vram_free_mb()
            if gpu_free_mb is not None:
                total_mb = int(self._gpu_vram_gb * 1024)
                free_mb = gpu_free_mb
                free_pct = (free_mb / max(1, total_mb)) * 100
            else:
                # Fallback to system RAM if sysfs unavailable
                free_mb = self._detect_free_ram()
                total_mb = self._total_ram_mb
                free_pct = (free_mb / max(1, total_mb)) * 100
        else:
            free_mb = self._detect_free_ram()
            total_mb = self._total_ram_mb
            free_pct = (free_mb / max(1, total_mb)) * 100

        notes: list[str] = []

        # Determine pressure level
        if free_pct > PRESSURE_THRESHOLDS["low"]:
            level = "low"
            ctx = target_context
            batch = 2048
        elif free_pct > PRESSURE_THRESHOLDS["moderate"]:
            level = "moderate"
            ctx = min(target_context, 4096)
            batch = 1024
            notes.append("Context reduced to 4096 due to moderate memory pressure")
        elif free_pct > PRESSURE_THRESHOLDS["high"]:
            level = "high"
            ctx = min(target_context, 2048)
            batch = 512
            notes.append("Context reduced to 2048 due to high memory pressure")
            notes.append("Consider switching to a smaller model or lower quantization")
        else:
            level = "critical"
            ctx = min(target_context, 1024)
            batch = 256
            notes.append("CRITICAL: Very low memory — using minimal context (1024)")
            notes.append("Strongly recommend a smaller model (1.5B-3B)")

        return ResourceStatus(
            level=level,
            free_ram_mb=free_mb,
            total_ram_mb=total_mb,
            free_pct=round(free_pct, 1),
            recommended_context=ctx,
            recommended_batch=batch,
            notes=notes,
        )

    def estimate_model_fit(
        self, model_size_gb: float, target_context: int = 8192, num_layers: int = 32,
    ) -> ModelFitResult:
        """Estimate whether a model will fit in available memory.

        For unified memory systems, uses GPU VRAM as the memory pool.

        Args:
            model_size_gb: Approximate model size in GB.
            target_context: Desired context window.
            num_layers: Number of model layers (affects KV cache size).

        Returns:
            ModelFitResult with fit status and recommendations.
        """
        if self._is_unified_memory:
            gpu_free = self._detect_gpu_vram_free_mb()
            free_mb = gpu_free if gpu_free is not None else self._detect_free_ram()
        else:
            free_mb = self._detect_free_ram()
        model_size_mb = int(model_size_gb * 1024)

        # KV cache estimate: bytes_per_token * num_layers * context * 2 (key+value)
        kv_cache_mb = int(BYTES_PER_TOKEN * num_layers * target_context * 2 / (1024 * 1024))

        total_needed = model_size_mb + kv_cache_mb + OVERHEAD_MB
        headroom = free_mb - total_needed
        fits = headroom >= MIN_FREE_MB

        # Calculate max context that would fit
        available_for_kv = max(0, free_mb - model_size_mb - OVERHEAD_MB - MIN_FREE_MB)
        max_ctx = int(available_for_kv * 1024 * 1024 / max(1, BYTES_PER_TOKEN * num_layers * 2))
        max_ctx = max(512, min(max_ctx, 131072))  # clamp

        if fits:
            recommendation = f"Model fits with {headroom:,} MB headroom. Safe to load."
        elif headroom > -512:
            recommendation = (
                f"Tight fit — reduce context to {max_ctx:,} tokens "
                f"or use a more aggressive quantization (q2_k, q3_k)."
            )
        else:
            recommendation = (
                f"Model too large by {abs(headroom):,} MB. Options:\n"
                f"  1. Use a smaller quantization (q2_k saves ~50% vs q4_k_m)\n"
                f"  2. Use a smaller model (3B or 1.5B)\n"
                f"  3. Reduce context window to {max_ctx:,} tokens\n"
                f"  4. Close other applications to free RAM"
            )

        return ModelFitResult(
            fits=fits,
            model_size_mb=model_size_mb,
            available_mb=free_mb,
            headroom_mb=headroom,
            max_context_if_loaded=max_ctx,
            recommendation=recommendation,
        )

    def get_gpu_offload_layers(
        self, model_size_gb: float, gpu_memory_gb: float, total_layers: int = 32,
    ) -> int:
        """Calculate how many layers to offload to GPU for iGPU/hybrid systems.

        For iGPU systems with limited shared memory, partial offloading can
        significantly speed up inference by running some layers on GPU.

        Args:
            model_size_gb: Model size in GB.
            gpu_memory_gb: Available GPU memory in GB.
            total_layers: Total number of model layers.

        Returns:
            Number of layers to offload to GPU (0 = CPU only).
        """
        if gpu_memory_gb <= 0:
            return 0

        # Each layer takes roughly model_size / total_layers
        per_layer_gb = model_size_gb / max(1, total_layers)

        # Reserve 20% of GPU memory for KV cache and overhead
        usable_gpu_gb = gpu_memory_gb * 0.8

        # Calculate how many layers fit
        layers = int(usable_gpu_gb / max(0.01, per_layer_gb))
        return min(layers, total_layers)
