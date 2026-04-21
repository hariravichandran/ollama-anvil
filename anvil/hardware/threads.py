"""CPU thread auto-tuning for optimal inference performance.

LLM inference on CPU is heavily compute-bound. Optimal thread count depends on:
- Physical cores vs logical threads (hyperthreading adds overhead, not speed)
- P-cores vs E-cores on Intel hybrid architectures
- Memory bandwidth (diminishing returns above NUMA node boundary)
- OS reservation (leave threads for system responsiveness)

Usage::

    from anvil.hardware.threads import recommend_threads

    threads = recommend_threads(cpu_threads=16, cpu_cores=8, hw_class="cpu")
"""

from __future__ import annotations

from anvil.utils.logging import get_logger

log = get_logger("hardware.threads")

__all__ = [
    "recommend_threads",
    "ThreadRecommendation",
    "OS_RESERVED_THREADS",
    "HT_EFFICIENCY",
]

# Threads reserved for OS and background processes
OS_RESERVED_THREADS = 2

# Hyperthreading efficiency factor for LLM inference
# HT threads add ~15-30% throughput for LLM workloads (not 100%)
HT_EFFICIENCY = 0.2


class ThreadRecommendation:
    """Thread count recommendation with reasoning."""

    __slots__ = ("threads", "reason", "physical_cores", "has_hyperthreading")

    def __init__(
        self,
        threads: int,
        reason: str,
        physical_cores: int,
        has_hyperthreading: bool,
    ) -> None:
        self.threads = threads
        self.reason = reason
        self.physical_cores = physical_cores
        self.has_hyperthreading = has_hyperthreading

    def __repr__(self) -> str:
        ht = " (HT)" if self.has_hyperthreading else ""
        return f"ThreadRecommendation({self.threads} threads, {self.physical_cores} cores{ht})"


def recommend_threads(
    cpu_threads: int,
    cpu_cores: int,
    hw_class: str = "cpu",
    aggressive: bool = False,
) -> ThreadRecommendation:
    """Recommend optimal thread count for LLM inference.

    Args:
        cpu_threads: Total logical CPU threads.
        cpu_cores: Physical CPU cores.
        hw_class: Hardware class ("cpu", "igpu", "dgpu").
        aggressive: If True, use more threads (at cost of system responsiveness).

    Returns:
        ThreadRecommendation with thread count and reasoning.
    """
    has_ht = cpu_threads > cpu_cores

    if hw_class == "dgpu":
        # dGPU: CPU mainly handles tokenization, not heavy compute
        # Use fewer threads to leave resources for GPU scheduling
        threads = max(1, min(cpu_cores // 2, 8))
        reason = "dGPU mode: CPU handles tokenization only"
        return ThreadRecommendation(threads, reason, cpu_cores, has_ht)

    if hw_class == "igpu":
        # iGPU: CPU and GPU share resources, balance between them
        reserved = OS_RESERVED_THREADS
        threads = max(1, cpu_cores - reserved)
        reason = "iGPU mode: balanced CPU/GPU resource sharing"
        return ThreadRecommendation(threads, reason, cpu_cores, has_ht)

    # CPU-only: maximize throughput
    reserved = 1 if aggressive else OS_RESERVED_THREADS

    if has_ht:
        # Hyperthreading detected — use physical cores + small HT bonus
        # LLM inference benefits more from physical cores than HT threads
        ht_bonus = int((cpu_threads - cpu_cores) * HT_EFFICIENCY)
        threads = cpu_cores + ht_bonus - reserved
        reason = (
            f"CPU-only: {cpu_cores} physical cores + "
            f"{ht_bonus} HT bonus - {reserved} reserved"
        )
    else:
        # No hyperthreading — use all cores minus reservation
        threads = cpu_cores - reserved
        reason = f"CPU-only: {cpu_cores} cores - {reserved} reserved"

    threads = max(1, threads)

    # Cap at reasonable maximum (diminishing returns above ~32 threads for LLMs)
    if threads > 32 and not aggressive:
        threads = 32
        reason += " (capped at 32 — diminishing returns)"

    return ThreadRecommendation(threads, reason, cpu_cores, has_ht)
