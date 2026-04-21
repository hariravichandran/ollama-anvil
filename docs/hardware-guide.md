# Hardware Guide

anvil auto-detects your hardware and picks a profile that fits. Run:

```bash
anvil hardware
```

## Profiles

| Profile     | GPU VRAM | Typical model         | Use case                      |
| ----------- | -------- | --------------------- | ----------------------------- |
| handheld    | < 4 GB   | llama3.2:1b, qwen2.5:1.5b | Phone-class chips, Pis    |
| laptop      | 4–8 GB   | qwen2.5:3b, llama3.2:3b   | Thin-and-light laptops    |
| desktop     | 8–20 GB  | qwen2.5-coder:7b      | Gaming PCs, midrange        |
| workstation | 20–60 GB | qwen2.5-coder:14b/32b | Pro GPUs, iGPUs w/ big RAM  |
| server      | 60+ GB   | qwen2.5:72b, llama3.1:70b | Multi-GPU rigs            |

## GPU backends

anvil detects:

- **AMD ROCm** — sets `HSA_OVERRIDE_GFX_VERSION` automatically for supported iGPUs
- **NVIDIA CUDA** — respects `CUDA_VISIBLE_DEVICES`
- **Vulkan** — fallback for unsupported GPUs
- **Apple Metal** — automatic on macOS
- **CPU-only** — fallback if no GPU is usable

### AMD iGPU tips

For Rembrandt / Phoenix / Strix Halo iGPUs, unified memory can expose a large
"VRAM" budget (the whole system RAM). anvil reports this and sizes model
recommendations accordingly.

If ROCm isn't loading properly:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 anvil hardware     # try a compatible target
anvil doctor                                        # surface any driver issues
```

### NVIDIA

Standard — if `nvidia-smi` shows your card, anvil will use it. Pin to one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 anvil chat
```

## Environment variables

All overrides go in `.env` (copy from `.env.example`). The most useful:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0     # AMD ROCm target
OLLAMA_FLASH_ATTENTION=1            # enable flash attention
OLLAMA_MAX_LOADED_MODELS=2          # how many models Ollama holds in memory
OLLAMA_NUM_PARALLEL=2               # parallel inference requests
CUDA_VISIBLE_DEVICES=0              # NVIDIA GPU pinning
```

Inspect what's active: `anvil env`.
