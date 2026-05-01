#!/bin/bash
# Install a ROCm-compatible LoRA / QLoRA training stack into the active venv.
#
# Prefer invoking via 'anvil train install [--strix-halo|--no-fa|--hqq-only]'.
#
# Pinned for ROCm 6.4 + PyTorch 2.5 (the combo with working bitsandbytes
# wheels and flash-attention support on RDNA3 / RDNA3.5). On Strix Halo
# (gfx1151) pass --strix-halo to swap in AMD's manylinux full-bundle
# wheels (only path that has gfx1151 kernels right now).
#
# Usage:
#   source .venv/bin/activate
#   bash install-training-stack.sh                  # default: full stack
#   bash install-training-stack.sh --no-fa          # skip flash-attn
#   bash install-training-stack.sh --hqq-only       # skip bitsandbytes
#   bash install-training-stack.sh --strix-halo     # gfx1151 / Bosgame M5

set -euo pipefail

ROCM_TAG="rocm6.4"
TORCH_VER=""
NO_FA=0
HQQ_ONLY=0

# Strix Halo / RDNA 3.5 (gfx1151) is NOT in the pytorch.org rocm6.4 wheel's
# arch list — torch.randn(...) throws hipErrorNoBinaryForGpu on first use.
# AMD's manylinux full-bundle wheels at repo.radeon.com/rocm/manylinux DO
# include gfx1151 kernels and bundle their own ROCm libs (5.6 GB download
# but self-contained). Set USE_AMD_MANYLINUX_WHEEL=1 to use this path.
USE_AMD_MANYLINUX_WHEEL="${USE_AMD_MANYLINUX_WHEEL:-0}"
AMD_MANYLINUX_TORCH_URL="${AMD_MANYLINUX_TORCH_URL:-https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.2/torch-2.7.1%2Brocm7.2.2.git1dab218d-cp312-cp312-linux_x86_64.whl}"
AMD_MANYLINUX_TORCHVISION_URL="${AMD_MANYLINUX_TORCHVISION_URL:-https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.2/torchvision-0.22.1%2Brocm7.2.2.git59a3e1f9-cp312-cp312-linux_x86_64.whl}"

for arg in "$@"; do
    case "$arg" in
        --no-fa)    NO_FA=1 ;;
        --hqq-only) HQQ_ONLY=1 ;;
        --rocm)     shift; ROCM_TAG="$1" ;;
        --strix-halo|--gfx1151)
            USE_AMD_MANYLINUX_WHEEL=1
            ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
    esac
done

RED='\033[0;31m'; GRN='\033[0;32m'; CYN='\033[0;36m'; NC='\033[0m'
info() { echo -e "${CYN}[install]${NC} $*"; }
ok()   { echo -e "${GRN}[ok]${NC}      $*"; }
err()  { echo -e "${RED}[error]${NC}   $*"; exit 1; }

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    err "No active virtualenv. Run 'source .venv/bin/activate' first."
fi
info "Target venv: $VIRTUAL_ENV"

python -m pip install --upgrade pip wheel setuptools

# 1. PyTorch (ROCm).
if [[ "$USE_AMD_MANYLINUX_WHEEL" == "1" ]]; then
    info "Installing AMD manylinux torch (full-bundle, includes ROCm runtime)..."
    info "  source: $AMD_MANYLINUX_TORCH_URL"
    python -m pip install --upgrade --force-reinstall --no-deps \
        "$AMD_MANYLINUX_TORCH_URL" \
        || err "AMD manylinux torch install failed"
    info "Installing matching torchvision..."
    python -m pip install --force-reinstall --no-deps \
        "$AMD_MANYLINUX_TORCHVISION_URL" \
        || err "AMD manylinux torchvision install failed"
else
    # --force-reinstall is important: a CUDA torch build sometimes gets
    # pulled in transitively by other deps and must be evicted, otherwise
    # torch.cuda.is_available() returns True but uses CUDA stubs that
    # crash on AMD silicon.
    TORCH_SPEC="torch"
    [[ -n "$TORCH_VER" ]] && TORCH_SPEC="torch==${TORCH_VER}"
    info "Installing ${TORCH_SPEC} from AMD's ${ROCM_TAG} wheel index..."
    python -m pip install --upgrade --force-reinstall \
        --index-url "https://download.pytorch.org/whl/${ROCM_TAG}" \
        $TORCH_SPEC torchvision torchaudio \
        || err "torch ROCm install failed — check ROCm tag matches /opt/rocm/.info/version"
fi

python - <<'PYEOF'
import torch, sys
hip = getattr(torch.version, "hip", None)
if not hip:
    sys.exit("torch installed but has no HIP backend (got CPU-only wheel)")
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False — ROCm runtime not picked up")
print(f"torch {torch.__version__} hip={hip} device={torch.cuda.get_device_name(0)}")
PYEOF
ok "torch + ROCm verified"

# 2. HuggingFace stack.
info "Installing transformers / peft / trl / accelerate / datasets..."
python -m pip install --upgrade \
    "transformers>=4.46,<5" \
    "peft>=0.13,<1" \
    "trl>=0.12,<1" \
    "accelerate>=1.1" \
    "datasets>=3.0,<4" \
    "safetensors>=0.4.5" \
    "huggingface_hub>=0.26,<1" \
    "sentencepiece>=0.2" \
    "protobuf>=3.20" \
    "tensorboard>=2.18" \
    "evaluate>=0.4"

# 3. HQQ — always install; reliable fallback when bnb kernels misbehave.
info "Installing HQQ (ROCm-friendly quantizer)..."
python -m pip install --upgrade "hqq>=0.2.2"

# 4. bitsandbytes (ROCm fork). Upstream PyPI bnb links against CUDA and
# silently no-ops the 4-bit kernels on AMD; use AMD's ROCm fork wheels.
if [[ $HQQ_ONLY -eq 0 ]]; then
    info "Installing bitsandbytes (ROCm fork)..."
    if python -m pip install --upgrade \
            --index-url "https://download.pytorch.org/whl/${ROCM_TAG}" \
            "bitsandbytes" 2>/dev/null; then
        ok "bitsandbytes installed from ROCm wheel index"
    else
        info "ROCm wheel index miss — trying AMD's GitHub fork..."
        python -m pip install --upgrade \
            "bitsandbytes @ git+https://github.com/ROCm/bitsandbytes.git@rocm_enabled_multi_backend" \
            || err "bitsandbytes ROCm install failed — re-run with --hqq-only to skip"
    fi
fi

# 5. flash-attn (ROCm fork). Compiles HIP kernels — needs ROCm dev headers.
if [[ $NO_FA -eq 0 ]]; then
    info "Installing flash-attn (ROCm fork) — can take 10+ minutes to build..."
    if [[ ! -d /opt/rocm/include ]]; then
        echo "warning: /opt/rocm/include missing — flash-attn build will fail; skipping"
    else
        FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
        python -m pip install --upgrade --no-build-isolation \
            "flash-attn @ git+https://github.com/ROCm/flash-attention.git@main_perf" \
            || echo "warning: flash-attn build failed — trainer will fall back to SDPA"
    fi
fi

# 6. Liger-Kernel (Triton fused kernels — works on ROCm via triton's HIP backend).
info "Installing liger-kernel..."
python -m pip install --upgrade "liger-kernel>=0.4" \
    || echo "warning: liger-kernel install failed (non-fatal — trainer falls back to plain HF)"

# 7. xformers (optional memory-efficient attention; ROCm support since 0.0.23).
info "Installing xformers..."
python -m pip install --upgrade --no-deps "xformers>=0.0.28" \
    || echo "warning: xformers install failed (non-fatal)"

# Final probe.
info "Verifying stack..."
python - <<'PYEOF'
import importlib, torch
mods = ["transformers", "peft", "trl", "accelerate", "datasets", "hqq"]
for m in mods:
    v = getattr(importlib.import_module(m), "__version__", "?")
    print(f"  {m:14s} {v}")
try:
    import bitsandbytes as bnb
    print(f"  {'bitsandbytes':14s} {bnb.__version__}")
    x = torch.randn(16, 16, device="cuda", dtype=torch.float16)
    q, s = bnb.functional.quantize_nf4(x)
    bnb.functional.dequantize_nf4(q, s)
    print("  bnb NF4 kernels: OK")
except ImportError:
    print("  bitsandbytes:   not installed (HQQ-only mode)")
except Exception as e:
    print(f"  bnb NF4 kernels FAILED: {e}")
    print("  -> Use HQQ in your trainer config (see docs/qlora-training.md)")
PYEOF

ok "Training stack installed."
echo ""
echo "Next: anvil train diagnose      # confirm system is ready"
echo "Then: anvil train run --model meta-llama/Llama-3.1-8B"
