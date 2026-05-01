#!/bin/bash
# Diagnose why QLoRA / LoRA fine-tuning fails on AMD ROCm machines.
#
# Prefer invoking via 'anvil train diagnose [--quiet]'.
#
# Targets the unified-memory APU case (Strix Halo / Ryzen AI MAX) where
# the BIOS carves a large fixed UMA buffer out of physical RAM, leaving
# the OS with too little system memory for HF Transformers' CPU-side
# load stage. Also flags ROCm gfx-version mismatches and missing libs.

# NOTE: intentionally no `set -e` / `pipefail` — this is a diagnostic
# script, individual probe failures are findings, not script aborts.
set -u

QUIET=0
[[ "${1:-}" == "--quiet" ]] && QUIET=1

RED='\033[0;31m'; YEL='\033[0;33m'; GRN='\033[0;32m'; CYN='\033[0;36m'; NC='\033[0m'
say()  { [[ $QUIET -eq 0 ]] && echo -e "$*"; }
hdr()  { say "\n${CYN}== $* ==${NC}"; }
ok()   { say "  ${GRN}OK${NC}    $*"; }
warn() { say "  ${YEL}WARN${NC}  $*"; }
bad()  { say "  ${RED}FAIL${NC}  $*"; }

FINDINGS=()
add() { FINDINGS+=("$1"); }

# --- Hardware & memory --------------------------------------------------------
hdr "Hardware"
GFX_REPORTED="unknown"
if command -v rocminfo &>/dev/null; then
    ROCM_OUT=$(rocminfo 2>/dev/null || true)
    GFX_REPORTED=$(echo "$ROCM_OUT" | awk '/gfx[0-9]+/ {match($0, /gfx[0-9]+/); print substr($0, RSTART, RLENGTH); exit}')
    GPU_NAME=$(echo "$ROCM_OUT" | awk -F': *' '/Marketing Name/ {print $2; exit}')
    [[ -z "$GFX_REPORTED" ]] && GFX_REPORTED="unknown"
    [[ -z "$GPU_NAME" ]] && GPU_NAME="unknown"
    say "  GPU:           $GPU_NAME"
    say "  Reported gfx:  $GFX_REPORTED"
else
    bad "rocminfo not found — install rocm-smi-lib / rocminfo"
    add "rocminfo missing"
fi

VRAM_KB=0
if [[ -n "${ROCM_OUT:-}" ]]; then
    VRAM_KB=$(echo "$ROCM_OUT" | awk '
        /^\s*Name:\s*gfx/        { in_gpu=1 }
        /^\s*Device Type:\s*CPU/ { in_gpu=0 }
        in_gpu && /Pool Info/    { pool=1 }
        in_gpu && pool && /Size:.*KB/ {
            gsub(/[^0-9]/, "", $2); if ($2+0 > max) max=$2+0; pool=0
        }
        END { print max+0 }')
fi
if [[ ${VRAM_KB:-0} -le 0 ]]; then
    VRAM_KB=$(cat /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null | head -1 | awk '{print int($1/1024)}')
    VRAM_KB=${VRAM_KB:-0}
fi
VRAM_GB=$((VRAM_KB / 1024 / 1024))

RAM_KB=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
RAM_GB=$((RAM_KB / 1024 / 1024))
SWAP_KB=$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)
SWAP_GB=$((SWAP_KB / 1024 / 1024))

say "  System RAM:    ${RAM_GB} GB"
say "  GPU VRAM:      ${VRAM_GB} GB"
say "  Swap:          ${SWAP_GB} GB"

IS_UNIFIED=0
if [[ $VRAM_GB -ge 16 && $RAM_GB -le $((VRAM_GB * 2)) ]]; then
    IS_UNIFIED=1
    say "  Topology:      unified memory (APU) — VRAM is carved from physical RAM"
fi

# --- The big one: is system RAM enough for FP16 staging of 7B / 14B? ----------
hdr "CPU-side load budget"
NEED_7B=24    # 14 GB weights + ~10 GB headroom
NEED_14B=40   # 28 GB weights + ~12 GB headroom

if [[ $RAM_GB -ge $NEED_14B ]]; then
    ok "${RAM_GB} GB system RAM is enough for 14B FP16 staging"
elif [[ $RAM_GB -ge $NEED_7B ]]; then
    warn "${RAM_GB} GB system RAM: 7B FP16 staging fits, 14B does not (need ${NEED_14B} GB)"
    add "ram-too-small-for-14b"
else
    bad "${RAM_GB} GB system RAM is below the 7B threshold (${NEED_7B} GB) for FP16 staging"
    add "ram-too-small-for-7b"
fi

if [[ $IS_UNIFIED -eq 1 && $RAM_GB -lt $NEED_14B ]]; then
    bad "Unified-memory APU detected: BIOS UMA carve-out is starving system RAM"
    add "uma-carveout-too-large"
fi

# --- Swap sanity --------------------------------------------------------------
hdr "Swap"
if [[ $SWAP_GB -ge 32 ]]; then
    ok "${SWAP_GB} GB swap is generous enough to absorb spillover"
elif [[ $SWAP_GB -ge 16 ]]; then
    warn "${SWAP_GB} GB swap is OK but tight — recommend 64 GB for 14B QLoRA"
    add "swap-small"
else
    bad "${SWAP_GB} GB swap is too small to catch CPU-side spillover during model load"
    add "swap-too-small"
fi

# --- ROCm gfx version vs HSA_OVERRIDE -----------------------------------------
hdr "ROCm gfx target"
HSA_OVR="${HSA_OVERRIDE_GFX_VERSION:-}"
say "  HSA_OVERRIDE_GFX_VERSION = ${HSA_OVR:-<unset>}"
# Detect Strix Halo even when HSA_OVERRIDE has masked it as gfx1100.
# rocminfo's Marketing Name string is read from PCI device tables and
# isn't affected by HSA_OVERRIDE.
IS_STRIX_HALO=0
if echo "${GPU_NAME:-}" | grep -qiE "ryzen ai max|radeon (8060s|8050s)"; then
    IS_STRIX_HALO=1
fi

case "$GFX_REPORTED" in
    gfx1100|gfx1101|gfx1102)
        if [[ $IS_STRIX_HALO -eq 1 ]]; then
            warn "GPU is Strix Halo (real gfx=gfx1151) masked as $GFX_REPORTED via HSA_OVERRIDE"
            warn "→ Prebuilt bitsandbytes ROCm wheels typically lack gfx1151 kernels and throw 'hipErrorNoBinaryForGpu'"
            warn "→ HQQ's triton kernel JIT has been observed to segfault on this gfx during model load"
            warn "→ Recommended path on this hardware: train with --quantizer none (full-precision base + LoRA), 96 GB VRAM is plenty"
            add "strix-halo-quantizer-fragile"
        else
            ok "RDNA3 target $GFX_REPORTED is supported by ROCm bitsandbytes fork"
        fi
        ;;
    gfx1151|gfx1150)
        warn "Strix Halo $GFX_REPORTED detected directly — set HSA_OVERRIDE_GFX_VERSION=11.0.0 for most kernels"
        warn "Quantizer kernels (bnb/HQQ) are fragile on this gfx — prefer --quantizer none"
        [[ -z "$HSA_OVR" ]] && add "hsa-override-missing"
        add "strix-halo-quantizer-fragile"
        ;;
    gfx906|gfx908|gfx90a|gfx940|gfx941|gfx942)
        ok "CDNA / Vega $GFX_REPORTED is well-supported on ROCm"
        ;;
    "")
        bad "Could not read GPU gfx version"
        add "gfx-unknown"
        ;;
    *)
        warn "$GFX_REPORTED has limited ROCm bitsandbytes support — prefer HQQ or --quantizer none"
        add "gfx-fragile-for-bnb"
        ;;
esac

# --- Python training stack ----------------------------------------------------
hdr "Python training stack"
PY="${PYTHON:-python3}"
[[ -x .venv/bin/python ]] && PY=.venv/bin/python

probe() {
    "$PY" -c "import $1; import sys; print(getattr(sys.modules['$1'], '__version__', '?'))" 2>/dev/null
}

check_pkg() {
    local pkg="$1" ver
    ver=$(probe "$pkg")
    if [[ -n "$ver" ]]; then
        ok "$pkg $ver"
    else
        warn "$pkg not installed"
        add "missing-$pkg"
    fi
}

for pkg in torch transformers peft trl accelerate datasets bitsandbytes; do
    check_pkg "$pkg"
done

TORCH_HIP=$("$PY" -c "import torch; print(getattr(torch.version, 'hip', None) or '')" 2>/dev/null || true)
TORCH_CUDA=$("$PY" -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || true)
if [[ -n "$TORCH_HIP" ]]; then
    ok "torch built with ROCm/HIP $TORCH_HIP"
elif [[ "$TORCH_CUDA" == "True" ]]; then
    warn "torch built for CUDA — on this AMD machine you want the ROCm wheel"
    add "torch-not-rocm"
fi

if "$PY" -c "import bitsandbytes" 2>/dev/null; then
    BNB_OK=$("$PY" - <<'PYEOF' 2>/dev/null || echo "fail"
import bitsandbytes as bnb, torch
try:
    x = torch.randn(16, 16, device="cuda", dtype=torch.float16)
    q, s = bnb.functional.quantize_nf4(x)
    bnb.functional.dequantize_nf4(q, s)
    print("ok")
except Exception as e:
    print(f"fail: {type(e).__name__}: {e}")
PYEOF
)
    if [[ "$BNB_OK" == "ok" ]]; then
        ok "bitsandbytes NF4 quant/dequant works on this GPU"
    else
        bad "bitsandbytes import OK but kernels fail: $BNB_OK"
        add "bnb-kernels-broken"
    fi
fi

# --- Verdict ------------------------------------------------------------------
say ""
hdr "Verdict"
if [[ ${#FINDINGS[@]} -eq 0 ]]; then
    say "  ${GRN}No issues detected — QLoRA should run.${NC}"
    exit 0
fi

MAX_MODEL="14B"
for f in "${FINDINGS[@]}"; do
    case "$f" in
        ram-too-small-for-7b)  MAX_MODEL="3B";;
        ram-too-small-for-14b) [[ "$MAX_MODEL" == "14B" ]] && MAX_MODEL="7B";;
    esac
done

if [[ "$MAX_MODEL" == "14B" ]]; then
    say "  ${GRN}Stack is healthy enough to attempt 14B QLoRA.${NC}"
else
    say "  ${YEL}Out-of-the-box you can only train up to ${MAX_MODEL} models on this machine.${NC}"
    say "  ${YEL}With the workarounds below you can reach 14B.${NC}"
fi
say ""
say "Recommended next steps:"
for f in "${FINDINGS[@]}"; do
    case "$f" in
        ram-too-small-for-7b|ram-too-small-for-14b|uma-carveout-too-large)
            say "  • Use a pre-quantized 4-bit checkpoint (e.g. unsloth/* on HF) to skip FP16 CPU staging"
            say "  • Or reduce BIOS UMA frame buffer (free up system RAM) — see docs/qlora-training.md"
            ;;
        swap-small|swap-too-small)
            say "  • Run: sudo anvil train setup     # grows swap to 64 GB"
            ;;
        hsa-override-missing)
            say "  • Run: sudo anvil train setup     # writes HSA_OVERRIDE_GFX_VERSION"
            ;;
        bnb-kernels-broken|gfx-fragile-for-bnb)
            say "  • Switch quantizer to HQQ (drop-in for QLoRA, framework-agnostic on ROCm)"
            ;;
        strix-halo-quantizer-fragile)
            say "  • This GPU has 96 GB VRAM — train with --quantizer none (full-precision base + LoRA, no quantization fragility)"
            say "  • For 14B+ where quantization is required, build bitsandbytes from source for gfx1151"
            ;;
        torch-not-rocm|missing-torch|missing-bitsandbytes|missing-transformers|missing-peft|missing-trl|missing-accelerate|missing-datasets)
            say "  • Run: anvil train install        # installs the ROCm training stack"
            ;;
    esac
done | sort -u

exit 1
