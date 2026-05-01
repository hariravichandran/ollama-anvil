#!/bin/bash
# Pre-flight checklist for a long QLoRA training run.
#
# Prefer invoking via 'anvil train preflight [--apply]'.
#
# Verifies the things that, if wrong, would cause the run to die mid-way
# (sometimes hours in). Read-only by default — fixes are printed as commands
# you can copy-paste. Pass --apply to let the script fix the safe ones
# (governor, sleep targets).
#
# Run AFTER 'anvil train install' has finished.

set -u

APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

RED='\033[0;31m'; YEL='\033[0;33m'; GRN='\033[0;32m'; CYN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "  ${GRN}OK${NC}    $*"; }
warn() { echo -e "  ${YEL}WARN${NC}  $*"; }
bad()  { echo -e "  ${RED}FAIL${NC}  $*"; }
hdr()  { echo -e "\n${CYN}== $* ==${NC}"; }

FAILS=0
fail() { bad "$1"; FAILS=$((FAILS + 1)); }

# --- 1. Python stack actually works -------------------------------------------
hdr "1. Python training stack"
PY=".venv/bin/python"
[[ -x "$PY" ]] || PY="python3"
HIP=$("$PY" -c "import torch; print(getattr(torch.version, 'hip', None) or '')" 2>/dev/null || true)
if [[ -n "$HIP" ]]; then
    ok "torch is a ROCm/HIP build (hip=$HIP)"
else
    fail "torch is NOT a ROCm build — re-run 'anvil train install'"
fi

if "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    ok "torch.cuda.is_available() == True"
else
    fail "torch.cuda.is_available() is False"
fi

for pkg in transformers peft trl accelerate datasets; do
    if "$PY" -c "import $pkg" 2>/dev/null; then
        ok "$pkg importable"
    else
        fail "$pkg import failed — pip install -e \".[training]\""
    fi
done

QUANT_OK=0
if "$PY" - <<'PYEOF' 2>/dev/null
import torch, bitsandbytes as bnb
x = torch.randn(16,16,device="cuda",dtype=torch.float16)
q,s = bnb.functional.quantize_nf4(x); bnb.functional.dequantize_nf4(q,s)
PYEOF
then
    ok "bitsandbytes NF4 kernels work"
    QUANT_OK=1
fi

if "$PY" -c "import hqq" 2>/dev/null; then
    ok "hqq importable (fallback quantizer)"
    QUANT_OK=1
fi

[[ $QUANT_OK -eq 0 ]] && fail "neither bitsandbytes nor hqq is usable — 'anvil train install' did not complete"

# --- 2. Disk space ------------------------------------------------------------
hdr "2. Disk space"
FREE_GB=$(df --output=avail -BG /home 2>/dev/null | tail -1 | tr -dc '0-9')
FREE_GB=${FREE_GB:-0}
if [[ $FREE_GB -ge 50 ]]; then
    ok "${FREE_GB} GB free in /home (need ~30 GB for base model + checkpoints + cache)"
elif [[ $FREE_GB -ge 30 ]]; then
    warn "${FREE_GB} GB free — tight but workable"
else
    fail "${FREE_GB} GB free — need at least 30 GB"
fi

# --- 3. Memory + swap state ---------------------------------------------------
hdr "3. Memory & swap"
RAM_GB=$(awk '/^MemTotal:/{print int($2/1024/1024)}' /proc/meminfo)
SWAP_GB=$(awk '/^SwapTotal:/{print int($2/1024/1024)}' /proc/meminfo)
[[ $SWAP_GB -ge 32 ]] && ok "${SWAP_GB} GB swap" || fail "${SWAP_GB} GB swap (need ≥ 32 GB; run 'sudo anvil train setup')"
ok "${RAM_GB} GB system RAM"

# --- 4. ROCm env + GPU --------------------------------------------------------
hdr "4. ROCm runtime"
if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
    ok "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
else
    fail "HSA_OVERRIDE_GFX_VERSION not set — log out & back in OR 'source /etc/profile.d/ollama-anvil-training.sh'"
fi

if [[ "${PYTORCH_HIP_ALLOC_CONF:-}" == *"expandable_segments"* ]]; then
    ok "PYTORCH_HIP_ALLOC_CONF includes expandable_segments"
else
    warn "PYTORCH_HIP_ALLOC_CONF=expandable_segments:True is not set — fragmentation risk over multi-hour runs"
fi

if "$PY" - <<'PYEOF' 2>/dev/null
import torch
x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
(x @ x).sum().item()
PYEOF
then
    ok "GPU matmul probe passed"
else
    fail "GPU matmul probe FAILED — ROCm runtime is broken; re-check rocminfo and HSA_OVERRIDE"
fi

# --- 5. CPU governor ----------------------------------------------------------
hdr "5. CPU governor (throughput)"
GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "?")
if [[ "$GOV" == "performance" ]]; then
    ok "governor=performance"
elif [[ $APPLY -eq 1 && $EUID -eq 0 ]]; then
    for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g"; done
    ok "governor switched to performance"
else
    warn "governor=$GOV — for ~25% more throughput run: sudo cpupower frequency-set -g performance"
fi

# --- 6. Sleep / suspend -------------------------------------------------------
hdr "6. Sleep / suspend"
MASKED=$(systemctl is-enabled sleep.target suspend.target hibernate.target hybrid-sleep.target 2>/dev/null | grep -c masked)
if [[ "$MASKED" -ge 4 ]]; then
    ok "all sleep targets masked"
elif [[ $APPLY -eq 1 && $EUID -eq 0 ]]; then
    systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target >/dev/null
    ok "sleep targets masked"
else
    warn "sleep targets not masked — long runs can be killed by idle suspend. Fix: sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target"
fi

# --- 7. HF Hub reachability ---------------------------------------------------
hdr "7. HF Hub reachability"
if curl -sSfI --max-time 10 https://huggingface.co/ >/dev/null 2>&1; then
    ok "huggingface.co reachable"
else
    fail "huggingface.co unreachable from this host — model download will fail"
fi

# --- 8. Model + dataset existence (cheap HEAD on the ones we'll pull) --------
hdr "8. Target model + dataset"
DEFAULT_MODEL="${PREFLIGHT_MODEL:-unsloth/Qwen2.5-7B-Instruct-bnb-4bit}"
DEFAULT_DS="${PREFLIGHT_DATASET:-tatsu-lab/alpaca}"
if curl -sSfI --max-time 10 "https://huggingface.co/$DEFAULT_MODEL" >/dev/null 2>&1; then
    ok "$DEFAULT_MODEL (model) reachable"
else
    fail "$DEFAULT_MODEL not reachable — check spelling or HF token"
fi
if curl -sSfI --max-time 10 "https://huggingface.co/datasets/$DEFAULT_DS" >/dev/null 2>&1; then
    ok "$DEFAULT_DS (dataset) reachable"
else
    fail "$DEFAULT_DS not reachable"
fi

# --- 9. amdgpu GTT (optional, mainly matters for 14B+) -----------------------
hdr "9. amdgpu module options"
if [[ -f /sys/module/amdgpu/parameters/gttsize ]]; then
    GTT=$(cat /sys/module/amdgpu/parameters/gttsize)
    [[ "$GTT" -ge 65536 ]] && ok "gttsize=${GTT} (>= 64 GB)" || warn "gttsize=${GTT} — for 14B headroom, regenerate initramfs and reboot"
else
    warn "/sys/module/amdgpu/parameters/gttsize not exposed (amdgpu likely built into the kernel image). For 7B this is fine. For 14B+: sudo update-initramfs -u && sudo reboot"
fi

# --- 10. Verdict --------------------------------------------------------------
echo
hdr "Verdict"
if [[ $FAILS -eq 0 ]]; then
    echo -e "  ${GRN}READY${NC} — proceed to smoke test:"
    echo "      anvil train run --model $DEFAULT_MODEL --dataset $DEFAULT_DS --output ./out/smoke --max-steps 30 --max-samples 200"
    exit 0
else
    echo -e "  ${RED}NOT READY${NC} — $FAILS hard failure(s) above. Fix before starting the long run."
    exit 1
fi
