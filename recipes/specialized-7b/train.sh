#!/bin/bash
# Train a specialized 7B LoRA adapter on a prepared JSONL corpus.
#
# Usage:
#   bash recipes/specialized-7b/train.sh <corpus.jsonl> [extra trainer args...]
#
# Example:
#   bash recipes/specialized-7b/train.sh ./data/specialized-v1.jsonl
#
# Defaults:
#   - Base: Qwen/Qwen2.5-7B-Instruct (override with MODEL=...)
#   - Output: ./out/specialized-7b   (override with OUTPUT=...)
#   - Full-precision base (bf16) + LoRA (r=64) on 96 GB VRAM (Strix Halo)
#   - --quantizer none, because the bnb/HQQ paths are fragile on gfx1151
#   - Step-based checkpoints every 250 steps so a crash loses minutes, not hours
#   - --resume picks up the latest checkpoint in OUTPUT if one exists

set -euo pipefail

CORPUS="${1:-}"
shift || true

if [[ -z "$CORPUS" ]]; then
    echo "Usage: $0 <corpus.jsonl>"
    echo
    echo "Build a corpus first:"
    echo "  python recipes/specialized-7b/prepare_data.py \\"
    echo "    --output ./data/specialized-v1.jsonl \\"
    echo "    --local-dir ~/path/to/your/corpus"
    exit 2
fi

if [[ ! -f "$CORPUS" ]]; then
    echo "Error: corpus file not found: $CORPUS"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT="${OUTPUT:-./out/specialized-7b}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"

# Sensible HF-stack env (also written by 'anvil train setup', set here as
# a backup so a fresh shell still trains correctly).
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Pick token from common locations if not already set.
if [[ -z "${HF_TOKEN:-}" && -s ~/.cache/huggingface/token ]]; then
    HF_TOKEN="$(tr -d '[:space:]' < ~/.cache/huggingface/token)"
    export HF_TOKEN
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# Activate venv if present.
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

# Hyperparameters tuned for hyper-specialization on a narrow domain:
#   r=64           ample LoRA capacity for absorbing the new domain
#   alpha=128      conventional 2x ratio
#   lr=1e-4        lower than generic SFT (avoids forgetting general capabilities)
#   epochs=3       narrow domain benefits from extra passes
#   seq_len=4096   long-form outputs; drop to 2048 if VRAM gets tight
#   grad_accum=16  effective batch = 16 — stable gradient signal
#   save_steps=250 checkpoint every ~30 min on this hardware
exec anvil train run \
    --model "$MODEL" \
    --dataset "$CORPUS" \
    --output "$OUTPUT" \
    --quantizer none \
    --epochs 3 \
    --batch-size 1 \
    --grad-accum 16 \
    --lr 1e-4 \
    --seq-len 4096 \
    --lora-r 64 \
    --lora-alpha 128 \
    --lora-dropout 0.05 \
    --save-steps 250 \
    --resume \
    "$@"
