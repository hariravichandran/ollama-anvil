# LoRA / QLoRA Training Guide

End-to-end guide for training LoRA / QLoRA adapters on AMD ROCm hardware,
with a special-case path for AMD Strix Halo (Ryzen AI MAX, gfx1151).

This page covers:
- The validated install path on Strix Halo (Bosgame M5)
- What to change for a different AMD GPU
- The four traps that bit us during validation, and how to avoid them
- Verified-working hyperparameters
- Troubleshooting per-symptom

---

## TL;DR — Easy install on a Strix Halo machine

```bash
# 1. Clone + venv
git clone https://github.com/hariravichandran/ollama-anvil
cd ollama-anvil
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

# 2. Base ollama-anvil + training extras (architecture-neutral HF stack)
pip install -e ".[training]"

# 3. System-level setup (sudo): grow swap, set ROCm env vars, amdgpu options.
#    Reboot when done.
sudo anvil train setup
sudo reboot

# 4. After reboot — install the gfx1151 torch + matching torchvision.
#    --strix-halo flag uses AMD's manylinux full-bundle wheel (5.6 GB
#    download, includes its own ROCm runtime so it runs regardless of
#    /opt/rocm version).
source .venv/bin/activate
anvil train install --strix-halo

# 5. Verify
anvil train diagnose
anvil train preflight
```

If `anvil train preflight` prints `READY`, you're done. If not, it prints
the exact remediation per failing check.

A single 60-step validation run on Qwen2.5-7B-Instruct + alpaca completes
in ~7 minutes on this hardware and saves a working adapter to
`./out/<run-name>/`. See [`recipes/specialized-7b/`](../recipes/specialized-7b/) for a complete recipe template.

---

## The four traps (and how this repo avoids them)

These are the exact issues we hit while validating training on the
Bosgame M5 (Strix Halo). They will bite anyone with similar hardware.

### Trap 1: pytorch.org's `rocm6.4` wheel has no gfx1151 kernels

`torch.cuda.get_arch_list()` for the standard wheel ends at `gfx1102, gfx1200`
— no gfx1151. Even `torch.randn(1024, 1024, device='cuda')` throws
`hipErrorNoBinaryForGpu: no kernel image is available for execution on the device`.

**`HSA_OVERRIDE_GFX_VERSION=11.0.0` does NOT fix this.** The override masks
the device at the runtime layer; it cannot synthesize missing kernel
binaries. You'll still crash on the first matmul.

**Fix:** install AMD's manylinux full-bundle torch from
`https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.2/torch-2.7.1+rocm7.2.2.git*-cp312-cp312-linux_x86_64.whl`
(the **non**-`.lw` variant — the `.lw` "lightweight" wheels need system
ROCm 7.x, which most users don't have).

This wheel:
- Includes gfx1151 in its arch list
- Bundles its own ROCm runtime (so it runs regardless of `/opt/rocm` version)
- Is ~5.6 GB
- Is what `anvil train install --strix-halo` installs for you

### Trap 2: bitsandbytes and HQQ are both fragile on gfx1151

- The pip ROCm `bitsandbytes` wheel ships without `libbitsandbytes_cpu.so`
  AND its compiled kernels lack a gfx1100/1151 binary →
  `hipErrorNoBinaryForGpu` again.
- HQQ uses Triton kernels; its JIT segfaults on first model load on
  RDNA 3.5 in our testing.

**Fix:** train with `--quantizer none` on this hardware. Full-precision
bf16 base + LoRA fits in ~14 GB VRAM for a 7B and ~28 GB for a 14B —
plenty in the 96 GB budget. It also trains *better* than QLoRA because
the base weights are not lossily quantized.

The trainer (`anvil train run`) supports
`--quantizer {bnb, hqq, none, auto}`; on Strix Halo you should always use
`none`.

### Trap 3: HF stack version mismatches

When AMD's torch 2.7.1 is in play, the HuggingFace stack must be pinned
to versions compatible with torch 2.7. The exact known-good combination:

| Package | Version | Notes |
|---|---|---|
| `torch` | `2.7.1+rocm7.2.2` | AMD manylinux wheel |
| `torchvision` | `0.22.1+rocm7.2.2` | Must match torch ABI |
| `transformers` | `4.46.3` | 5.x breaks under torch 2.7 (Bloom imports fail) |
| `peft` | `0.13.2` | Newer peft chases transformers 5 |
| `trl` | `0.12.2` | 1.x has its own breaking changes |
| `accelerate` | `1.1.1` | |
| `datasets` | `3.0.2` | Needs a one-line patch — see Trap 4 |
| `tokenizers` | `>=0.20,<0.21` | |
| `huggingface_hub` | `>=0.26,<1.0` | |
| `liger-kernel` | uninstalled | Requires transformers ≥ 4.52 |
| `bitsandbytes` | uninstalled | Broken on this hardware; not needed with `--quantizer none` |
| `hf_transfer` | latest | Speeds up HF Hub downloads when `HF_HUB_ENABLE_HF_TRANSFER=1` |

`pyproject.toml`'s `[training]` extras pin to this combination.

### Trap 4: `datasets/utils/_dill.py` NameError on dill 0.3.9+

`datasets 3.0.2` defines its `log()` helper only inside `if/elif` branches
for dill ≤ 0.3.8. On dill ≥ 0.3.9 (newer), `log` is undefined at module
scope, and trl's `SFTTrainer.train()` triggers it via `dataset.map()` with
a tokenizer in the closure.

**Fix:** add an `else: def log(pickler, msg): pass` branch to the file.
Also: capture `eos_token` as a plain string in `format_dataset()` rather
than closing over the tokenizer object — the trainer in
[`anvil/training/trainer.py`](../anvil/training/trainer.py) already does
this. To patch the datasets bug manually:

```bash
python3 - <<'PYEOF'
import importlib.util
spec = importlib.util.find_spec("datasets")
path = spec.submodule_search_locations[0] + "/utils/_dill.py"
src = open(path).read()
needle = 'def log(pickler, msg):\n        dill._dill.logger.trace(pickler, msg)\n\n\n@pklregister(set)'
patch  = 'def log(pickler, msg):\n        dill._dill.logger.trace(pickler, msg)\n\nelse:\n    def log(pickler, msg):\n        pass\n\n\n@pklregister(set)'
if needle in src:
    open(path, "w").write(src.replace(needle, patch))
    print("patched")
else:
    print("already patched or different version")
PYEOF
```

---

## Verified-working hyperparameters

Validated on the Bosgame M5 (Strix Halo, 96 GB VRAM, 30 GB system RAM,
71 GB swap):

### Smoke test (3 steps, ~10 sec)

```bash
anvil train run \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset tatsu-lab/alpaca \
    --output ./out/_smoke \
    --quantizer none \
    --max-steps 3 --max-samples 32 --save-steps 3 \
    --seq-len 1024 \
    --lora-r 16 --lora-alpha 32 \
    --batch-size 1 --grad-accum 1 \
    --lr 1e-4
```

**Result:** train_loss=1.45, ~3.3 sec/step, adapter saved.

### 30-minute validation run (60 steps, ~7 min)

```bash
anvil train run \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset data/<your-corpus>.jsonl \
    --output ./out/<name>-test \
    --quantizer none \
    --max-steps 60 --max-samples 500 --save-steps 30 \
    --seq-len 2048 \
    --lora-r 32 --lora-alpha 64 \
    --batch-size 1 --grad-accum 1 \
    --lr 1e-4
```

**Result:** 6.6 sec/step, train_loss 2.00 → 1.86, adapter +
checkpoint-30 saved. Total runtime: 6 min 35 sec.

### Full hyper-specialization run (overnight)

```bash
bash recipes/specialized-7b/train.sh data/<your-corpus>.jsonl
# Internally: --quantizer none, lora_r=64, lr=1e-4, epochs=3, seq_len=4096,
# grad_accum=16, save_steps=250, --resume
```

Estimated runtime for a ~50k-row corpus: 12–18 hours.

---

## Different AMD GPU?

| GPU | gfx target | What changes |
|---|---|---|
| Strix Halo (Ryzen AI MAX, Radeon 8060S/8050S) | gfx1151 | Use `--strix-halo` flag. |
| Phoenix2 (Ryzen 7040, Radeon 780M/760M) | gfx1150/1153 | Same `--strix-halo` flag — AMD's wheel includes gfx1150. |
| RX 7900 XTX/XT (RDNA3 discrete) | gfx1100 | Standard `pytorch.org/whl/rocm6.4` wheels work; drop the `--strix-halo` flag. bnb ROCm fork should work — try `--quantizer bnb`. |
| RX 6800/6900 (RDNA2) | gfx1030 | Standard rocm6.4 wheels work; expect slower throughput. |
| MI200/MI300 (CDNA datacenter) | gfx90a/gfx942 | Standard wheels; bnb works well. |

For non-Strix-Halo discrete cards, `--quantizer bnb` gets you genuine
4-bit QLoRA. On Strix Halo, stay with `--quantizer none` until upstream
fixes the kernel coverage.

---

## Hardware budget reference (7B Qwen2.5, FP16 base + LoRA)

For Strix Halo with 96 GB VRAM, 30 GB system RAM, 71 GB swap:

| Mode | VRAM | System RAM peak | Swap used | sec/step |
|---|---|---|---|---|
| Smoke (seq=1024, r=16, bs=1) | ~9 GB | ~7 GB | 0 | 3.3 |
| Validation (seq=2048, r=32, bs=1) | ~14 GB | ~8 GB | 0 | 6.6 |
| Full (seq=4096, r=64, bs=1, ga=16) | ~24 GB | ~10 GB | 0–4 GB | 100–180 (effective per opt step) |

The CPU-side load is the historically-OOM-prone moment when system RAM
is tight; loading `Qwen2.5-7B-Instruct` peaks at ~7 GB with
`low_cpu_mem_usage=True` — well under our 30 GB budget.

For 14B models, system RAM peak during load is ~22 GB. Sequence length
4096 + LoRA r=64 + batch_size=1 keeps VRAM under ~50 GB.

---

## Operations during a long run

**Watch progress** (from anywhere via SSH):
```bash
tail -50 /tmp/<run-name>.log         # latest training output
ls -t out/<run-name>/checkpoint-*/   # latest checkpoint
```

**Resume after a crash:** the trainer saves checkpoints every
`--save-steps` steps. Pass `--resume` to pick up from the latest one
in `--output`.

**Survive logout / SSH disconnect:** wrap the trainer in `nohup` (the
recipe's `train.sh` does this). Sleep targets are masked by
`anvil train preflight --apply` (run as sudo).

**Free VRAM after training:** `pkill -f anvil.training.trainer`. The
trainer releases VRAM cleanly on `Ctrl+C` if running in foreground.

---

## Troubleshooting

### `hipErrorNoBinaryForGpu` on first matmul (Strix Halo / gfx1151)

**Symptom:** `torch.AcceleratorError: HIP error: no kernel image is
available for execution on the device` — even on `torch.randn(...)`.

**Cause:** the `pytorch.org/whl/rocm6.4` torch wheel does not include
gfx1151 kernel binaries. `HSA_OVERRIDE_GFX_VERSION` masks the device at
the runtime layer but cannot synthesize missing kernels.

**Fix:** `anvil train install --strix-halo` installs AMD's full-bundle
wheel which includes gfx1151. See [Trap 1](#trap-1-pytorchorgs-rocm64-wheel-has-no-gfx1151-kernels).

### Trainer dies silently after model loads

**Symptom:** log shows model checkpoint shards loading 100%, then the
process disappears with no traceback.

**Cause:** most often HQQ's Triton kernel JIT segfaulting on RDNA 3.5.
Sometimes bitsandbytes' missing `.so` files when the ROCm wheel was
incomplete.

**Fix:** train with `--quantizer none` (full-precision base + LoRA).
The 7B Qwen2.5 base is only ~14 GB in bf16, well within Strix Halo's
96 GB VRAM budget. This is the recommended mode on this hardware.

### `NameError: name 'log' is not defined` from `datasets/_dill.py`

**Symptom:** traceback ends in `datasets/utils/_dill.py:209` with
`NameError: name 'log' is not defined`. Fires when SFTTrainer calls
`dataset.map()` with a tokenizer in scope.

**Cause:** bug in `datasets 3.0.2` — `log()` is only defined for
dill ≤ 0.3.8; on dill ≥ 0.3.9 it ends up undefined.

**Fix:** see the patch in [Trap 4](#trap-4-datasetsutils_dillpy-nameerror-on-dill-039). The
trainer in this repo already captures `eos_token` as a plain string to
avoid pickling the tokenizer object.

### `ValueError: model is quantized with BitsAndBytesConfig but you are passing a HqqConfig`

**Symptom:** when loading e.g. `unsloth/llama-3-8b-bnb-4bit` with the
auto-fallback HQQ path.

**Cause:** pre-quantized BnB checkpoints have an embedded
`quantization_config` that conflicts with the runtime HQQ config.

**Fix:** use a non-pre-quantized base model (e.g.
`Qwen/Qwen2.5-7B-Instruct`) with `--quantizer none`.

### Training process killed mid-run

**Symptom:** log stops, no traceback, process gone after hours.

**Likely causes:** idle suspend, OOM-killer when system RAM exhausted,
or SSH disconnect.

**Fixes:**
- `sudo anvil train preflight --apply` masks sleep targets.
- `sudo anvil train setup` grows swap to 64 GB.
- The trainer saves checkpoints every `--save-steps` steps. Pass
  `--resume` to pick up after a kill.
- Wrap the trainer in `nohup` (the recipe's `train.sh` does this).

### Verifying readiness before a long run

```bash
anvil train preflight    # exits 0 if READY
```

Checks HIP runtime, GPU matmul probe, swap inventory, sleep-target state,
HF Hub reachability, and a HEAD on the target model + dataset.

---

## Known TODOs / open issues

- `bitsandbytes` from source for gfx1151 — would unlock genuine 4-bit
  QLoRA for 14B+ on this hardware.
- `liger-kernel` compatibility with transformers 4.46 — pinned away
  for now.
- Pyproject `[training]` extras don't include `torch` itself (because it
  needs `--index-url` flags pip extras can't express). `anvil train install`
  does the torch step separately.
