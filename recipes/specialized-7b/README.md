# Specialized 7B — Recipe Template

A copy-and-customize template for training a hyper-specialized 7B LoRA
adapter on a narrow domain. Validated on Strix Halo (Bosgame M5).

## When to use this

Pick a 7B base + LoRA when you want a small, locally-runnable model that
beats a much larger generalist *on a specific narrow target* (a writing
voice, a domain vocabulary, a structured output format, a reasoning
pattern). For broad-ability tasks, stay with a generalist.

## What's in here

| File | Purpose |
|---|---|
| `prepare_data.py` | Build a JSONL training corpus from your local files (`.jsonl`/`.json`/`.md`/`.txt`/`.csv`/`.pdf`) and optional HF datasets. |
| `train.sh` | Wrapper that calls `anvil train run` with sensible hyper-specialization defaults (LoRA r=64, lr=1e-4, 3 epochs, seq=4096, step-checkpoints every 250 steps, auto-resume). |
| `eval.py` | Generic eval harness: format-conformance + held-out perplexity vs the untuned base. Add a domain-vocabulary list to `DOMAIN_VOCAB` for a third signal. |
| `seed_prompts.jsonl` | Empty by default — fill with your domain-specific seed prompts if you want synthetic generation via local Ollama. |

## Customize before training

1. Edit `prepare_data.py`:
   - Set `DEFAULT_HF_DATASETS` to any public datasets you want to mix in
     (or leave empty and rely entirely on your local corpus).
   - Adjust the markdown / txt / pdf instruction templates to match the
     voice you want the model to learn.
2. Edit `eval.py`:
   - Replace `DOMAIN_VOCAB` with terms from your domain (or leave empty
     to skip the vocab signal).
   - Replace `EVAL_PROMPTS` with prompts that exercise your target
     distribution.
3. Edit `train.sh`:
   - Change `MODEL` and `OUTPUT` to taste.
   - Tune `lora-r`, `seq-len`, and `grad-accum` for your VRAM budget.
4. Optionally add seed prompts to `seed_prompts.jsonl` for synthetic
   bootstrapping with a local Ollama model.

## Base model

`Qwen/Qwen2.5-7B-Instruct` (Apache-2.0). Strong reasoning + long-form
prose for its size. Loaded in **bfloat16 with LoRA adapters** rather than
QLoRA — on Strix Halo (96 GB VRAM) we have plenty of VRAM to skip
quantization, which also avoids the `hipErrorNoBinaryForGpu` and
triton-JIT issues that plague the RDNA 3.5 GPU's quantization kernel
paths.

Swap in a different base by setting `MODEL=...` before `train.sh`.

## Hardware target

- **Primary:** Bosgame M5 / Ryzen AI MAX+ 395 (Strix Halo, gfx1151) with
  30 GB system RAM and 96 GB VRAM after the BIOS UMA carve-out.
- Requires `torch 2.7.1+rocm7.2.2` (full-bundle wheel from AMD's
  manylinux index). The standard `rocm6.4` PyTorch wheel **does not
  contain gfx1151 kernels** and will crash on first matmul. See
  [`docs/qlora-training.md`](../../docs/qlora-training.md).
- Trains with `--quantizer none` (full-precision base + LoRA), which
  fits in ~28 GB VRAM and trains faster + better than QLoRA on this
  hardware.

## Quick start

After running `anvil train setup` and `anvil train install --strix-halo`
once on this host:

```bash
# 1. Build the training corpus from your local files
python recipes/specialized-7b/prepare_data.py \
    --output ./data/specialized-v1.jsonl \
    --local-dir ~/path/to/your/corpus

# 2. Kick off training
bash recipes/specialized-7b/train.sh ./data/specialized-v1.jsonl

# 3. Evaluate
python recipes/specialized-7b/eval.py ./out/specialized-7b
```

Training takes ~12–18 hours for the default 3-epoch run on a ~50k-sample
corpus; checkpoints land every 250 steps so a crash loses minutes, not
hours.

## Output

LoRA adapters in `./out/specialized-7b/`. To use the trained model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "./out/specialized-7b")
tok = AutoTokenizer.from_pretrained("./out/specialized-7b")
```

Or merge adapters into the base for serving:

```bash
python -c "from peft import PeftModel; from transformers import AutoModelForCausalLM; \
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', torch_dtype='bfloat16'); \
m = PeftModel.from_pretrained(m, './out/specialized-7b').merge_and_unload(); \
m.save_pretrained('./out/specialized-7b-merged')"
```

Then convert to GGUF for Ollama with `anvil models convert`.

## Why hyper-specialization

A 7B generalist can't compete with frontier-scale models on broad tasks.
But a 7B hyper-specialist with a tight target distribution can match or
beat much larger generalists *on its narrow target*. The pattern:
many small, focused models routed by an orchestrator, each cheap to run
locally. Good fits: a single writing voice, a fixed report format, a
domain vocabulary with stable conventions, a narrow reasoning pattern.
