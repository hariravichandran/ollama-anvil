"""QLoRA / LoRA fine-tuning for ROCm / unified-memory APUs.

Default settings are tuned for a Ryzen AI MAX (Strix Halo) machine with a
generous VRAM allocation but tight system RAM (the BIOS UMA carve-out
leaves ~30 GB to the OS). Same settings work on any ROCm 6.4 host — they
just become conservative.

Key design choices that make 7B and 14B fit on a 30 GB-RAM host:

1. Pre-quantized 4-bit checkpoint by default. Loading FP16 weights to
   CPU first peaks at ``2 * params`` bytes (28 GB for a 14B), which OOMs.
   The unsloth/* mirrors ship NF4 weights so CPU staging stays under
   ~10 GB even for 14B.
2. ``low_cpu_mem_usage=True`` + sharded safetensors streaming.
3. Paged 8-bit AdamW (bnb) when bnb is the active quantizer; plain
   ``adamw_torch`` otherwise.
4. Gradient checkpointing on (trades ~25% throughput for ~60% less
   activation VRAM).
5. HQQ as fallback when bnb's NF4 kernels misbehave on RDNA3.5.
6. ``--quantizer none`` path: full-precision base + LoRA, the recommended
   mode on Strix Halo where both bnb and HQQ have gfx1151 issues.
7. flash-attention-2 (ROCm fork) when present, else SDPA.

Invoke via ``anvil train run`` or directly: ``python -m anvil.training.trainer ...``.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass


@dataclass
class Args:
    model: str
    dataset: str
    output: str
    quantizer: str
    epochs: float
    batch_size: int
    grad_accum: int
    lr: float
    seq_len: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    seed: int
    save_steps: int
    max_steps: int
    max_samples: int
    resume: bool


def parse_args(argv: list[str] | None = None) -> Args:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="unsloth/llama-3-8b-bnb-4bit",
                   help="HF repo id. Prefer the 'unsloth/*-bnb-4bit' mirrors — they ship "
                        "pre-quantized weights and avoid the FP16 CPU staging that OOMs "
                        "on small-RAM unified-memory APUs.")
    p.add_argument("--dataset", default="tatsu-lab/alpaca",
                   help="HF dataset id, or a local .jsonl/.json/.csv file.")
    p.add_argument("--output", default="./out/qlora-run")
    p.add_argument("--quantizer", choices=["bnb", "hqq", "none", "auto"], default="auto",
                   help="auto: try bnb, fall back to HQQ on failure. "
                        "none: skip quantization, train on full-precision base "
                        "(use this when bnb and HQQ both misbehave on the GPU; "
                        "needs ~14 GB VRAM for 7B, fine on Strix Halo's 96 GB).")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Per-device train batch size. Keep at 1 for 14B on 30 GB RAM.")
    p.add_argument("--grad-accum", type=int, default=16,
                   help="Gradient accumulation. Effective batch = batch_size * grad_accum.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-steps", type=int, default=500,
                   help="Checkpoint cadence (in optimizer steps).")
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Cap total optimizer steps. -1 = use --epochs. Useful for smoke tests.")
    p.add_argument("--max-samples", type=int, default=-1,
                   help="Cap dataset size after formatting. -1 = use all.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the latest checkpoint in --output if one exists.")
    a = p.parse_args(argv)
    return Args(**vars(a))


def apply_rocm_env_defaults() -> None:
    """Set ROCm + HF env vars if the user hasn't set them already.

    Mirrors /etc/profile.d/ollama-anvil-training.sh so the trainer works
    even on a host where the system-level setup hasn't been run.
    """
    defaults = {
        "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
        # torch>=2.9 unified the allocator env var name; set both so older
        # runtimes still pick it up.
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "BITSANDBYTES_NOWELCOME": "1",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)


def pick_quantizer(name: str):
    """Return (label, quantization_config_or_None).

    name='none' skips quantization entirely — train LoRA on the full-precision
    base. Use this on hardware where both bnb and HQQ misbehave (Strix Halo,
    where HQQ's triton kernel JIT segfaults on first use). Costs more VRAM but
    trains better and is the most reliable path on 96 GB VRAM hosts.
    """
    import torch

    def _try_bnb():
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig
        x = torch.randn(16, 16, device="cuda", dtype=torch.float16)
        q, s = bnb.functional.quantize_nf4(x)
        bnb.functional.dequantize_nf4(q, s)
        return "bnb", BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_torch_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    def _try_hqq():
        from transformers import HqqConfig
        return "hqq", HqqConfig(nbits=4, group_size=64, axis=1)

    if name == "none":
        return "none", None
    if name == "bnb":
        return _try_bnb()
    if name == "hqq":
        return _try_hqq()
    # auto
    try:
        return _try_bnb()
    except Exception as e:  # noqa: BLE001 — any failure should fall through
        print(f"[trainer] bitsandbytes unusable ({type(e).__name__}: {e}); falling back to HQQ.")
        return _try_hqq()


def pick_attention_impl() -> str:
    """flash-attn-2 if importable, else SDPA. Both are VRAM-efficient."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def format_dataset(ds, tokenizer, seq_len: int):
    """Convert common instruction-tuning schemas to a single 'text' column.

    Handles three cases:
      - Already has 'text'              → pass through
      - {instruction, input?, output}   → Alpaca-style template
      - {prompt, response}              → simple concatenation
    """
    cols = set(ds.column_names)
    if "text" in cols:
        return ds

    # Capture eos as a plain string. If we close over `tokenizer`, datasets'
    # caching tries to pickle the tokenizer object and hits a known
    # NameError ('log' undefined) in datasets/utils/_dill.py for some versions.
    eos = tokenizer.eos_token or ""

    if {"instruction", "output"}.issubset(cols):
        def _to_text(row):
            instr = row["instruction"]
            inp = row.get("input") or ""
            out = row["output"]
            if inp:
                txt = (f"### Instruction:\n{instr}\n\n"
                       f"### Input:\n{inp}\n\n"
                       f"### Response:\n{out}{eos}")
            else:
                txt = (f"### Instruction:\n{instr}\n\n"
                       f"### Response:\n{out}{eos}")
            return {"text": txt}
        return ds.map(_to_text, remove_columns=list(cols))

    if {"prompt", "response"}.issubset(cols):
        return ds.map(lambda r: {"text": r["prompt"] + r["response"] + eos},
                      remove_columns=list(cols))

    raise ValueError(f"Don't know how to format dataset with columns {cols!r} — "
                     "add a 'text' column or extend format_dataset().")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    apply_rocm_env_defaults()

    # Lazy imports so --help doesn't pay the torch import cost (and so the
    # CLI's `anvil train --help` works on installs without [training] extras).
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        print("ERROR: torch reports no CUDA/HIP device. Run 'anvil train diagnose'.",
              file=sys.stderr)
        return 2

    quant_name, quant_cfg = pick_quantizer(args.quantizer)
    attn_impl = pick_attention_impl()
    print(f"[trainer] quantizer={quant_name}  attention={attn_impl}  "
          f"vram={torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB")

    print(f"[trainer] loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # low_cpu_mem_usage + device_map='auto' streams shards directly to GPU
    # without materializing the full FP16 model in CPU RAM. Critical when
    # system RAM is tight.
    print(f"[trainer] loading model {args.model} (OOM-prone step on small-RAM hosts)...")
    load_kwargs = dict(
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )
    if quant_cfg is not None:
        load_kwargs["quantization_config"] = quant_cfg
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.config.use_cache = False
    if quant_name in ("bnb", "hqq"):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        # Full-precision base. Enable gradient checkpointing manually.
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    print(f"[trainer] loading dataset {args.dataset}...")
    if os.path.isfile(args.dataset):
        ext = os.path.splitext(args.dataset)[1].lower()
        loader = "json" if ext in (".jsonl", ".json") else (
                 "csv"  if ext == ".csv" else None)
        if loader is None:
            sys.exit(f"Don't know how to load {args.dataset} (extension {ext}); "
                     "use .jsonl/.json/.csv or pass an HF dataset id.")
        raw = load_dataset(loader, data_files=args.dataset, split="train")
    else:
        raw = load_dataset(args.dataset, split="train")
    ds = format_dataset(raw, tokenizer, args.seq_len)
    if args.max_samples > 0 and len(ds) > args.max_samples:
        ds = ds.select(range(args.max_samples))
        print(f"[trainer] capped dataset to {len(ds)} samples (--max-samples)")

    # Paged 8-bit AdamW only works with bnb. Otherwise plain torch AdamW.
    optim = "paged_adamw_8bit" if quant_name == "bnb" else "adamw_torch"

    cfg = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=optim,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        max_seq_length=args.seq_len,
        packing=True,
        dataset_text_field="text",
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=["tensorboard"],
        seed=args.seed,
        # Single dataloader worker keeps RSS small on 30 GB-RAM hosts.
        dataloader_num_workers=1,
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        peft_config=lora,
        tokenizer=tokenizer,
    )

    # Auto-resume: if --resume and a checkpoint exists in output_dir, pick it up.
    resume = False
    if args.resume:
        ckpts = sorted(
            (p for p in os.listdir(args.output) if p.startswith("checkpoint-")),
            key=lambda p: int(p.split("-")[1]),
        ) if os.path.isdir(args.output) else []
        if ckpts:
            resume = os.path.join(args.output, ckpts[-1])
            print(f"[trainer] resuming from {resume}")
        else:
            print("[trainer] --resume passed but no checkpoint found; starting fresh")

    print(f"[trainer] starting training -> {args.output}")
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_model(args.output)
    print(f"[trainer] done. Adapter saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
