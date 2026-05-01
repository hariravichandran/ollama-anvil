"""Evaluate a trained specialized-7B LoRA adapter against the untuned base.

Three checks, ordered cheapest to most expensive:

  1. **Format conformance** — does the model produce structured output?
     Counts headings, paragraph length, bullets, total length. Cheap,
     deterministic, catches catastrophic regressions.

  2. **Domain vocabulary** — does the output use your domain's terms at
     appropriate density? Customize `DOMAIN_VOCAB` for your domain
     (set it to an empty set to skip this signal).

  3. **Held-out perplexity** — token loss on a held-out slice of the
     corpus. Lower is better; useful for tracking progress between runs.

A baseline (untuned base model) is run alongside so we can see the
*delta* — that's the meaningful signal.

Usage:
    python recipes/specialized-7b/eval.py ./out/specialized-7b
    python recipes/specialized-7b/eval.py ./out/specialized-7b --eval-set ./data/heldout.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Customize: prompts that exercise your target distribution. Replace with
# 5–10 prompts that the tuned model should handle distinctively better
# than the base.
EVAL_PROMPTS = [
    "Write a structured passage on a topic of your choice. Use clear sectioning, "
    "rigorous prose, and the conventions appropriate to this domain.",
]

# Customize: domain vocabulary you expect the tuned model to use at higher
# density than the base. Empty set = skip this signal.
DOMAIN_VOCAB: set[str] = set()


def load_pair(adapter_path: Path):
    """Returns (model_with_adapter, tokenizer, base_model_for_baseline)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_id = "Qwen/Qwen2.5-7B-Instruct"
    cfg_path = adapter_path / "adapter_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        base_id = cfg.get("base_model_name_or_path", base_id)

    print(f"[eval] loading base {base_id}...")
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )

    print(f"[eval] attaching adapter from {adapter_path}...")
    tuned = PeftModel.from_pretrained(base, str(adapter_path))
    tuned.eval()

    print(f"[eval] loading baseline (untuned base) {base_id}...")
    baseline = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    baseline.eval()
    return tuned, tok, baseline


def generate(model, tok, prompt: str, max_new: int = 600) -> str:
    import torch
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def score_format(text: str) -> dict:
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    headings = re.findall(r"^#{1,4}\s+|^[A-Z][A-Za-z ]{4,}:$", text, flags=re.MULTILINE)
    bullets = sum(1 for ln in text.splitlines() if re.match(r"^\s*[-*]\s+", ln))
    long_para = sum(1 for p in paragraphs if len(p) >= 200)
    return {
        "chars":       len(text),
        "paragraphs":  len(paragraphs),
        "headings":    len(headings),
        "bullets":     bullets,
        "long_paras":  long_para,
        "format_score": min(1.0, (len(headings) * 0.15 + long_para * 0.10 + bullets * 0.02 +
                                   (1 if len(text) >= 800 else 0) * 0.20)),
    }


def score_vocab(text: str) -> dict:
    if not DOMAIN_VOCAB:
        return {"vocab_hits": 0, "vocab_density_per_100w": 0.0}
    low = text.lower()
    hits = sum(1 for w in DOMAIN_VOCAB if w in low)
    density = hits / max(1, len(text.split()) / 100)
    return {"vocab_hits": hits, "vocab_density_per_100w": round(density, 2)}


def heldout_perplexity(model, tok, jsonl_path: Path, max_n: int = 50) -> float:
    import math

    import torch
    losses, n = [], 0
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or n >= max_n:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("output") or row.get("text") or ""
            if len(text) < 100:
                continue
            ids = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
            with torch.no_grad():
                out = model(**ids, labels=ids["input_ids"])
            losses.append(out.loss.item())
            n += 1
    if not losses:
        return float("nan")
    return math.exp(sum(losses) / len(losses))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("adapter_path", type=Path,
                   help="Directory containing the trained LoRA adapter.")
    p.add_argument("--eval-set", type=Path, default=None,
                   help="Optional held-out JSONL (instruction/output) for perplexity.")
    p.add_argument("--out", type=Path, default=None,
                   help="Where to save the JSON eval report. Defaults to <adapter>/eval.json")
    args = p.parse_args()

    if not args.adapter_path.exists():
        sys.exit(f"adapter dir not found: {args.adapter_path}")

    tuned, tok, baseline = load_pair(args.adapter_path)

    report = {"prompts": [], "summary": {}}

    print("\n[eval] generating outputs (deterministic, max_new=600)...")
    for prompt in EVAL_PROMPTS:
        tuned_out = generate(tuned, tok, prompt)
        base_out  = generate(baseline, tok, prompt)
        entry = {
            "prompt": prompt,
            "tuned": {"text": tuned_out, **score_format(tuned_out), **score_vocab(tuned_out)},
            "base":  {"text": base_out,  **score_format(base_out),  **score_vocab(base_out)},
        }
        report["prompts"].append(entry)
        print(f"\n--- {prompt[:80]} ---")
        print(f"  tuned: format={entry['tuned']['format_score']:.2f}  "
              f"vocab_density={entry['tuned']['vocab_density_per_100w']:.2f}  "
              f"chars={entry['tuned']['chars']}")
        print(f"  base : format={entry['base']['format_score']:.2f}  "
              f"vocab_density={entry['base']['vocab_density_per_100w']:.2f}  "
              f"chars={entry['base']['chars']}")

    def avg(field, slot):
        return sum(p[slot][field] for p in report["prompts"]) / len(report["prompts"])
    report["summary"] = {
        "tuned_format":   round(avg("format_score", "tuned"), 3),
        "base_format":    round(avg("format_score", "base"),  3),
        "tuned_vocab_density": round(avg("vocab_density_per_100w", "tuned"), 2),
        "base_vocab_density":  round(avg("vocab_density_per_100w", "base"),  2),
    }

    if args.eval_set and args.eval_set.exists():
        print(f"\n[eval] computing held-out perplexity from {args.eval_set}...")
        report["summary"]["tuned_ppl"] = round(heldout_perplexity(tuned, tok, args.eval_set), 2)
        report["summary"]["base_ppl"]  = round(heldout_perplexity(baseline, tok, args.eval_set), 2)

    out_path = args.out or (args.adapter_path / "eval.json")
    out_path.write_text(json.dumps(report, indent=2))
    print("\n[eval] summary:")
    for k, v in report["summary"].items():
        print(f"  {k}: {v}")
    print(f"\n[eval] full report -> {out_path}")


if __name__ == "__main__":
    main()
