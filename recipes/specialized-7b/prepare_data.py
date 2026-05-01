"""Build a specialized-7B training corpus from local files (+ optional HF datasets).

Output is Alpaca-style JSONL: rows are
    {"instruction": str, "input": str?, "output": str}
which the trainer (`anvil train run`) auto-formats into a `text` column.

Local file types handled:
  - `.jsonl` / `.json`  — accepts {instruction, output} | {prompt, response} | {text}
  - `.md`               — splits on `# ` top-level sections; each section becomes one example
  - `.txt`              — whole file as one example
  - `.csv`              — expects `instruction` + `output` columns (and optional `input`)
  - `.pdf`              — extracted via PyMuPDF, chunked at 4000 chars

Customize before training:
  - The instruction templates in `load_local_*` shape what the model learns.
    Edit them to match the voice/format/structure of your domain.
  - Add HF dataset IDs to `DEFAULT_HF_DATASETS` if you want to mix in
    public data alongside your private corpus.
  - For synthetic bootstrapping, fill `seed_prompts.jsonl` and pass
    `--synth-prompts` to generate (prompt, response) pairs via local Ollama.

Usage:
    python recipes/specialized-7b/prepare_data.py \
        --output ./data/specialized-v1.jsonl \
        --local-dir ~/path/to/your/corpus
        --hf-dataset some-org/some-dataset       # optional, repeatable
        --max-public 15000
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

# Default-empty: this template is domain-neutral. Add HF dataset IDs here
# (and the matching loader in `load_hf_dataset` below) if you want to mix
# in public data.
DEFAULT_HF_DATASETS: list[str] = []


def _emit(records, fout, count_dict, label):
    n = 0
    for rec in records:
        if not rec or not rec.get("output"):
            continue
        if not rec.get("instruction"):
            continue
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
    count_dict[label] = count_dict.get(label, 0) + n


def load_hf_dataset(ds_id: str, max_n: int):
    """Load an Alpaca-style HF dataset and yield {instruction, input, output} rows.

    Assumes the dataset has `instruction` and `output` columns. Extend this
    function (or add new loaders) for datasets with different schemas.
    """
    from datasets import load_dataset
    ds = load_dataset(ds_id, split="train")
    if max_n > 0:
        ds = ds.shuffle(seed=42).select(range(min(max_n, len(ds))))
    for row in ds:
        instr = (row.get("instruction") or "").strip()
        out = (row.get("output") or "").strip()
        if not instr or not out:
            continue
        yield {
            "instruction": instr,
            "input": (row.get("input") or "").strip(),
            "output": out,
        }


# ---------- Local files ------------------------------------------------------

def iter_local_files(root: Path):
    exts = {".jsonl", ".json", ".md", ".txt", ".csv", ".pdf"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and p.stat().st_size > 0:
            yield p, p.suffix.lower()


def load_local_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "instruction" in row and "output" in row:
                yield {"instruction": row["instruction"].strip(),
                       "input": (row.get("input") or "").strip(),
                       "output": row["output"].strip()}
            elif "prompt" in row and "response" in row:
                yield {"instruction": row["prompt"].strip(),
                       "input": "",
                       "output": row["response"].strip()}
            elif "text" in row:
                # Free text — wrap in a generic "continue this passage" instruction.
                # Customize this template to match your domain's voice.
                yield {"instruction": "Continue the following passage in the same voice and style.",
                       "input": row["text"].strip()[:1000],
                       "output": row["text"].strip()}


def load_local_markdown(path: Path):
    """Markdown → split on top-level `# ` headings; each section is one example."""
    text = path.read_text(errors="replace")
    sections = re.split(r"^# +", text, flags=re.MULTILINE)
    for sec in sections:
        sec = sec.strip()
        if len(sec) < 400:
            continue
        first_nl = sec.find("\n")
        if first_nl <= 0:
            continue
        title, body = sec[:first_nl].strip(), sec[first_nl:].strip()
        # Customize this instruction to match the voice you want to train.
        yield {
            "instruction": f"Write the section titled '{title}'. Match the voice, depth, and structure expected in this domain.",
            "input": "",
            "output": body[:4000],
        }


def load_local_txt(path: Path):
    body = path.read_text(errors="replace").strip()
    if len(body) < 400:
        return
    yield {
        "instruction": f"Reproduce a piece of writing on the topic suggested by the filename '{path.stem}', with the rigor and structure expected in this domain.",
        "input": "",
        "output": body[:6000],
    }


def load_local_pdf(path: Path):
    """Extract text from PDFs via PyMuPDF and chunk at 4000 chars."""
    try:
        import fitz  # pymupdf
    except ImportError:
        print(f"[prepare_data] pymupdf not installed; skipping {path.name} "
              f"(install: pip install pymupdf)", file=sys.stderr)
        return

    try:
        doc = fitz.open(path)
    except Exception as e:
        print(f"[prepare_data] failed to open PDF {path.name}: {e}", file=sys.stderr)
        return

    full = "\n".join(page.get_text() for page in doc)
    doc.close()

    full = re.sub(r"\n{3,}", "\n\n", full).strip()
    if len(full) < 400:
        return

    chunk_size = 4000
    title = path.stem.replace("_", " ").replace("-", " ")
    for i in range(0, len(full), chunk_size):
        chunk = full[i:i + chunk_size].strip()
        if len(chunk) < 400:
            continue
        yield {
            "instruction": f"Reproduce a section of a document on '{title}'. Match the voice, depth, and structure expected in this domain.",
            "input": "",
            "output": chunk,
        }


def load_local_csv(path: Path):
    """CSV: expects columns instruction[,input],output."""
    import csv
    with path.open(newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return
        cols = {c.lower(): c for c in reader.fieldnames}
        if "instruction" not in cols or "output" not in cols:
            return
        for row in reader:
            yield {
                "instruction": (row.get(cols["instruction"]) or "").strip(),
                "input":       (row.get(cols.get("input", ""), "") or "").strip(),
                "output":      (row.get(cols["output"]) or "").strip(),
            }


def load_local_dir(root: Path):
    if not root.exists():
        print(f"[prepare_data] --local-dir {root} does not exist; skipping.", file=sys.stderr)
        return
    for path, ext in iter_local_files(root):
        loader = {".jsonl": load_local_jsonl, ".json": load_local_jsonl,
                  ".md":    load_local_markdown,
                  ".txt":   load_local_txt,
                  ".csv":   load_local_csv,
                  ".pdf":   load_local_pdf}.get(ext)
        if loader is None:
            continue
        yield from loader(path)


# ---------- Optional: synthetic generation via local Ollama -----------------

def load_synthetic(seed_path: Path, ollama_model: str, max_n: int):
    """For each seed prompt, ask a local Ollama model for a draft response.

    Useful for bootstrapping when you don't have a large private corpus
    yet. Quality depends on the local model — a strong 14B+ model gives
    reasonable seeds the user can later curate.
    """
    try:
        import ollama  # type: ignore
    except ImportError:
        print("[prepare_data] 'ollama' python package not installed; skipping synthetic generation.",
              file=sys.stderr)
        return

    if not seed_path.exists():
        print(f"[prepare_data] seed file {seed_path} not found; skipping synthetic.", file=sys.stderr)
        return

    seeds = []
    with seed_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seeds.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    n = 0
    for seed in seeds:
        if n >= max_n:
            break
        instr = seed.get("instruction", "").strip()
        if not instr:
            continue
        try:
            resp = ollama.generate(model=ollama_model, prompt=instr,
                                   options={"temperature": 0.6, "num_predict": 1500})
            out = resp.get("response", "").strip()
        except Exception as e:
            print(f"[prepare_data] ollama generate failed for one seed: {e}", file=sys.stderr)
            continue
        if len(out) < 200:
            continue
        yield {"instruction": instr, "input": "", "output": out}
        n += 1


# ---------- Driver ----------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", required=True, help="Output JSONL path.")
    p.add_argument("--local-dir", type=Path, default=None,
                   help="Directory of your own files (jsonl/md/txt/csv/pdf).")
    p.add_argument("--hf-dataset", action="append", default=[],
                   help="HF dataset id, instruction-style (repeatable).")
    p.add_argument("--max-public", type=int, default=15000,
                   help="Cap rows per public source.")
    p.add_argument("--max-local", type=int, default=-1, help="Cap rows from local dir.")
    p.add_argument("--synth-prompts", type=Path, default=None,
                   help="Path to seed prompts JSONL. Triggers synthetic generation via Ollama.")
    p.add_argument("--ollama-model", default="qwen2.5:14b",
                   help="Local Ollama model to use for synthetic responses.")
    p.add_argument("--max-synth", type=int, default=200, help="Cap synthetic samples.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    counts = {}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as fout:
        # 1. Default HF datasets (none by default — this is a generic template).
        for ds_id in DEFAULT_HF_DATASETS:
            print(f"[prepare_data] loading default HF: {ds_id}")
            try:
                _emit(load_hf_dataset(ds_id, args.max_public), fout, counts, ds_id)
            except Exception as e:
                print(f"[prepare_data] {ds_id} failed: {e}; continuing.", file=sys.stderr)

        # 2. Extra HF datasets the user passed
        for extra in args.hf_dataset:
            print(f"[prepare_data] loading extra HF: {extra}")
            try:
                _emit(load_hf_dataset(extra, args.max_public), fout, counts, extra)
            except Exception as e:
                print(f"[prepare_data] {extra} failed: {e}; skipping.", file=sys.stderr)

        # 3. Local dir
        if args.local_dir:
            print(f"[prepare_data] loading local: {args.local_dir}")
            recs = load_local_dir(Path(args.local_dir).expanduser())
            if args.max_local > 0:
                def _capped():
                    n = 0
                    for r in recs:
                        if n >= args.max_local:
                            break
                        yield r
                        n += 1
                recs = _capped()
            _emit(recs, fout, counts, str(args.local_dir))

        # 4. Synthetic
        if args.synth_prompts:
            print(f"[prepare_data] generating synthetic via Ollama ({args.ollama_model})...")
            _emit(load_synthetic(args.synth_prompts, args.ollama_model, args.max_synth),
                  fout, counts, "synthetic")

    total = sum(counts.values())
    print(f"\n[prepare_data] wrote {total} rows -> {out_path}")
    for k, v in counts.items():
        print(f"  {v:>8}  {k}")


if __name__ == "__main__":
    main()
