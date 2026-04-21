# Hugging Face Models

Ollama's registry is great but it doesn't have everything. anvil lets you pull
GGUF models from Hugging Face and, when only safetensors are available, convert
them via llama.cpp.

## Three workflows

### 1. Pull a pre-made GGUF (fastest)

```bash
anvil models search-hf "qwen 7b gguf" --show-quants
anvil models pull-hf TheBloke/Llama-2-7B-GGUF --quant Q4_K_M
```

This downloads the chosen `.gguf` file and registers it with Ollama. You can
then:

```bash
anvil chat --model llama-2-7b-q4_k_m
```

### 2. Import a local GGUF

Already downloaded from somewhere else?

```bash
anvil models import ~/Downloads/my-model.Q5_K_M.gguf
anvil models import ~/Downloads/my-model.Q5_K_M.gguf --name my-custom-name
```

### 3. Convert from safetensors (full pipeline)

For models that don't have a GGUF on HF, anvil can do the full
safetensors → GGUF → quantize → import pipeline.

**One-time setup** (clones and builds llama.cpp):

```bash
pip install -e '.[convert]'         # Python-side deps
anvil models convert-setup          # clones llama.cpp + builds llama-quantize
```

This requires `cmake`, `make`, and a C++ compiler. On Debian/Ubuntu:

```bash
sudo apt install build-essential cmake
```

**Convert:**

```bash
anvil models convert meta-llama/Meta-Llama-3-8B --quant Q4_K_M
anvil models convert mistralai/Mistral-7B-v0.3 \
    --quant Q5_K_M \
    --name my-mistral
```

## Common quantizations

| Tag        | Size vs fp16 | Quality | Use case                          |
| ---------- | ------------ | ------- | --------------------------------- |
| Q2_K       | ~12%         | Poor    | Tiny boxes only                   |
| Q4_K_M     | ~25%         | Good    | **Default sweet spot**            |
| Q5_K_M     | ~30%         | Better  | When you have VRAM headroom       |
| Q6_K       | ~36%         | Great   | Quality-sensitive work            |
| Q8_0       | ~50%         | Near-fp16 | Max practical quality           |
| F16        | 100%         | Full    | Debugging / full-precision eval   |

## Troubleshooting

**"cmake not found" during convert-setup** — install build tools:

```bash
sudo apt install build-essential cmake          # Debian/Ubuntu
brew install cmake                              # macOS
```

**Conversion OOMs** — the unquantized fp16 intermediate can be large.
Make sure you have ~2× the model size free on disk and enough RAM to hold
one layer at a time (usually fine on 16+ GB systems).

**Model doesn't show up after import** — list what Ollama sees:

```bash
anvil models list
ollama list
```

If Ollama lists it but anvil doesn't (unlikely), restart the Ollama daemon.
