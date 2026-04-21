# ollama-anvil

**Batteries-included CLI for [Ollama](https://ollama.com).**

Ollama is great at running local models. `anvil` adds the pieces that make it
pleasant to use every day: hardware-aware setup, context compression, Hugging
Face model import, MCP web search, agents, and RAG — all from one `anvil`
command.

---

## What it does

| Ollama gives you           | Anvil adds                                                      |
| -------------------------- | --------------------------------------------------------------- |
| `ollama pull / run / rm`   | `anvil hardware` — detect GPU/VRAM/RAM, recommend models        |
| Model registry             | `anvil models search-hf / pull-hf / convert / import` for GGUF  |
| Bare chat                  | Context compression so long convos don't blow your context      |
| No tools                   | MCP web search on by default; filesystem/shell/git tools        |
| No agents                  | `coder`, `researcher`, `writer` agents + YAML templates         |
| No RAG                     | `anvil rag ingest / query` over local docs                      |
| No hardware tuning         | Auto ROCm / CUDA / Vulkan env setup                             |

---

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
anvil doctor              # verifies Ollama + hardware
anvil models recommend    # picks a model for your box
```

Requires Python 3.10+ and [Ollama](https://ollama.com/download) running locally.

---

## Quick start

```bash
anvil                             # interactive chat, auto-detects model
anvil ask "what is the GIL?"      # one-shot question
anvil run qwen3:8b                # pull + chat in one step

anvil chat --agent coder          # start with the coder agent
anvil chat --agent researcher     # web-search-enabled research agent

anvil hardware                    # show GPU/CPU/RAM + recommended profile
anvil doctor                      # diagnose setup issues
anvil models recommend            # pick a model for this box
```

### Hugging Face models (not in Ollama registry)

```bash
anvil models search-hf "qwen 7b gguf"
anvil models pull-hf TheBloke/Llama-2-7B-GGUF --quant Q4_K_M
anvil models import ~/Downloads/my-model.gguf

# Full safetensors -> GGUF conversion (needs one-time llama.cpp setup)
anvil models convert-setup
anvil models convert meta-llama/Meta-Llama-3-8B --quant Q4_K_M
```

### Agents

```bash
anvil agent list                  # built-ins + user-defined
anvil agent templates             # bundled: coder / researcher / writer
anvil agent install coder         # copy template into ./agents/
anvil agent create my-agent       # scratch YAML
```

### RAG over local docs

```bash
anvil rag ingest ./docs ./README.md
anvil rag query "how do I configure ROCm?"
anvil rag status
```

### MCP

```bash
anvil mcp list                    # active servers
anvil mcp search filesystem
anvil mcp add filesystem
```

### AI dev tools

```bash
anvil diff                        # AI code review of git diff
anvil explain src/foo.py
anvil commit                      # commit message from staged changes
anvil review src/foo.py
anvil refactor src/foo.py --pattern "split into smaller functions"
anvil doc src/foo.py
anvil test-gen src/foo.py
anvil security src/
anvil translate src/foo.py --to rust
anvil todos
```

---

## Key features

### Context compression
Long conversations are summarized as they approach the model's context limit,
preserving recent turns verbatim and collapsing older ones. See
[anvil/llm/context.py](anvil/llm/context.py).

### Hardware-aware model selection
`anvil hardware` detects GPU vendor (AMD / NVIDIA / Intel / Apple), VRAM, RAM, CPU,
and picks a profile (handheld → workstation → server). Recommended models fit.

### Hugging Face import
`search-hf` / `pull-hf` grab existing GGUFs; `convert` runs a full safetensors →
GGUF pipeline via llama.cpp (set up once with `anvil models convert-setup`).

### MCP + web search out of the box
Web-search MCP is on by default — agents can look things up without API keys
(DuckDuckGo). Add filesystem, git, and other MCPs à la carte.

### Lightweight agents
3 built-ins (`assistant`, `coder`, `researcher`) plus 3 YAML templates. Create
your own with `anvil agent create <name>` — just a YAML file.

### Community contributions (opt-in)
Anyone can help improve anvil:

```bash
anvil self-improve --enable       # opt in
anvil self-improve                # run one iteration — proposes PRs
```

Improvements go through GitHub PRs (requires `gh` CLI). Ideas can also be
submitted with `anvil idea submit "..."` (anonymous, opt-out via
`ANVIL_COMMUNITY_IDEAS=0`).

---

## Configuration

Copy `.env.example` to `.env` and edit. Common overrides:

```bash
OLLAMA_BASE_URL=http://localhost:11434
ANVIL_DEFAULT_MODEL=qwen2.5-coder:7b
ANVIL_WEB_SEARCH=1
ANVIL_LOG_LEVEL=INFO
HSA_OVERRIDE_GFX_VERSION=10.3.0      # AMD iGPU override
CUDA_VISIBLE_DEVICES=0
```

---

## License

[MIT](LICENSE).

## Contributing

PRs welcome. The self-improvement agent (`anvil self-improve`) also opens PRs
automatically when enabled.
