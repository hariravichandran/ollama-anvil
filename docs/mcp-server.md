# MCP Server — Local LLMs as Claude Code Tools

`anvil.mcp.ollama_server` bridges Claude Code (and other MCP-compatible
clients) to your local Ollama installation. Once wired up, Claude Code can
call local models as tools — offloading bulk work (corpus generation,
batch summarization, draft completions) to your GPU instead of burning
cloud API credits.

## What it exposes

| Tool | What it does |
|---|---|
| `ollama_generate` | Generate text with a local model |
| `ollama_chat` | Chat completion with a local model |
| `ollama_list_models` | List models cached locally |
| `ollama_list_running` | Show currently loaded models (GPU/CPU split) |
| `ollama_pull_model` | Pull a model from the Ollama registry |
| `ollama_embed` | Generate embeddings for text |

## Install

```bash
cd ~/Documents/GitHub/ollama-anvil
pip install -e .[mcp]   # the [mcp] extra brings in the `mcp` SDK
```

The `mcp` dependency is optional — only needed if you're running the MCP
server.

## Wire to Claude Code

The repo ships a `.mcp.json` at the project root:

```json
{
  "mcpServers": {
    "ollama": {
      "command": "python3",
      "args": ["-m", "anvil.mcp.ollama_server"],
      "env": { "OLLAMA_HOST": "http://localhost:11434" }
    }
  }
}
```

Two ways to enable it:

1. **Per-project** — launch Claude Code from the anvil directory; it
   automatically registers any `.mcp.json` at the working-dir root.
2. **Global** — copy the `mcpServers` block into `~/.claude/settings.json`
   so Claude Code picks it up from any cwd.

When Claude Code starts, the `ollama_*` tools become available alongside
the built-ins. Try:

```
> /tools
> ollama_list_running
> ollama_generate model="qwen2.5-coder:7b" prompt="Summarize this file: ..."
```

## Speculative decoding companion

Sibling module `anvil.llm.speculative` (also new — see
`anvil/llm/speculative.py`) provides three classes for accelerating local
inference on a slow target model via a fast draft:

- `SpeculativeDecoder` — logprob-verified speculation; identical output to
  the target alone at ~2-3.5× throughput (when target is bottlenecked, not
  draft)
- `DraftSelector` — best-of-N from a fast draft, target ranks the
  candidates; ~2.5× throughput, different but quality-controlled output
- `DraftRefiner` — fast draft, target refines for higher quality than
  either alone

See the class docstrings for usage. Speculative decoding only pays off on
the local hardware when the target is genuinely slow (<10 tok/s native)
and is not a reasoning-style model. For fast targets like MoE 80B+ at
30+ tok/s, the draft overhead exceeds the gain.

## Why this is separate from `anvil.mcp.manager`

`anvil.mcp.manager` is the **client side** — it manages OTHER MCP servers
(web search, etc.) that Claude Code consumes via anvil.

`anvil.mcp.ollama_server` is the **server side** — it's the MCP server
that exposes anvil's local Ollama integration to Claude Code (or any
other MCP client).

Both packages live under `anvil/mcp/` but serve opposite directions of
the MCP protocol.
