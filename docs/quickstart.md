# Quickstart

Five minutes from zero to chatting with a local model.

## 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

Linux, macOS, and Windows (WSL) are supported. Leave Ollama running in the
background; anvil talks to it over `http://localhost:11434`.

## 2. Install anvil

```bash
git clone https://github.com/hariravichandran/ollama-anvil
cd ollama-anvil
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For the Textual terminal UI: `pip install -e '.[tui]'`.
For Hugging Face conversion: `pip install -e '.[convert]'`.

## 3. Verify

```bash
anvil doctor
```

All checks should be green. If Ollama isn't reachable, start it: `ollama serve`.

## 4. Pick a model

```bash
anvil hardware            # see your profile
anvil models recommend    # see suggested models for that profile
anvil models pull qwen2.5-coder:7b
```

## 5. Chat

```bash
anvil                                 # interactive, default agent
anvil ask "explain the GIL in 3 lines"
anvil chat --agent coder              # coder agent, tools enabled
anvil chat --agent researcher         # web-search-enabled
```

## Next

- **Hardware tuning** — see [hardware-guide.md](hardware-guide.md)
- **HF / external models** — see [hf-models.md](hf-models.md)
- **Agents** — `anvil agent templates` and `anvil agent create <name>`
- **RAG** — `anvil rag ingest ./docs && anvil rag query "..."`
