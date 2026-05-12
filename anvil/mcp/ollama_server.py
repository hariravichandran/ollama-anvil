#!/usr/bin/env python3
"""Ollama MCP Server — expose local LLMs to Claude Code as tools.

Bridges Claude Code to local Ollama models via the Model Context Protocol.
Claude Code can delegate tasks to local LLMs (corpus generation, batch
processing, draft generation) instead of burning API tokens.

Available tools:
  - ollama_generate: Generate text with a local model
  - ollama_chat: Chat completion with a local model
  - ollama_list_models: List available models
  - ollama_list_running: Show currently loaded models
  - ollama_pull_model: Pull a model from the registry
  - ollama_embed: Generate embeddings for text

Usage (standalone):
  python -m anvil.mcp.ollama_server

Configuration in ~/.claude/settings.json:
  {
    "mcpServers": {
      "ollama": {
        "command": "python3",
        "args": ["-m", "anvil.mcp.ollama_server"],
        "cwd": "/path/to/ollama-anvil",
        "env": {
          "OLLAMA_HOST": "http://localhost:11434"
        }
      }
    }
  }
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import requests
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
TIMEOUT_GENERATE = 300
TIMEOUT_CHAT = 300
TIMEOUT_EMBED = 120
TIMEOUT_LIST = 10
TIMEOUT_PULL = 600

server = Server("ollama")


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool definitions
# ═══════════════════════════════════════════════════════════════════════════════


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Define available Ollama tools for Claude Code."""
    return [
        Tool(
            name="ollama_generate",
            description=(
                "Generate text using a local Ollama model. Use for corpus "
                "generation, batch processing, drafting, or any task that "
                "can be offloaded to a local GPU instead of using Claude API tokens."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model name (e.g., 'qwen2.5:14b', 'qwen2.5-coder:3b')",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Input prompt for generation",
                    },
                    "system": {
                        "type": "string",
                        "description": "System prompt (optional)",
                        "default": "",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum output tokens (default: 512)",
                        "default": 512,
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Temperature 0.0-1.0 (default: 0.7)",
                        "default": 0.7,
                    },
                },
                "required": ["model", "prompt"],
            },
        ),
        Tool(
            name="ollama_chat",
            description=(
                "Chat completion using a local Ollama model. Supports multi-turn "
                "conversations. Use for interactive tasks, code review by local "
                "model, or quality checks."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model name",
                    },
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": ["system", "user", "assistant"]},
                                "content": {"type": "string"},
                            },
                            "required": ["role", "content"],
                        },
                        "description": "Chat messages array",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum output tokens (default: 512)",
                        "default": 512,
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Temperature 0.0-1.0 (default: 0.7)",
                        "default": 0.7,
                    },
                },
                "required": ["model", "messages"],
            },
        ),
        Tool(
            name="ollama_list_models",
            description="List all locally available Ollama models with sizes.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="ollama_list_running",
            description="Show which models are currently loaded in GPU memory.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="ollama_pull_model",
            description="Pull/download a model from the Ollama registry.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model name to pull (e.g., 'qwen2.5:14b')",
                    },
                },
                "required": ["model"],
            },
        ),
        Tool(
            name="ollama_embed",
            description=(
                "Generate embeddings for text using a local model. Useful for "
                "RAG, semantic search, or similarity comparisons without API costs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Embedding model name",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to embed",
                    },
                },
                "required": ["model", "text"],
            },
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool handlers
# ═══════════════════════════════════════════════════════════════════════════════


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Dispatch tool calls to Ollama API."""
    try:
        if name == "ollama_generate":
            return _handle_generate(arguments)
        elif name == "ollama_chat":
            return _handle_chat(arguments)
        elif name == "ollama_list_models":
            return _handle_list_models()
        elif name == "ollama_list_running":
            return _handle_list_running()
        elif name == "ollama_pull_model":
            return _handle_pull_model(arguments)
        elif name == "ollama_embed":
            return _handle_embed(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except requests.ConnectionError:
        return [TextContent(
            type="text",
            text=f"Error: Cannot connect to Ollama at {OLLAMA_HOST}. Is it running?",
        )]
    except requests.Timeout:
        return [TextContent(type="text", text="Error: Ollama request timed out.")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {type(e).__name__}: {e}")]


def _handle_generate(args: dict[str, Any]) -> list[TextContent]:
    """Generate text with Ollama /api/generate."""
    payload: dict[str, Any] = {
        "model": args["model"],
        "prompt": args["prompt"],
        "stream": False,
        "options": {
            "temperature": args.get("temperature", 0.7),
            "num_predict": args.get("max_tokens", 512),
        },
        "keep_alive": "30m",
    }
    system = args.get("system", "")
    if system:
        payload["system"] = system

    r = requests.post(
        f"{OLLAMA_HOST}/api/generate", json=payload, timeout=TIMEOUT_GENERATE,
    )
    r.raise_for_status()
    data = r.json()

    response = data.get("response", "")
    eval_count = data.get("eval_count", 0)
    eval_duration = data.get("eval_duration", 0) / 1e9
    tok_s = eval_count / eval_duration if eval_duration > 0 else 0

    result = (
        f"{response}\n\n"
        f"---\n"
        f"Model: {args['model']} | Tokens: {eval_count} | "
        f"Speed: {tok_s:.1f} tok/s | Time: {eval_duration:.1f}s"
    )
    return [TextContent(type="text", text=result)]


def _handle_chat(args: dict[str, Any]) -> list[TextContent]:
    """Chat completion with Ollama /api/chat."""
    payload: dict[str, Any] = {
        "model": args["model"],
        "messages": args["messages"],
        "stream": False,
        "options": {
            "temperature": args.get("temperature", 0.7),
            "num_predict": args.get("max_tokens", 512),
        },
        "keep_alive": "30m",
    }

    r = requests.post(
        f"{OLLAMA_HOST}/api/chat", json=payload, timeout=TIMEOUT_CHAT,
    )
    r.raise_for_status()
    data = r.json()

    content = data.get("message", {}).get("content", "")
    eval_count = data.get("eval_count", 0)
    eval_duration = data.get("eval_duration", 0) / 1e9
    tok_s = eval_count / eval_duration if eval_duration > 0 else 0

    result = (
        f"{content}\n\n"
        f"---\n"
        f"Model: {args['model']} | Tokens: {eval_count} | "
        f"Speed: {tok_s:.1f} tok/s | Time: {eval_duration:.1f}s"
    )
    return [TextContent(type="text", text=result)]


def _handle_list_models() -> list[TextContent]:
    """List available models."""
    r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=TIMEOUT_LIST)
    r.raise_for_status()
    models = r.json().get("models", [])

    if not models:
        return [TextContent(type="text", text="No models found. Pull one with ollama_pull_model.")]

    lines = ["Available Ollama models:\n"]
    for m in models:
        name = m.get("name", "?")
        size_gb = m.get("size", 0) / (1024 ** 3)
        family = m.get("details", {}).get("family", "?")
        quant = m.get("details", {}).get("quantization_level", "?")
        lines.append(f"  {name:<35s} {size_gb:>5.1f} GB  {family}/{quant}")

    return [TextContent(type="text", text="\n".join(lines))]


def _handle_list_running() -> list[TextContent]:
    """List currently loaded models."""
    r = requests.get(f"{OLLAMA_HOST}/api/ps", timeout=TIMEOUT_LIST)
    r.raise_for_status()
    models = r.json().get("models", [])

    if not models:
        return [TextContent(type="text", text="No models currently loaded.")]

    lines = ["Currently loaded models:\n"]
    for m in models:
        name = m.get("name", "?")
        vram_gb = m.get("size_vram", 0) / (1024 ** 3)
        lines.append(f"  {name:<35s} {vram_gb:>5.1f} GB VRAM")

    return [TextContent(type="text", text="\n".join(lines))]


def _handle_pull_model(args: dict[str, Any]) -> list[TextContent]:
    """Pull a model from the registry."""
    model = args["model"]
    r = requests.post(
        f"{OLLAMA_HOST}/api/pull",
        json={"name": model, "stream": False},
        timeout=TIMEOUT_PULL,
    )
    r.raise_for_status()
    data = r.json()

    status = data.get("status", "unknown")
    if "error" in data:
        return [TextContent(type="text", text=f"Pull failed: {data['error']}")]
    return [TextContent(type="text", text=f"Successfully pulled {model}: {status}")]


def _handle_embed(args: dict[str, Any]) -> list[TextContent]:
    """Generate embeddings."""
    r = requests.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": args["model"], "input": args["text"]},
        timeout=TIMEOUT_EMBED,
    )
    r.raise_for_status()
    data = r.json()

    embeddings = data.get("embeddings", [])
    if not embeddings:
        return [TextContent(type="text", text="No embeddings returned.")]

    dim = len(embeddings[0]) if embeddings else 0
    preview = embeddings[0][:5] if embeddings else []
    return [TextContent(
        type="text",
        text=f"Embedding dimension: {dim}\nFirst 5 values: {preview}\n(full vector available)",
    )]


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    """Run the Ollama MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def run():
    """Sync entry point."""
    import asyncio
    asyncio.run(main())


if __name__ == "__main__":
    run()
