"""Model capability detection — introspect what a model can do.

Auto-detects whether a model supports vision, tool calling, thinking
(chain-of-thought), embeddings, and structured output by inspecting the
model's metadata via the Ollama API and known model patterns.

Usage::

    from anvil.llm.client import OllamaClient
    from anvil.llm.capabilities import detect_capabilities

    client = OllamaClient(model="llama3.2-vision:11b")
    caps = detect_capabilities(client)

    if caps.vision:
        print("Model supports image input!")
    if caps.thinking:
        print("Model supports chain-of-thought reasoning!")
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from anvil.llm.client import OllamaClient
from anvil.utils.logging import get_logger

log = get_logger("llm.capabilities")

__all__ = [
    "ModelCapabilities",
    "detect_capabilities",
    "VISION_PATTERNS",
    "THINKING_PATTERNS",
    "EMBEDDING_PATTERNS",
    "TOOL_SUPPORT_FAMILIES",
]

# Model name patterns that indicate specific capabilities
VISION_PATTERNS = re.compile(
    r"(vision|llava|bakllava|moondream|minicpm-v|cogvlm)",
    re.IGNORECASE,
)

THINKING_PATTERNS = re.compile(
    r"(deepseek-r1|qwq|o1|reasoning)",
    re.IGNORECASE,
)

EMBEDDING_PATTERNS = re.compile(
    r"(embed|embedding|nomic-embed|mxbai-embed|bge-|e5-|gte-)",
    re.IGNORECASE,
)

# Model families known to support tool/function calling
TOOL_SUPPORT_FAMILIES = frozenset({
    "llama3.1", "llama3.2", "llama3.3",
    "qwen2.5", "qwen2.5-coder", "qwen3",
    "mistral", "mixtral",
    "gemma3",
    "command-r",
    "firefunction",
    "hermes",
})


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Detected capabilities for a model.

    All fields default to False when unknown. Use detect_capabilities()
    to populate from the Ollama API.
    """

    model: str
    vision: bool = False
    tools: bool = False
    thinking: bool = False
    embedding: bool = False
    structured_output: bool = False
    system_prompt: bool = True  # almost all chat models support this
    context_window: int = 0
    parameter_count: str = ""

    def summary(self) -> str:
        """One-line capability summary."""
        caps = []
        if self.vision:
            caps.append("vision")
        if self.tools:
            caps.append("tools")
        if self.thinking:
            caps.append("thinking")
        if self.embedding:
            caps.append("embedding")
        if self.structured_output:
            caps.append("structured-output")
        cap_str = ", ".join(caps) if caps else "chat-only"
        ctx = f", ctx={self.context_window}" if self.context_window else ""
        return f"{self.model}: [{cap_str}]{ctx}"


def detect_capabilities(
    client: OllamaClient,
    model: str | None = None,
) -> ModelCapabilities:
    """Detect model capabilities by querying the Ollama API.

    Combines API introspection (model template, parameters) with
    known model name patterns for reliable detection.

    Args:
        client: OllamaClient instance.
        model: Model name to check. Defaults to client's current model.

    Returns ModelCapabilities with detected flags.
    """
    model_name = model or client.model

    # Query Ollama for model metadata
    model_info = client.show_model(model_name)

    # Start with name-based heuristics
    vision = bool(VISION_PATTERNS.search(model_name))
    thinking = bool(THINKING_PATTERNS.search(model_name))
    embedding = bool(EMBEDDING_PATTERNS.search(model_name))
    tools = _check_tool_support(model_name)
    structured_output = True  # Ollama supports format on all chat models
    context_window = 0
    parameter_count = ""

    if model_info:
        # Extract from model template
        template = model_info.get("template", "")
        _model_file = model_info.get("modelfile", "")

        # Vision: check for image placeholders in template
        if "{{ .Images }}" in template or "{{ if .Images }}" in template:
            vision = True

        # Tools: check for tool placeholders in template
        if "{{ .Tools }}" in template or "{{ if .Tools }}" in template:
            tools = True

        # Context window from parameters
        params = model_info.get("parameters", "")
        ctx_match = re.search(r"num_ctx\s+(\d+)", params)
        if ctx_match:
            context_window = int(ctx_match.group(1))

        # Parameter count from model details
        details = model_info.get("details", {})
        parameter_count = details.get("parameter_size", "")

        # Family-based detection from details
        family = details.get("family", "").lower()
        if family in ("llama", "qwen2", "gemma", "mistral"):
            structured_output = True

    # Embedding models typically don't support chat features
    if embedding:
        tools = False
        thinking = False
        structured_output = False

    caps = ModelCapabilities(
        model=model_name,
        vision=vision,
        tools=tools,
        thinking=thinking,
        embedding=embedding,
        structured_output=structured_output,
        context_window=context_window,
        parameter_count=parameter_count,
    )

    log.info("Detected capabilities: %s", caps.summary())
    return caps


def _check_tool_support(model_name: str) -> bool:
    """Check if a model name matches a known tool-supporting family."""
    # Extract base family (e.g., "llama3.1" from "llama3.1:8b")
    base = model_name.split(":")[0].lower()
    # Strip namespace (e.g., "library/llama3.1" -> "llama3.1")
    if "/" in base:
        base = base.split("/")[-1]

    return any(base.startswith(family) for family in TOOL_SUPPORT_FAMILIES)
