"""Researcher agent: web search, information synthesis, and citation tracking."""

from __future__ import annotations

from pathlib import Path

from anvil.agents.base import AgentConfig, BaseAgent
from anvil.llm.client import OllamaClient
from anvil.tools import BUILTIN_TOOLS
from anvil.utils.logging import get_logger

log = get_logger("agents.researcher")

__all__ = [
    "RESEARCHER_TEMPERATURE",
    "RESEARCHER_SYSTEM_PROMPT",
    "create_researcher_agent",
    "MIN_TEMPERATURE",
    "MAX_TEMPERATURE",
]

# Temperature bounds for research
RESEARCHER_TEMPERATURE = 0.5
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0

RESEARCHER_SYSTEM_PROMPT = """\
You are a research assistant running locally via Ollama. Your capabilities:

1. **Web search**: Use web_search to find current information
2. **Fetch pages**: Use web_fetch to read full web pages
3. **Read files**: Use read_file to examine local documents
4. **Write files**: Use write_file to save research findings

Guidelines:
- Always search the web before answering questions about current events or recent topics.
- Cite your sources — include URLs and titles.
- Synthesize information from multiple sources when possible.
- Be transparent about uncertainty — if results are conflicting, say so.
- Organize findings clearly with headings and bullet points.
- Save important research to files when the user asks.
- Distinguish between facts, widely-held opinions, and speculation.
"""


def create_researcher_agent(
    client: OllamaClient,
    working_dir: str = ".",
    temperature: float = RESEARCHER_TEMPERATURE,
) -> BaseAgent:
    """Create a researcher agent with optimal settings for research tasks.

    Validates tool availability and working directory.
    """
    # Validate working directory
    wd = Path(working_dir)
    if not wd.exists():
        log.warning("Working directory does not exist: %s, using '.'", working_dir)
        working_dir = "."

    # Validate and clamp temperature
    temperature = max(MIN_TEMPERATURE, min(temperature, MAX_TEMPERATURE))

    # Validate tool availability
    requested_tools = ["web", "filesystem", "codebase"]
    available_tools = [t for t in requested_tools if t in BUILTIN_TOOLS]
    if len(available_tools) < len(requested_tools):
        missing = set(requested_tools) - set(available_tools)
        log.warning("Researcher agent: missing tools %s, using available: %s", missing, available_tools)

    config = AgentConfig(
        name="researcher",
        system_prompt=RESEARCHER_SYSTEM_PROMPT,
        tools=available_tools,
        temperature=temperature,
        description="Research assistant — searches web, summarizes findings",
    )
    return BaseAgent(client=client, config=config, working_dir=working_dir)
