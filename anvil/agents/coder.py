"""Coder agent: specialized for code editing, debugging, and project management."""

from __future__ import annotations

from pathlib import Path

from anvil.agents.base import AgentConfig, BaseAgent
from anvil.llm.client import OllamaClient
from anvil.tools import BUILTIN_TOOLS
from anvil.utils.logging import get_logger

log = get_logger("agents.coder")

__all__ = [
    "CODER_TEMPERATURE",
    "CODER_SYSTEM_PROMPT",
    "create_coder_agent",
    "MIN_TEMPERATURE",
    "MAX_TEMPERATURE",
]

# Temperature bounds for code generation
CODER_TEMPERATURE = 0.3
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0

CODER_SYSTEM_PROMPT = """\
You are an expert coding assistant running locally via Ollama. Your capabilities:

1. **Read files**: Use read_file to examine source code
2. **Edit files**: Use edit_file for precise string replacements
3. **Write files**: Use write_file to create new files
4. **Search code**: Use search_files to find patterns across the codebase
5. **Codebase search**: Use codebase_search and find_symbol to navigate the project
6. **Run commands**: Use run_command to build, test, lint, etc.
7. **Git operations**: Use git_status, git_diff, git_log, git_commit

Guidelines:
- Read code before modifying it. Understand context first.
- Use codebase_search and find_symbol to explore the project structure before diving in.
- Make minimal, focused changes. Don't refactor what you weren't asked to change.
- Prefer editing existing files over creating new ones.
- Always explain what you changed and why.
- Run tests after making changes when test infrastructure exists.
- Write clean, well-documented code following the project's existing style.
- When you encounter errors, debug systematically — don't guess randomly.
"""


def create_coder_agent(
    client: OllamaClient,
    working_dir: str = ".",
    temperature: float = CODER_TEMPERATURE,
) -> BaseAgent:
    """Create a coder agent with optimal settings for code tasks.

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
    requested_tools = ["filesystem", "shell", "git", "codebase", "diff", "image"]
    available_tools = [t for t in requested_tools if t in BUILTIN_TOOLS]
    if len(available_tools) < len(requested_tools):
        missing = set(requested_tools) - set(available_tools)
        log.warning("Coder agent: missing tools %s, using available: %s", missing, available_tools)

    config = AgentConfig(
        name="coder",
        system_prompt=CODER_SYSTEM_PROMPT,
        tools=available_tools,
        temperature=temperature,
        description="Coding assistant — writes, debugs, and refactors code",
    )
    return BaseAgent(client=client, config=config, working_dir=working_dir)
