"""Natural language interface for MCP management.

Allows users to add/remove MCPs via plain English:
  "add a web search tool"
  "I need to query a PostgreSQL database"
  "remove the GitHub integration"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anvil.mcp.registry import MCP_REGISTRY, search_registry, suggest_mcps
from anvil.utils.logging import get_logger

if TYPE_CHECKING:
    from anvil.mcp.manager import MCPManager

__all__ = [
    "parse_mcp_request",
    "handle_mcp_request",
    "MAX_REQUEST_TEXT_LENGTH",
    "MAX_MCP_LIST_DISPLAY",
    "MCP_NAME_COLUMN_WIDTH",
]

log = get_logger("mcp.natural_language")

# Input limits
MAX_REQUEST_TEXT_LENGTH = 500  # max NL request length
MAX_MCP_LIST_DISPLAY = 50  # max MCPs to display in list
MCP_NAME_COLUMN_WIDTH = 20  # formatting width for MCP names


def parse_mcp_request(text: str) -> dict[str, Any]:
    """Parse a natural language MCP request.

    Returns dict with:
      action: "add", "remove", "list", "search", "suggest", or None
      mcp_name: matched MCP name or None
      query: original text for fuzzy matching
    """
    if not text or not text.strip():
        return {"action": None, "mcp_name": None, "query": ""}
    if len(text) > MAX_REQUEST_TEXT_LENGTH:
        text = text[:MAX_REQUEST_TEXT_LENGTH]
    text_lower = text.lower().strip()

    # Detect action
    action = None
    if any(w in text_lower for w in ["add", "install", "enable", "set up", "i need", "i want"]):
        action = "add"
    elif any(w in text_lower for w in ["remove", "disable", "uninstall", "turn off"]):
        action = "remove"
    elif any(w in text_lower for w in ["list", "show", "what mcp", "which mcp"]):
        action = "list"
    elif any(w in text_lower for w in ["search", "find", "look for"]):
        action = "search"
    elif any(w in text_lower for w in ["suggest", "recommend", "what should"]):
        action = "suggest"

    # Try to match an MCP name
    matched_name = None
    for name, entry in MCP_REGISTRY.items():
        if name in text_lower or entry.name in text_lower:
            matched_name = name
            break

    # Fuzzy keyword matching if no exact name match
    if not matched_name and action in ("add", "remove"):
        keyword_map = {
            "web-search": ["web search", "search the web", "duckduckgo", "internet search"],
            "github": ["github", "git hub", "repository", "pull request"],
            "sqlite": ["sqlite", "database", "sql"],
            "postgres": ["postgres", "postgresql"],
            "puppeteer": ["browser", "screenshot", "web scraping"],
            "brave-search": ["brave search", "brave"],
            "slack": ["slack", "messaging"],
            "memory": ["memory", "remember", "knowledge graph"],
            "fetch": ["fetch url", "download page", "web page"],
        }
        for mcp_name, keywords in keyword_map.items():
            if any(kw in text_lower for kw in keywords):
                matched_name = mcp_name
                break

    return {
        "action": action,
        "mcp_name": matched_name,
        "query": text,
    }


def handle_mcp_request(manager: MCPManager, text: str) -> str:
    """Handle a natural language MCP request and return a response.

    This is the main entry point for NL-based MCP management.
    """
    if not text or not text.strip():
        return "Please provide an MCP request (e.g., 'add web search' or 'list MCPs')"
    parsed = parse_mcp_request(text)
    action = parsed["action"]
    name = parsed["mcp_name"]

    if action == "add":
        if name:
            return manager.enable(name)
        # Try to suggest based on context
        suggestions = suggest_mcps(text)
        if suggestions:
            names = ", ".join(s.name for s in suggestions)
            return f"Based on your request, you might want: {names}. Use 'anvil mcp add <name>' to enable."
        return "I couldn't determine which MCP you want. Try 'anvil mcp search <keyword>' to find one."

    if action == "remove":
        if name:
            return manager.disable(name)
        return "Please specify which MCP to remove. Use 'anvil mcp list' to see enabled MCPs."

    if action == "list":
        available = manager.list_available()
        lines = ["Available MCP servers:\n"]
        for mcp in available:
            status_icon = {"enabled": "[on]", "disabled": "[off]", "built-in": "[built-in]"}.get(mcp["status"], "[ ]")
            lines.append(f"  {status_icon} {mcp['name']:{MCP_NAME_COLUMN_WIDTH}s} {mcp['description']}")
        return "\n".join(lines)

    if action == "search":
        query = text.lower()
        for word in ["search", "find", "look for", "mcp"]:
            query = query.replace(word, "").strip()
        results = search_registry(query)
        if results:
            lines = [f"MCPs matching '{query}':\n"]
            for r in results:
                lines.append(f"  {r.name:{MCP_NAME_COLUMN_WIDTH}s} {r.description}")
            return "\n".join(lines)
        return f"No MCPs found matching '{query}'"

    if action == "suggest":
        suggestions = suggest_mcps(text)
        if suggestions:
            lines = ["Suggested MCPs:\n"]
            for s in suggestions:
                lines.append(f"  {s.name:{MCP_NAME_COLUMN_WIDTH}s} {s.description}")
            return "\n".join(lines)
        return "No specific suggestions for your current context. Try 'anvil mcp list' to see all options."

    return ""  # Not an MCP request
