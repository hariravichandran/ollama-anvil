"""MCP (Model Context Protocol) integration for ollama-anvil."""

from anvil.mcp.manager import MCPManager
from anvil.mcp.registry import MCP_REGISTRY
from anvil.mcp.web_search import WebSearchMCP

__all__ = ["MCPManager", "MCP_REGISTRY", "WebSearchMCP"]
