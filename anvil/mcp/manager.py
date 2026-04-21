"""MCP server lifecycle management — install, start, stop, configure."""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from anvil.mcp.registry import MCP_REGISTRY, MCPEntry
from anvil.mcp.web_search import WebSearchMCP
from anvil.utils.fileio import atomic_write
from anvil.utils.logging import get_logger

__all__ = [
    "MCPManager",
    "MCP_INSTALL_MAX_RETRIES",
    "MCP_INSTALL_TIMEOUT",
    "DANGEROUS_INSTALL_PATTERNS",
    "MCP_INSTALL_BACKOFF_BASE",
    "MCP_HEALTH_CHECK_TIMEOUT",
    "MCP_NAME_PATTERN",
    "MAX_MCP_NAME_LENGTH",
    "MAX_INSTALL_CMD_DISPLAY",
    "MAX_STDERR_DISPLAY",
]

log = get_logger("mcp.manager")

# Pre-compiled regex patterns
_UNSAFE_CHARS_RE = re.compile(r'[^a-zA-Z0-9_]')
_PACKAGE_NAME_RE = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_.]*$')

# Installation retry settings
MCP_INSTALL_MAX_RETRIES = 3
MCP_INSTALL_BACKOFF_BASE = 2.0  # seconds
MCP_INSTALL_TIMEOUT = 120  # seconds
MCP_HEALTH_CHECK_TIMEOUT = 5  # seconds

# Dangerous shell patterns in install commands
DANGEROUS_INSTALL_PATTERNS = re.compile(
    r"(&&|;|\||`|\$\(|>\s|<\s|rm\s|curl\s.*\|\s*sh|wget\s.*\|\s*sh)",
    re.IGNORECASE,
)

# Valid MCP name pattern
MCP_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,49}$")

# Maximum query/config sizes
MAX_MCP_NAME_LENGTH = 50
MAX_INSTALL_CMD_DISPLAY = 100  # truncation for install command in error messages
MAX_STDERR_DISPLAY = 100  # truncation for stderr in retry warnings


class MCPManager:
    """Manages MCP server lifecycle and configuration.

    Handles installation, startup, shutdown, and configuration of MCP servers.
    Web search is enabled by default.
    """

    def __init__(self, config_path: str = ""):
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = Path("~/.config/ollama-anvil/mcp.yaml").expanduser()

        self._config: dict[str, Any] = self._load_config()
        self._active_servers: dict[str, Any] = {}
        self._server_health: dict[str, bool] = {}  # tracks last known health status

        # Built-in web search — always available
        self.web_search = WebSearchMCP()

        # Ensure web search is enabled by default
        if "web-search" not in self._config:
            self._config["web-search"] = {"enabled": True}
            self._save_config()

    def __repr__(self) -> str:
        active = len(self._active_servers)
        configured = len(self._config)
        return f"MCPManager(configured={configured}, active={active})"

    def close(self) -> None:
        """Release resources and clear active server state."""
        self._active_servers.clear()
        self._server_health.clear()

    def __enter__(self) -> MCPManager:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def list_available(self) -> list[dict[str, Any]]:
        """List all known MCP servers with their status."""
        results = []
        for name, entry in MCP_REGISTRY.items():
            status = "built-in" if entry.builtin else "available"
            config = self._config.get(name, {})
            if config.get("enabled"):
                status = "enabled"
            elif name in self._config and not config.get("enabled", True):
                status = "disabled"

            results.append({
                "name": name,
                "description": entry.description,
                "category": entry.category,
                "status": status,
                "builtin": entry.builtin,
            })
        return results

    @staticmethod
    def validate_mcp_name(name: str) -> str:
        """Validate an MCP name. Returns error string or empty if valid."""
        if not name or not name.strip():
            return "MCP name cannot be empty"
        if len(name) > MAX_MCP_NAME_LENGTH:
            return f"MCP name too long ({len(name)} chars, max {MAX_MCP_NAME_LENGTH})"
        if not MCP_NAME_PATTERN.match(name):
            return f"Invalid MCP name: {name}"
        return ""

    @staticmethod
    def _validate_install_cmd(cmd: str) -> str:
        """Validate an install command for dangerous patterns. Returns error or empty."""
        if not cmd:
            return ""
        if DANGEROUS_INSTALL_PATTERNS.search(cmd):
            return f"Install command contains dangerous shell patterns: {cmd[:MAX_INSTALL_CMD_DISPLAY]}"
        return ""

    def enable(self, name: str, config: dict[str, Any] | None = None) -> str:
        """Enable an MCP server."""
        entry = MCP_REGISTRY.get(name)
        if not entry:
            return f"Unknown MCP: {name}. Use 'anvil mcp search' to find available MCPs."

        # Install if needed (with retries)
        if not entry.builtin and entry.install_cmd:
            # Validate install command
            cmd_err = self._validate_install_cmd(entry.install_cmd)
            if cmd_err:
                log.error("Refusing to install MCP '%s': %s", name, cmd_err)
                return f"Install refused: {cmd_err}"

            log.info("Installing MCP: %s", name)
            last_error = ""
            for attempt in range(MCP_INSTALL_MAX_RETRIES):
                try:
                    result = subprocess.run(
                        entry.install_cmd, shell=True, capture_output=True, text=True,
                        timeout=MCP_INSTALL_TIMEOUT,
                    )
                    if result.returncode == 0:
                        break
                    last_error = result.stderr
                    if attempt < MCP_INSTALL_MAX_RETRIES - 1:
                        delay = MCP_INSTALL_BACKOFF_BASE ** (attempt + 1)
                        log.warning(
                            "MCP install attempt %d failed, retrying in %.0fs: %s",
                            attempt + 1, delay, result.stderr[:MAX_STDERR_DISPLAY],
                        )
                        time.sleep(delay)
                except subprocess.TimeoutExpired:
                    last_error = "installation timed out"
                    if attempt < MCP_INSTALL_MAX_RETRIES - 1:
                        log.warning("MCP install timed out (attempt %d), retrying", attempt + 1)
                        continue
            else:
                return f"Failed to install {name} after {MCP_INSTALL_MAX_RETRIES} attempts: {last_error}"

        # Save config
        mcp_config = config or entry.config_example.copy()
        mcp_config["enabled"] = True
        self._config[name] = mcp_config
        self._save_config()

        log.info("Enabled MCP: %s", name)
        return f"MCP '{name}' enabled successfully"

    def disable(self, name: str) -> str:
        """Disable an MCP server."""
        if name in self._config:
            self._config[name]["enabled"] = False
            self._save_config()
            return f"MCP '{name}' disabled"
        return f"MCP '{name}' was not enabled"

    def get_enabled(self) -> list[str]:
        """Get list of enabled MCP names."""
        enabled = []
        for name, config in self._config.items():
            if config.get("enabled", False):
                enabled.append(name)
        return enabled

    def get_tools_for_agent(self) -> list[dict[str, Any]]:
        """Get all tool definitions from enabled MCPs."""
        tools = []

        for name in self.get_enabled():
            entry = MCP_REGISTRY.get(name)
            if not entry:
                continue

            if name == "web-search":
                # Built-in web search
                tools.append({
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web using DuckDuckGo",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "Search query"},
                            },
                            "required": ["query"],
                        },
                    },
                })
            else:
                # External MCP — expose a generic tool using the MCP name
                # Sanitize tool name to ensure valid Python identifier
                safe_name = _UNSAFE_CHARS_RE.sub('_', name)
                tools.append({
                    "type": "function",
                    "function": {
                        "name": f"mcp_{safe_name}",
                        "description": entry.description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "description": "Action to perform"},
                                "params": {"type": "string", "description": "Parameters as JSON string"},
                            },
                            "required": ["action"],
                        },
                    },
                })

        return tools

    def health_check(self) -> dict[str, bool]:
        """Check health of all enabled MCPs.

        Returns a dict mapping MCP name to health status (True=healthy).
        Built-in MCPs are always healthy. External MCPs are checked
        by verifying their binary/command exists.
        """
        results: dict[str, bool] = {}
        for name in self.get_enabled():
            entry = MCP_REGISTRY.get(name)
            if not entry:
                results[name] = False
                continue

            if entry.builtin:
                results[name] = True
            else:
                # Check if the start command's binary exists
                healthy = self._check_mcp_binary(entry)
                results[name] = healthy
                if not healthy:
                    log.warning("MCP '%s' health check failed", name)

        self._server_health = results
        return results

    @staticmethod
    def _check_mcp_binary(entry: MCPEntry) -> bool:
        """Check if an MCP's package binary is importable/available."""
        if not entry.package:
            return True  # No package means it's a built-in or not applicable
        # Try to check if the package is installed
        try:
            # Sanitize package name to prevent injection
            pkg_name = entry.package.replace('-', '_')
            if not _PACKAGE_NAME_RE.match(pkg_name):
                log.warning("Invalid package name for health check: %s", pkg_name)
                return False
            result = subprocess.run(
                [sys.executable, "-c", f"import importlib; importlib.import_module('{pkg_name}')"],
                capture_output=True, text=True, timeout=MCP_HEALTH_CHECK_TIMEOUT,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def _load_config(self) -> dict[str, Any]:
        if self.config_path.exists():
            try:
                import yaml
                with open(self.config_path, encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except (OSError, ValueError) as e:
                log.debug("Could not load MCP config: %s", e)
        return {}

    def _save_config(self) -> None:
        """Save MCP configuration to disk with atomic write."""
        try:
            import yaml
            content = yaml.dump(self._config, default_flow_style=False)
            atomic_write(self.config_path, content)
        except OSError as e:
            log.error("Failed to save MCP config: %s", e)
