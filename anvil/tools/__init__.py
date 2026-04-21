"""Built-in tools for agents: filesystem, shell, git, web, codebase, diff, handoff."""

from anvil.tools.codebase import CodebaseTool
from anvil.tools.diff import DiffTool
from anvil.tools.filesystem import FilesystemTool
from anvil.tools.git import GitTool
from anvil.tools.handoff import HandoffTool
from anvil.tools.shell import ShellTool
from anvil.tools.web import WebTool

BUILTIN_TOOLS: dict[str, type] = {
    "filesystem": FilesystemTool,
    "shell": ShellTool,
    "git": GitTool,
    "web": WebTool,
    "codebase": CodebaseTool,
    "diff": DiffTool,
    "handoff": HandoffTool,
}

__all__ = [
    "FilesystemTool", "ShellTool", "GitTool", "WebTool",
    "CodebaseTool", "DiffTool", "HandoffTool", "BUILTIN_TOOLS",
]
