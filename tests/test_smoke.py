"""Smoke tests: imports, CLI help, module structure."""

from __future__ import annotations

import subprocess
import sys


def test_package_imports():
    import anvil  # noqa: F401
    from anvil.agents.base import BaseAgent  # noqa: F401
    from anvil.agents.orchestrator import AgentOrchestrator  # noqa: F401
    from anvil.community.ideas import IdeaCollector  # noqa: F401
    from anvil.community.self_improve import SelfImproveAgent  # noqa: F401
    from anvil.hardware import detect_hardware, select_profile  # noqa: F401
    from anvil.llm.client import OllamaClient  # noqa: F401
    from anvil.llm.context import ContextCompressor  # noqa: F401
    from anvil.llm.convert import convert_hf_repo, import_gguf  # noqa: F401
    from anvil.llm.huggingface import search_models  # noqa: F401
    from anvil.llm.rag import RAGPipeline  # noqa: F401
    from anvil.mcp.manager import MCPManager  # noqa: F401
    from anvil.tools import BUILTIN_TOOLS  # noqa: F401


def test_builtin_tools():
    from anvil.tools import BUILTIN_TOOLS

    expected = {"filesystem", "shell", "git", "web", "codebase", "diff", "handoff"}
    assert expected <= set(BUILTIN_TOOLS)


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "anvil.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0
    assert "ollama-anvil" in result.stdout
    for group in ("models", "mcp", "agent", "rag", "idea"):
        assert group in result.stdout


def test_hardware_detection():
    from anvil.hardware import detect_hardware, select_profile

    hw = detect_hardware()
    assert hw.cpu.cores >= 1
    assert hw.ram_gb > 0
    profile = select_profile(hw)
    assert profile.name
    assert profile.recommended_model


def test_agent_templates_exist():
    from pathlib import Path

    import anvil

    tmpl_dir = Path(anvil.__file__).parent / "agents" / "templates"
    for name in ("coder", "researcher", "writer"):
        assert (tmpl_dir / f"{name}.yaml").exists()
