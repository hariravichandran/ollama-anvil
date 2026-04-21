"""ollama-anvil CLI — main entry point."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """ollama-anvil: Batteries-included CLI for Ollama.

    \b
    Quick start:
        anvil                         Start chatting (auto-detect model)
        anvil run qwen3:8b            Pull + chat in one step
        anvil ask "what is Python"    One-shot question

    \b
    Key groups:
        models / mcp / agent / rag / idea
        hardware / doctor / version / env
        chat / ui / tui / self-improve

    \b
    Dev tools:
        diff / explain / commit / review / refactor / doc / test-gen / security

    Run 'anvil <command> --help' for details.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(chat)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _get_client():
    from anvil.llm.client import OllamaClient

    return OllamaClient()


def _default_model() -> str:
    """Pick a default model: env override, then hardware recommendation, then first installed."""
    if env_model := os.environ.get("ANVIL_DEFAULT_MODEL"):
        return env_model
    try:
        client = _get_client()
        installed = [m["name"] for m in client.list_models()]
        if not installed:
            return ""
        # Prefer a chat-capable small model if present
        for preferred in ("qwen2.5-coder:7b", "qwen3:8b", "llama3.2:3b", "llama3.1:8b"):
            if preferred in installed:
                return preferred
        return installed[0]
    except Exception:
        return ""


def _run_prompt(model: str, prompt: str, system: str = "", stream: bool = True) -> str:
    """Run a single non-tool prompt through Ollama. Returns full response text."""
    client = _get_client()
    model = model or _default_model()
    if not model:
        console.print("[red]No model available.[/red] Run: [cyan]anvil models pull <name>[/cyan]")
        sys.exit(1)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    full = ""
    for chunk in client.chat(model=model, messages=messages, stream=stream):
        text = chunk.get("message", {}).get("content", "")
        if stream:
            console.print(text, end="")
        full += text
    if stream:
        console.print()
    return full


# ─── Chat ────────────────────────────────────────────────────────────────────


@main.command()
@click.option("--model", "-m", default="", help="Model to use (auto-detected if empty)")
@click.option("--agent", "-a", default="assistant", help="Agent: assistant, coder, researcher")
@click.option("--working-dir", "-d", default=".", help="Working directory for file operations")
@click.option("--auto-approve", is_flag=True, help="Auto-approve tool actions (skip prompts)")
@click.option("--oneshot", "-1", "oneshot_prompt", default="", help="Ask one question and exit")
@click.option("--system", "system_prompt", default="", help="Custom system prompt")
@click.option("--no-stream", is_flag=True, help="Disable streaming output")
def chat(model: str, agent: str, working_dir: str, auto_approve: bool,
         oneshot_prompt: str, system_prompt: str, no_stream: bool):
    """Interactive chat with an agent (default command)."""
    from anvil.agents.orchestrator import AgentOrchestrator
    from anvil.agents.permissions import PermissionManager

    client = _get_client()
    model = model or _default_model()
    if not model:
        console.print("[red]No model installed.[/red] Run: [cyan]anvil models recommend[/cyan]")
        sys.exit(1)

    orch = AgentOrchestrator(client=client, working_dir=working_dir)
    if agent not in orch.agents:
        console.print(f"[red]Unknown agent: {agent}[/red]. Available: {', '.join(orch.agents)}")
        sys.exit(1)
    orch.active_agent = agent

    agent_obj = orch.agents[agent]
    agent_obj.config.model = model
    if system_prompt:
        agent_obj.config.system_prompt = system_prompt
    agent_obj.permissions = PermissionManager(auto_approve_all=auto_approve)

    console.print(Panel.fit(
        f"[bold]anvil chat[/bold] — model: [cyan]{model}[/cyan]  agent: [green]{agent}[/green]",
        border_style="blue",
    ))

    if oneshot_prompt:
        for chunk in agent_obj.run(oneshot_prompt, stream=not no_stream):
            console.print(chunk, end="")
        console.print()
        return

    console.print("[dim]Type /quit to exit. /agent <name> to switch.[/dim]\n")
    while True:
        try:
            user_input = click.prompt(f"[{orch.active_agent}]", prompt_suffix="> ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break
        if user_input.strip() in ("/quit", "/exit", "/q"):
            break
        if user_input.startswith("/agent "):
            new_agent = user_input.split(None, 1)[1].strip()
            if new_agent in orch.agents:
                orch.active_agent = new_agent
                console.print(f"[dim]Switched to {new_agent}[/dim]")
            else:
                console.print(f"[red]Unknown agent.[/red] Available: {', '.join(orch.agents)}")
            continue
        current = orch.agents[orch.active_agent]
        for chunk in current.run(user_input, stream=not no_stream):
            console.print(chunk, end="")
        console.print()


@main.command()
@click.argument("question", nargs=-1, required=True)
@click.option("--model", "-m", default="", help="Model (auto-detected if empty)")
def ask(question: tuple, model: str):
    """Ask a one-shot question — no agent, no tools, just LLM."""
    q = " ".join(question)
    _run_prompt(model, q, system="You are a concise, helpful assistant. Answer directly.")


@main.command()
@click.argument("model_name")
@click.argument("prompt", required=False, default="")
def run(model_name: str, prompt: str):
    """Pull a model if needed, then chat with it (Ollama-style shortcut)."""
    client = _get_client()
    installed = {m["name"] for m in client.list_models()}
    if model_name not in installed:
        console.print(f"[yellow]Pulling {model_name}...[/yellow]")
        for status in client.pull(model_name):
            console.print(f"  {status}", end="\r")
        console.print()
    if prompt:
        _run_prompt(model_name, prompt)
    else:
        ctx = click.get_current_context()
        ctx.invoke(chat, model=model_name)


@main.command()
@click.option("--model", "-m", default="", help="Model to use")
@click.option("--agent", "-a", default="assistant", help="Agent to use")
@click.option("--working-dir", "-d", default=".")
def tui(model: str, agent: str, working_dir: str):
    """Launch the Textual terminal UI."""
    try:
        from anvil.ui.terminal import launch_tui
    except ImportError:
        console.print("[red]TUI requires textual.[/red] Install: [cyan]pip install 'ollama-anvil[tui]'[/cyan]")
        sys.exit(1)
    launch_tui(model=model or _default_model(), agent=agent, working_dir=working_dir)


# ─── Hardware / doctor / version / env ───────────────────────────────────────


@main.command()
def hardware():
    """Show detected hardware and recommended profile."""
    from anvil.hardware import detect_hardware, select_profile

    hw = detect_hardware()
    profile = select_profile(hw)

    table = Table(title="Hardware Profile", show_header=False)
    igpu_tag = " iGPU" if hw.gpu.is_igpu else ""
    table.add_row("GPU", f"{hw.gpu.vendor} {hw.gpu.name or '—'}{igpu_tag} ({hw.gpu.vram_gb:.1f} GB)")
    table.add_row("CPU", f"{hw.cpu.model} ({hw.cpu.cores} cores / {hw.cpu.threads} threads)")
    table.add_row("RAM", f"{hw.ram_gb:.1f} GB")
    table.add_row("Driver", hw.gpu.driver or "—")
    table.add_row("Profile", f"[bold green]{profile.name}[/bold green]")
    table.add_row("Description", profile.description)
    table.add_row("Recommended", profile.recommended_model)
    if profile.larger_model:
        table.add_row("Larger option", profile.larger_model)
    console.print(table)


@main.command()
def doctor():
    """Diagnose setup issues (Ollama, hardware, imports)."""
    ok = True

    def check(name: str, passed: bool, detail: str = "", fix: str = ""):
        nonlocal ok
        mark = "[green]OK[/green]" if passed else "[red]FAIL[/red]"
        console.print(f"  {mark}  {name}" + (f"  [dim]{detail}[/dim]" if detail else ""))
        if not passed:
            ok = False
            if fix:
                console.print(f"       [yellow]fix:[/yellow] {fix}")

    console.print("[bold]Python[/bold]")
    check("Python 3.10+", sys.version_info >= (3, 10), f"{sys.version.split()[0]}")

    console.print("\n[bold]Ollama[/bold]")
    try:
        client = _get_client()
        client.list_models()
        check("Ollama server reachable", True, os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    except Exception as e:
        check("Ollama server reachable", False, str(e)[:80],
              fix="Start Ollama: 'ollama serve' or install: curl -fsSL https://ollama.com/install.sh | sh")

    console.print("\n[bold]Hardware[/bold]")
    try:
        from anvil.hardware import detect_hardware

        hw = detect_hardware()
        check("Hardware detected", True, f"{hw.gpu.vendor} / {hw.gpu.vram_gb:.1f} GB VRAM")
    except Exception as e:
        check("Hardware detection", False, str(e)[:80])

    console.print("\n[bold]Imports[/bold]")
    for mod in ("anvil.llm.context", "anvil.mcp.manager", "anvil.agents.base"):
        try:
            __import__(mod)
            check(mod, True)
        except Exception as e:
            check(mod, False, str(e)[:80])

    console.print()
    if ok:
        console.print("[bold green]All checks passed.[/bold green]")
    else:
        console.print("[bold red]Some checks failed — see fixes above.[/bold red]")
        sys.exit(1)


@main.command()
def version():
    """Show anvil + Ollama versions."""
    try:
        from importlib.metadata import version as _v

        anvil_v = _v("ollama-anvil")
    except Exception:
        anvil_v = "dev"
    console.print(f"ollama-anvil [cyan]{anvil_v}[/cyan]")
    try:
        import subprocess

        out = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=3)
        console.print(f"ollama         [cyan]{out.stdout.strip() or out.stderr.strip()}[/cyan]")
    except Exception:
        console.print("ollama         [dim]not found[/dim]")


@main.command()
def env():
    """Show ANVIL_*, OLLAMA_*, and relevant GPU env vars."""
    keys = sorted(
        k for k in os.environ
        if k.startswith(("ANVIL_", "OLLAMA_", "HSA_", "ROCM_", "CUDA_", "HIP_"))
    )
    if not keys:
        console.print("[dim]No anvil/ollama env vars set.[/dim]")
        return
    for k in keys:
        console.print(f"  [cyan]{k}[/cyan] = {os.environ[k]}")


# ─── Models ──────────────────────────────────────────────────────────────────


@main.group()
def models():
    """Manage Ollama models (list, pull, info, HF import, recommend)."""
    pass


@models.command("list")
def models_list():
    """List locally installed models."""
    client = _get_client()
    ms = client.list_models()
    if not ms:
        console.print("[dim]No models installed.[/dim] Run: [cyan]anvil models recommend[/cyan]")
        return
    table = Table(title=f"Installed models ({len(ms)})")
    table.add_column("Name", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Modified")
    for m in ms:
        size_gb = m.get("size", 0) / 1e9
        table.add_row(m["name"], f"{size_gb:.1f} GB", m.get("modified_at", "")[:10])
    console.print(table)


@models.command("pull")
@click.argument("model_name")
def models_pull(model_name: str):
    """Pull a model from the Ollama registry."""
    client = _get_client()
    console.print(f"Pulling [cyan]{model_name}[/cyan]...")
    for status in client.pull(model_name):
        console.print(f"  {status}", end="\r")
    console.print("\n[green]Done.[/green]")


@models.command("remove")
@click.argument("model_name")
def models_remove(model_name: str):
    """Remove a locally installed model."""
    client = _get_client()
    client.delete_model(model_name)
    console.print(f"[green]Removed {model_name}[/green]")


@models.command("info")
@click.argument("model_name")
def models_info(model_name: str):
    """Show details about a model (params, size, modelfile)."""
    client = _get_client()
    info = client.show_model(model_name)
    console.print(Panel(
        f"[bold]{model_name}[/bold]\n"
        f"Family:  {info.get('details', {}).get('family', '—')}\n"
        f"Params:  {info.get('details', {}).get('parameter_size', '—')}\n"
        f"Quant:   {info.get('details', {}).get('quantization_level', '—')}\n"
        f"Format:  {info.get('details', {}).get('format', '—')}",
        title="Model info",
    ))


@models.command("recommend")
def models_recommend():
    """Recommend models for your hardware."""
    from anvil.hardware import detect_hardware, select_profile

    hw = detect_hardware()
    profile = select_profile(hw)
    console.print(f"\n[bold]Profile:[/bold] {profile.name}  ([dim]{hw.gpu.vram_gb:.1f} GB VRAM, {hw.ram_gb:.1f} GB RAM[/dim])\n")
    recommendations = [profile.recommended_model]
    if profile.larger_model:
        recommendations.append(profile.larger_model)
    if profile.fallback_model and profile.fallback_model not in recommendations:
        recommendations.append(profile.fallback_model)
    for i, model in enumerate(recommendations, 1):
        console.print(f"  {i}. [cyan]{model}[/cyan]")
    console.print("\nPull one with: [cyan]anvil models pull <name>[/cyan]")


@models.command("search")
@click.argument("query")
def models_search(query: str):
    """Search the Ollama library for models (offline index)."""
    from anvil.llm.models import search_ollama_library

    results = search_ollama_library(query)
    if not results:
        console.print("[dim]No matches.[/dim]")
        return
    table = Table(title=f"Ollama library: '{query}'")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    for r in results[:20]:
        table.add_row(r["name"], r.get("description", "")[:80])
    console.print(table)


@models.command("disk")
def models_disk():
    """Show disk usage of installed models."""
    client = _get_client()
    ms = client.list_models()
    total = sum(m.get("size", 0) for m in ms)
    console.print(f"[bold]{len(ms)} models[/bold]  —  total: [cyan]{total / 1e9:.1f} GB[/cyan]")
    for m in sorted(ms, key=lambda x: -x.get("size", 0))[:10]:
        console.print(f"  {m.get('size', 0) / 1e9:6.1f} GB  [cyan]{m['name']}[/cyan]")


@models.command("search-hf")
@click.argument("query")
@click.option("--limit", default=20, show_default=True)
@click.option("--show-quants", is_flag=True, help="List available GGUF quantizations")
def models_search_hf(query: str, limit: int, show_quants: bool):
    """Search Hugging Face for GGUF models."""
    from anvil.llm.huggingface import search_models

    results = search_models(query, limit=limit)
    if not results:
        console.print("[dim]No results.[/dim]")
        return
    for r in results:
        console.print(f"  [cyan]{r.repo_id}[/cyan]  [dim]({r.downloads:,} dl, {r.likes:,} likes)[/dim]")
        if show_quants and r.quant_files:
            for q in r.quant_files[:5]:
                console.print(f"      - {q}")


@models.command("pull-hf")
@click.argument("repo_id")
@click.option("--quant", default=None, help="Quantization tag (e.g. Q4_K_M)")
@click.option("--list-only", is_flag=True, help="Just list files, don't download")
def models_pull_hf(repo_id: str, quant: str | None, list_only: bool):
    """Download a GGUF file from a Hugging Face repo and import into Ollama."""
    from anvil.llm.huggingface import pull_gguf

    result = pull_gguf(repo_id, quant=quant, list_only=list_only)
    if list_only:
        for f in result.available_files:
            console.print(f"  {f}")
        return
    console.print(f"[green]Pulled {result.downloaded_path}[/green]")
    console.print(f"Import with: [cyan]anvil models import {result.downloaded_path}[/cyan]")


@models.command("convert")
@click.argument("repo_id")
@click.option("--quant", default="Q4_K_M", show_default=True, help="Target quantization")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
@click.option("--name", "ollama_name", default=None, help="Ollama model name")
def models_convert(repo_id: str, quant: str, output_dir: Path | None, ollama_name: str | None):
    """Convert a Hugging Face safetensors repo to GGUF and import to Ollama."""
    from anvil.llm.convert import convert_hf_repo

    def progress(status: str) -> None:
        console.print(f"  {status}")

    result = convert_hf_repo(repo_id, quant=quant, output_dir=output_dir,
                             ollama_name=ollama_name, progress_cb=progress)
    console.print(f"\n[green]{result.message}[/green]")


@models.command("import")
@click.argument("gguf_path", type=click.Path(exists=True, path_type=Path))
@click.option("--name", default=None, help="Ollama model name (defaults to filename)")
def models_import(gguf_path: Path, name: str | None):
    """Import a local GGUF file into Ollama."""
    from anvil.llm.convert import import_gguf

    def progress(status: str) -> None:
        console.print(f"  {status}")

    result = import_gguf(gguf_path, target_name=name, progress_cb=progress)
    console.print(f"\n[green]{result.message}[/green]")


@models.command("convert-setup")
@click.option("--no-build", is_flag=True, help="Clone only, don't build")
def models_convert_setup(no_build: bool):
    """One-time setup: clone + build llama.cpp for GGUF conversion."""
    from anvil.llm.convert import setup_llama_cpp

    def progress(status: str) -> None:
        console.print(f"  {status}")

    setup_llama_cpp(build=not no_build, progress_cb=progress)


# ─── MCP ─────────────────────────────────────────────────────────────────────


@main.group()
def mcp():
    """Manage MCP servers (web search, filesystem, etc.)."""
    pass


@mcp.command("list")
def mcp_list():
    """List active MCP servers."""
    from anvil.mcp.manager import MCPManager

    mgr = MCPManager()
    active = mgr.list_active()
    if not active:
        console.print("[dim]No MCP servers active.[/dim]")
        return
    table = Table(title=f"Active MCPs ({len(active)})")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    for m in active:
        table.add_row(m.name, m.description)
    console.print(table)


@mcp.command("add")
@click.argument("name")
def mcp_add(name: str):
    """Enable an MCP server by name."""
    from anvil.mcp.manager import MCPManager

    mgr = MCPManager()
    mgr.enable(name)
    console.print(f"[green]Enabled MCP: {name}[/green]")


@mcp.command("remove")
@click.argument("name")
def mcp_remove(name: str):
    """Disable an MCP server."""
    from anvil.mcp.manager import MCPManager

    mgr = MCPManager()
    mgr.disable(name)
    console.print(f"[green]Disabled MCP: {name}[/green]")


@mcp.command("search")
@click.argument("query")
def mcp_search(query: str):
    """Search the MCP registry."""
    from anvil.mcp.registry import MCP_REGISTRY

    q = query.lower()
    matches = [m for m in MCP_REGISTRY if q in m.name.lower() or q in m.description.lower()]
    if not matches:
        console.print("[dim]No matches.[/dim]")
        return
    for m in matches[:20]:
        console.print(f"  [cyan]{m.name}[/cyan]  {m.description}")


# ─── Agents ──────────────────────────────────────────────────────────────────


@main.group()
def agent():
    """Manage agents (list, create, install templates)."""
    pass


@agent.command("list")
def agent_list():
    """List built-in and user-defined agents."""
    from anvil.agents.orchestrator import AgentOrchestrator

    client = _get_client()
    orch = AgentOrchestrator(client=client, working_dir=".")
    table = Table(title=f"Agents ({len(orch.agents)})")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Tools")
    for name, a in orch.agents.items():
        table.add_row(name, a.config.description or "—", ", ".join(a.config.tools))
    console.print(table)


@agent.command("templates")
def agent_templates():
    """List bundled agent templates."""
    tmpl_dir = Path(__file__).parent / "agents" / "templates"
    table = Table(title="Bundled templates")
    table.add_column("Template", style="cyan")
    table.add_column("Path")
    for t in sorted(tmpl_dir.glob("*.yaml")):
        table.add_row(t.stem, str(t))
    console.print(table)
    console.print("\nInstall with: [cyan]anvil agent install <name>[/cyan]")


@agent.command("install")
@click.argument("template_name")
def agent_install(template_name: str):
    """Copy a template into ./agents/ for customization."""
    src = Path(__file__).parent / "agents" / "templates" / f"{template_name}.yaml"
    if not src.exists():
        console.print(f"[red]Unknown template: {template_name}[/red]")
        ctx = click.get_current_context()
        ctx.invoke(agent_templates)
        sys.exit(1)
    dst_dir = Path("agents")
    dst_dir.mkdir(exist_ok=True)
    dst = dst_dir / f"{template_name}.yaml"
    if dst.exists():
        console.print(f"[yellow]{dst} already exists.[/yellow]")
        return
    dst.write_text(src.read_text())
    console.print(f"[green]Installed {dst}[/green]. Edit it, then: [cyan]anvil chat --agent {template_name}[/cyan]")


@agent.command("create")
@click.argument("name")
@click.option("--description", default="", help="Short description")
@click.option("--model", default="", help="Model to use")
def agent_create(name: str, description: str, model: str):
    """Create a new custom agent YAML in ./agents/."""
    dst_dir = Path("agents")
    dst_dir.mkdir(exist_ok=True)
    dst = dst_dir / f"{name}.yaml"
    if dst.exists():
        console.print(f"[red]{dst} already exists.[/red]")
        sys.exit(1)
    dst.write_text(
        f"name: {name}\n"
        f"description: {description or name}\n"
        f"model: {model or _default_model()}\n"
        "system_prompt: |\n"
        f"  You are {name}. Describe your role and approach here.\n"
        "tools:\n"
        "  - filesystem\n"
        "  - shell\n"
        "  - web\n"
    )
    console.print(f"[green]Created {dst}[/green]")


@agent.command("run")
@click.argument("name")
def agent_run(name: str):
    """Run a named agent in chat mode."""
    ctx = click.get_current_context()
    ctx.invoke(chat, agent=name)


# ─── RAG ─────────────────────────────────────────────────────────────────────


@main.group()
def rag():
    """Retrieval-augmented generation over local docs."""
    pass


@rag.command("ingest")
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--store", default="default", show_default=True)
def rag_ingest(paths: tuple, store: str):
    """Ingest files/directories into a RAG store."""
    from anvil.llm.rag import RAGPipeline

    pipe = RAGPipeline(store_name=store)
    for p in paths:
        console.print(f"Ingesting [cyan]{p}[/cyan]...")
        pipe.ingest_path(Path(p))
    console.print(f"[green]Done. Store: {store}[/green]")


@rag.command("query")
@click.argument("question", nargs=-1, required=True)
@click.option("--store", default="default")
@click.option("--top-k", default=5, show_default=True)
def rag_query(question: tuple, store: str, top_k: int):
    """Query a RAG store."""
    from anvil.llm.rag import RAGPipeline

    pipe = RAGPipeline(store_name=store)
    result = pipe.query(" ".join(question), top_k=top_k)
    console.print(Panel(result.answer, title="Answer"))
    console.print("\n[bold]Sources:[/bold]")
    for s in result.sources:
        console.print(f"  • {s}")


@rag.command("status")
@click.option("--store", default="default")
def rag_status(store: str):
    """Show RAG store stats."""
    from anvil.llm.rag import RAGPipeline

    pipe = RAGPipeline(store_name=store)
    s = pipe.status()
    console.print(f"Store: [cyan]{store}[/cyan]  docs: {s.get('docs', 0)}  chunks: {s.get('chunks', 0)}")


@rag.command("clear")
@click.option("--store", default="default")
def rag_clear(store: str):
    """Delete all content from a RAG store."""
    from anvil.llm.rag import RAGPipeline

    if not click.confirm(f"Delete all content in store '{store}'?"):
        return
    RAGPipeline(store_name=store).clear()
    console.print(f"[green]Cleared {store}[/green]")


# ─── Community: ideas + self-improvement ─────────────────────────────────────


@main.group()
def idea():
    """Submit and browse community ideas (opt-out via ANVIL_COMMUNITY_IDEAS=0)."""
    pass


@idea.command("list")
def idea_list():
    """List recent community ideas."""
    from anvil.community.ideas import IdeaCollector

    for i in IdeaCollector().list_ideas()[:30]:
        console.print(f"  • {i}")


@idea.command("submit")
@click.argument("description", nargs=-1, required=True)
def idea_submit(description: tuple):
    """Submit an idea to improve ollama-anvil."""
    from anvil.community.ideas import IdeaCollector

    IdeaCollector().submit(" ".join(description))
    console.print("[green]Thanks![/green] Your idea was recorded.")


@main.command("self-improve")
@click.option("--iterations", default=1, show_default=True)
@click.option("--enable", is_flag=True, help="Enable the self-improvement agent")
@click.option("--maintainer", is_flag=True, help="Maintainer mode: direct push (repo owner only)")
def self_improve(iterations: int, enable: bool, maintainer: bool):
    """Run the self-improvement agent to propose PRs against anvil itself."""
    from anvil.community.self_improve import SelfImproveAgent

    if enable:
        env_file = Path(".env")
        line = "ANVIL_SELF_IMPROVE=1\n"
        if env_file.exists():
            content = env_file.read_text()
            if "ANVIL_SELF_IMPROVE=" not in content:
                env_file.write_text(content.rstrip() + "\n" + line)
        else:
            env_file.write_text(line)
        console.print("[green]Enabled.[/green] Set ANVIL_SELF_IMPROVE=1 in .env")
        return

    if os.environ.get("ANVIL_SELF_IMPROVE") != "1":
        console.print("[yellow]Self-improvement is opt-in.[/yellow] Run: [cyan]anvil self-improve --enable[/cyan]")
        return

    agent = SelfImproveAgent(client=_get_client(), repo_dir=Path("."), maintainer_mode=maintainer)
    for i in range(iterations):
        console.print(f"\n[bold]Iteration {i + 1}/{iterations}[/bold]")
        agent.run_iteration()


# ─── Tools ───────────────────────────────────────────────────────────────────


@main.group()
def tools():
    """Inspect built-in tools."""
    pass


@tools.command("list")
def tools_list():
    """List available built-in tools."""
    from anvil.tools import BUILTIN_TOOLS

    table = Table(title=f"Built-in tools ({len(BUILTIN_TOOLS)})")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    for name, cls in BUILTIN_TOOLS.items():
        table.add_row(name, (cls.__doc__ or "").strip().split("\n")[0][:80])
    console.print(table)


# ─── AI Dev tools (concise subset of forge's ~100 dev commands) ──────────────


def _read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        console.print(f"[red]File not found: {path}[/red]")
        sys.exit(1)
    return p.read_text()


@main.command("diff")
@click.option("--model", default="", help="Review model")
@click.option("--staged", is_flag=True, help="Review staged changes")
def diff_cmd(model: str, staged: bool):
    """AI code review of git diff."""
    import subprocess

    cmd = ["git", "diff"] + (["--cached"] if staged else [])
    diff = subprocess.check_output(cmd, text=True)
    if not diff.strip():
        console.print("[dim]No changes to review.[/dim]")
        return
    _run_prompt(model, f"Review this diff and flag bugs, regressions, or style issues. Be concise.\n\n{diff}",
                system="You are a senior code reviewer.")


@main.command("explain")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--model", default="")
def explain_cmd(file_path: str, model: str):
    """Explain what a file does."""
    code = _read_file(file_path)
    _run_prompt(model, f"Explain what this code does, at a high level.\n\n```\n{code}\n```",
                system="You are a concise technical writer.")


@main.command("commit")
@click.option("--model", default="")
@click.option("--stage-all", is_flag=True, help="git add -A first")
def commit_cmd(model: str, stage_all: bool):
    """Generate a commit message from staged changes."""
    import subprocess

    if stage_all:
        subprocess.run(["git", "add", "-A"], check=True)
    diff = subprocess.check_output(["git", "diff", "--cached"], text=True)
    if not diff.strip():
        console.print("[yellow]No staged changes.[/yellow] Run [cyan]git add <files>[/cyan] or use [cyan]--stage-all[/cyan]")
        return
    msg = _run_prompt(model, f"Write a concise git commit message (conventional commits style) for this diff. "
                             f"Subject line under 70 chars, then blank line, then bullet points.\n\n{diff}",
                      system="You write clear commit messages.", stream=False)
    console.print("\n[bold]Generated commit message:[/bold]\n")
    console.print(msg)
    if click.confirm("\nUse this message?"):
        subprocess.run(["git", "commit", "-m", msg.strip()], check=True)


@main.command("review")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--focus", default="", help="Focus area: security, perf, style, etc.")
@click.option("--model", default="")
def review_cmd(file_path: str, focus: str, model: str):
    """Full code review of a file."""
    code = _read_file(file_path)
    focus_str = f" Focus on {focus}." if focus else ""
    _run_prompt(model, f"Review this file.{focus_str}\n\n```\n{code}\n```",
                system="You are a senior code reviewer. Be specific and cite line numbers.")


@main.command("refactor")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--pattern", default="", help="Refactor pattern to apply")
@click.option("--output", "-o", default="", help="Write to file")
@click.option("--model", default="")
def refactor_cmd(file_path: str, pattern: str, output: str, model: str):
    """Suggest refactors for a file."""
    code = _read_file(file_path)
    prompt = "Refactor this code"
    if pattern:
        prompt += f" using: {pattern}"
    prompt += f". Output only the refactored code.\n\n```\n{code}\n```"
    result = _run_prompt(model, prompt, system="You produce clean, idiomatic refactors.",
                         stream=not output)
    if output:
        Path(output).write_text(result)
        console.print(f"[green]Written to {output}[/green]")


@main.command("doc")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--model", default="")
def doc_cmd(file_path: str, model: str):
    """Generate/improve docstrings for a file."""
    code = _read_file(file_path)
    _run_prompt(model, f"Add or improve docstrings in this file. Output the full updated file.\n\n```\n{code}\n```",
                system="You write clear, concise docstrings.")


@main.command("test-gen")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--framework", default="pytest", show_default=True)
@click.option("--output", "-o", default="")
@click.option("--model", default="")
def test_gen_cmd(file_path: str, framework: str, output: str, model: str):
    """Generate unit tests for a file."""
    code = _read_file(file_path)
    result = _run_prompt(model, f"Generate {framework} tests for this code.\n\n```\n{code}\n```",
                         system="You write thorough, isolated tests.", stream=not output)
    if output:
        Path(output).write_text(result)
        console.print(f"[green]Written to {output}[/green]")


@main.command("security")
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--model", default="")
def security_cmd(paths: tuple, model: str):
    """Security audit over files/directories."""
    contents = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in list(p.rglob("*.py"))[:30]:
                contents.append(f"### {f}\n```\n{f.read_text()[:5000]}\n```")
        else:
            contents.append(f"### {p}\n```\n{p.read_text()[:5000]}\n```")
    _run_prompt(model, "Find security issues (OWASP top 10, injection, secrets, unsafe deserialization, "
                       "command injection). Cite file:line. Be concrete.\n\n" + "\n\n".join(contents),
                system="You are a security auditor.")


@main.command("summarize")
@click.argument("source")
@click.option("--length", type=click.Choice(["short", "medium", "long"]), default="short", show_default=True)
@click.option("--model", default="")
def summarize_cmd(source: str, length: str, model: str):
    """Summarize a file or URL."""
    if source.startswith(("http://", "https://")):
        import requests

        text = requests.get(source, timeout=15).text[:40000]
    else:
        text = _read_file(source)
    lengths = {"short": "1-2 sentences", "medium": "1 paragraph", "long": "3-5 paragraphs"}
    _run_prompt(model, f"Summarize this in {lengths[length]}.\n\n{text}",
                system="You are a concise summarizer.")


@main.command("translate")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--to", "target_lang", required=True, help="Target language (rust, go, ts, etc.)")
@click.option("--output", "-o", default="")
@click.option("--model", default="")
def translate_cmd(file_path: str, target_lang: str, output: str, model: str):
    """Translate a source file between programming languages."""
    code = _read_file(file_path)
    result = _run_prompt(model, f"Translate this code to {target_lang}. Output only the translated code.\n\n```\n{code}\n```",
                         system="You translate idiomatically between languages.", stream=not output)
    if output:
        Path(output).write_text(result)
        console.print(f"[green]Written to {output}[/green]")


@main.command("todos")
@click.option("--directory", "-d", default=".")
def todos_cmd(directory: str):
    """Find TODO/FIXME/HACK/XXX comments."""
    import re

    pattern = re.compile(r"\b(TODO|FIXME|HACK|XXX)\b[:\s]?(.*)", re.IGNORECASE)
    total = 0
    for f in Path(directory).rglob("*"):
        if not f.is_file() or f.suffix not in {".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".md"}:
            continue
        if any(part.startswith(".") or part in {"node_modules", "venv", ".venv", "dist", "build"} for part in f.parts):
            continue
        try:
            for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
                if m := pattern.search(line):
                    console.print(f"  [cyan]{f}:{i}[/cyan]  [yellow]{m.group(1)}[/yellow] {m.group(2).strip()}")
                    total += 1
        except Exception:
            continue
    console.print(f"\n[bold]{total} items.[/bold]")


if __name__ == "__main__":
    main()
