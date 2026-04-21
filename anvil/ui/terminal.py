"""Terminal UI using Textual — rich chat interface for ollama-anvil.

Provides a full TUI with:
- Chat panel with markdown rendering
- Model picker sidebar
- Status bar (hardware, model, agent, MCP)
- Agent switcher
- Slash commands (/model, /agent, /reset, /quit)

Install with: pip install ollama-anvil[tui]

Usage:
    anvil tui
    anvil tui --model qwen2.5-coder:7b
    anvil tui --agent coder
"""

from __future__ import annotations

from anvil.utils.logging import get_logger

log = get_logger("ui.terminal")

__all__ = [
    "launch_tui",
    "MAX_MODELS_DISPLAY",
]

# Display limits
MAX_MODELS_DISPLAY = 10  # max models to show in list command


def launch_tui(
    model: str = "",
    agent: str = "assistant",
    working_dir: str = ".",
    cascade: bool = False,
    auto_approve: bool = False,
):
    """Launch the Terminal UI."""
    try:
        from textual.app import App  # noqa: F401
    except ImportError:
        print("Terminal UI requires textual. Install with:")
        print("  pip install ollama-anvil[tui]")
        print("  # or: pip install textual")
        return

    app = ForgeApp(
        model=model,
        agent_name=agent,
        working_dir=working_dir,
        cascade=cascade,
        auto_approve=auto_approve,
    )
    app.run()


# ─── Textual App ─────────────────────────────────────────────────────────────

try:
    from textual import on, work
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.css.query import NoMatches
    from textual.reactive import reactive
    from textual.widgets import (
        Footer,
        Header,
        Input,
        Label,
        ListItem,
        ListView,
        Markdown,
        Static,
    )

    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False

if TEXTUAL_AVAILABLE:

    class ChatMessage(Static):
        """A single chat message bubble."""

        def __init__(self, role: str, content: str, **kwargs):
            super().__init__(**kwargs)
            self.role = role
            self.content = content

        def compose(self) -> ComposeResult:
            if self.role == "user":
                yield Static("[bold cyan]You[/bold cyan]")
                yield Static(self.content)
            elif self.role == "assistant":
                yield Static("[bold green]Agent[/bold green]")
                yield Markdown(self.content)
            else:
                yield Static(f"[dim]{self.role}[/dim]: {self.content}")

    class StatusPanel(Static):
        """Status bar showing hardware, model, agent info."""

        model_name: reactive[str] = reactive("")
        agent_name: reactive[str] = reactive("assistant")
        hardware_profile: reactive[str] = reactive("")
        gpu_name: reactive[str] = reactive("")
        message_count: reactive[int] = reactive(0)
        is_thinking: reactive[bool] = reactive(False)

        def render(self) -> str:
            thinking = " [bold yellow]thinking...[/bold yellow]" if self.is_thinking else ""
            return (
                f"[bold]Model:[/bold] [cyan]{self.model_name}[/cyan] | "
                f"[bold]Agent:[/bold] [green]{self.agent_name}[/green] | "
                f"[bold]Profile:[/bold] {self.hardware_profile} | "
                f"[bold]GPU:[/bold] {self.gpu_name} | "
                f"Messages: {self.message_count}"
                f"{thinking}"
            )

    class ModelList(ListView):
        """Sidebar showing available models."""

        def __init__(self, models: list[str] | None = None, **kwargs):
            super().__init__(**kwargs)
            self._model_names = models or []

        def compose(self) -> ComposeResult:
            for name in self._model_names:
                yield ListItem(Label(name), name=name)

    class ForgeApp(App):
        """ollama-anvil Terminal UI."""

        CSS = """
        Screen {
            layout: grid;
            grid-size: 4 3;
            grid-columns: 1fr 1fr 1fr 4fr;
            grid-rows: auto 1fr auto;
        }

        #header-bar {
            column-span: 4;
            dock: top;
            height: 1;
            background: $primary-background;
            color: $text;
            text-align: center;
        }

        #sidebar {
            column-span: 1;
            row-span: 1;
            border-right: solid $primary;
            max-width: 28;
            min-width: 20;
        }

        #sidebar-title {
            text-align: center;
            text-style: bold;
            padding: 0 1;
            background: $primary-background;
        }

        #chat-area {
            column-span: 3;
            row-span: 1;
        }

        #chat-scroll {
            height: 1fr;
            padding: 0 1;
        }

        ChatMessage {
            margin: 1 0;
            padding: 0 1;
        }

        #input-area {
            column-span: 4;
            dock: bottom;
            height: auto;
            padding: 0 1;
        }

        #chat-input {
            margin: 0 0;
        }

        #status-panel {
            column-span: 4;
            dock: bottom;
            height: 1;
            background: $primary-background;
            padding: 0 1;
        }

        #welcome-msg {
            padding: 1 2;
            margin: 1 0;
        }

        .model-item {
            padding: 0 1;
        }

        ListView > ListItem.--highlight {
            background: $primary 30%;
        }
        """

        BINDINGS = [
            Binding("ctrl+q", "quit", "Quit"),
            Binding("ctrl+r", "reset", "Reset chat"),
            Binding("ctrl+s", "toggle_sidebar", "Sidebar"),
            Binding("escape", "focus_input", "Focus input"),
        ]

        TITLE = "ollama-anvil"

        def __init__(
            self,
            model: str = "",
            agent_name: str = "assistant",
            working_dir: str = ".",
            cascade: bool = False,
            auto_approve: bool = False,
        ):
            super().__init__()
            self._model_arg = model
            self._agent_name = agent_name
            self._working_dir = working_dir
            self._cascade = cascade
            self._auto_approve = auto_approve
            self._client = None
            self._orchestrator = None
            self._hw = None
            self._profile = None
            self._available_models: list[str] = []
            self._sidebar_visible = True

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal():
                with Vertical(id="sidebar"):
                    yield Static("Models", id="sidebar-title")
                    yield ModelList(id="model-list")
                with Vertical(id="chat-area"), VerticalScroll(id="chat-scroll"):
                    yield Static(
                        "[bold]Welcome to ollama-anvil TUI[/bold]\n\n"
                        "Commands: /model <name>, /agent <name>, /reset, /quit\n"
                        "Keys: Ctrl+Q quit, Ctrl+R reset, Ctrl+S sidebar",
                        id="welcome-msg",
                    )
            yield Input(placeholder="Type a message or /command...", id="chat-input")
            yield StatusPanel(id="status-panel")
            yield Footer()

        def on_mount(self) -> None:
            """Initialize when the app mounts."""
            self._init_backend()

        @work(thread=True)
        def _init_backend(self) -> None:
            """Initialize hardware detection, client, and orchestrator in background."""
            from anvil.agents.orchestrator import AgentOrchestrator
            from anvil.agents.permissions import AutoApproveManager, PermissionManager
            from anvil.config import load_config
            from anvil.hardware import detect_hardware, select_profile
            from anvil.hardware.rocm import configure_rocm_env
            from anvil.llm.client import OllamaClient

            config = load_config()
            self._hw = detect_hardware()
            self._profile = select_profile(self._hw)
            configure_rocm_env(self._hw.gpu)

            model_name = self._model_arg or config.default_model or self._profile.recommended_model

            self._client = OllamaClient(
                model=model_name,
                base_url=config.ollama_base_url,
                num_ctx=self._profile.num_ctx,
                num_thread=self._profile.max_threads,
                num_batch=self._profile.num_batch,
            )

            _permissions = AutoApproveManager() if self._auto_approve else PermissionManager()

            self._orchestrator = AgentOrchestrator(
                client=self._client,
                working_dir=self._working_dir,
            )
            if self._agent_name != "assistant":
                self._orchestrator.switch_agent(self._agent_name)

            # Load available models
            if self._client.is_available():
                models = self._client.list_models()
                self._available_models = [m.get("name", "") for m in models]

            # Update UI from the worker thread
            self.call_from_thread(self._update_status_after_init, model_name)

        def _update_status_after_init(self, model_name: str) -> None:
            """Update the status panel and model list after init."""
            status = self.query_one("#status-panel", StatusPanel)
            status.model_name = model_name
            status.agent_name = self._agent_name
            status.hardware_profile = self._profile.name if self._profile else "?"
            status.gpu_name = self._hw.gpu.name if self._hw else "?"

            # Populate model list
            try:
                model_list = self.query_one("#model-list", ModelList)
                for name in self._available_models:
                    model_list.append(ListItem(Label(name), name=name))
            except NoMatches:
                pass  # Sidebar not mounted yet; model list populated later

            # Focus input
            self.query_one("#chat-input", Input).focus()

        @on(Input.Submitted, "#chat-input")
        def handle_input(self, event: Input.Submitted) -> None:
            """Handle user input submission."""
            text = event.value.strip()
            if not text:
                return

            event.input.clear()

            # Handle slash commands
            if text.startswith("/"):
                self._handle_command(text)
                return

            # Add user message to chat
            self._add_message("user", text)

            # Send to agent in background
            self._send_message(text)

        def _handle_command(self, command: str) -> None:
            """Handle slash commands."""
            parts = command.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit"):
                self.exit()
            elif cmd == "/reset":
                self._reset_chat()
                self._add_system_message("Chat reset.")
            elif cmd == "/model":
                if arg:
                    self._switch_model(arg)
                else:
                    models_str = ", ".join(self._available_models[:MAX_MODELS_DISPLAY])
                    self._add_system_message(f"Available models: {models_str}")
            elif cmd == "/agent":
                if arg:
                    self._switch_agent(arg)
                else:
                    self._add_system_message("Agents: assistant, coder, researcher")
            elif cmd == "/agents":
                self._add_system_message("Agents: assistant, coder, researcher")
            elif cmd == "/stats":
                if self._orchestrator:
                    stats = self._orchestrator.get_all_stats()
                    import json
                    self._add_system_message(f"```json\n{json.dumps(stats, indent=2)}\n```")
                else:
                    self._add_system_message("Not initialized yet.")
            elif cmd == "/help":
                self._add_system_message(
                    "**Commands:**\n"
                    "- `/model <name>` — Switch model\n"
                    "- `/agent <name>` — Switch agent\n"
                    "- `/reset` — Clear chat history\n"
                    "- `/stats` — Show usage statistics\n"
                    "- `/quit` — Exit\n\n"
                    "**Keys:** Ctrl+Q quit, Ctrl+R reset, Ctrl+S sidebar"
                )
            else:
                # Pass unknown commands to orchestrator
                if self._orchestrator:
                    response = self._orchestrator.chat(command)
                    self._add_message("assistant", response)
                else:
                    self._add_system_message(f"Unknown command: {cmd}")

        def _switch_model(self, model_name: str) -> None:
            """Switch to a different model."""
            if self._client and self._client.switch_model(model_name):
                status = self.query_one("#status-panel", StatusPanel)
                status.model_name = model_name
                self._add_system_message(f"Switched to model: {model_name}")
            else:
                self._add_system_message(f"Failed to switch to {model_name}")

        def _switch_agent(self, agent_name: str) -> None:
            """Switch to a different agent."""
            if self._orchestrator:
                result = self._orchestrator.switch_agent(agent_name)
                self._agent_name = agent_name
                status = self.query_one("#status-panel", StatusPanel)
                status.agent_name = agent_name
                self._add_system_message(result)

        @work(thread=True)
        def _send_message(self, text: str) -> None:
            """Send a message to the agent in a background thread."""
            if not self._orchestrator:
                self.call_from_thread(
                    self._add_system_message, "Still initializing... please wait."
                )
                return

            # Show thinking indicator
            self.call_from_thread(self._set_thinking, True)

            try:
                response = self._orchestrator.chat(text)
                self.call_from_thread(self._add_message, "assistant", response)
            except (OSError, TimeoutError, ValueError) as e:
                log.error("Error in chat: %s", e, exc_info=True)
                self.call_from_thread(
                    self._add_system_message, f"Error: {e}"
                )
            finally:
                self.call_from_thread(self._set_thinking, False)

        def _set_thinking(self, thinking: bool) -> None:
            """Toggle the thinking indicator."""
            status = self.query_one("#status-panel", StatusPanel)
            status.is_thinking = thinking

        def _add_message(self, role: str, content: str) -> None:
            """Add a chat message to the scroll area."""
            scroll = self.query_one("#chat-scroll", VerticalScroll)
            msg = ChatMessage(role, content)
            scroll.mount(msg)
            scroll.scroll_end(animate=False)

            # Update message count
            status = self.query_one("#status-panel", StatusPanel)
            status.message_count += 1

        def _add_system_message(self, content: str) -> None:
            """Add a system/info message."""
            scroll = self.query_one("#chat-scroll", VerticalScroll)
            scroll.mount(Markdown(content))
            scroll.scroll_end(animate=False)

        def _reset_chat(self) -> None:
            """Clear the chat history."""
            if self._orchestrator:
                current = self._orchestrator.agents.get(self._orchestrator.active_agent)
                if current:
                    current.reset()

            # Clear chat scroll
            scroll = self.query_one("#chat-scroll", VerticalScroll)
            for child in list(scroll.children):
                child.remove()

            # Reset count
            status = self.query_one("#status-panel", StatusPanel)
            status.message_count = 0

        @on(ListView.Selected, "#model-list")
        def model_selected(self, event: ListView.Selected) -> None:
            """Handle model selection from sidebar."""
            if event.item.name:
                self._switch_model(event.item.name)

        def action_quit(self) -> None:
            """Quit the app."""
            self.exit()

        def action_reset(self) -> None:
            """Reset chat via keybinding."""
            self._reset_chat()
            self._add_system_message("Chat reset.")

        def action_toggle_sidebar(self) -> None:
            """Toggle the sidebar visibility."""
            try:
                sidebar = self.query_one("#sidebar", Vertical)
                self._sidebar_visible = not self._sidebar_visible
                sidebar.display = self._sidebar_visible
            except NoMatches:
                pass  # Sidebar not present in current layout; ignore toggle

        def action_focus_input(self) -> None:
            """Focus the chat input."""
            self.query_one("#chat-input", Input).focus()
