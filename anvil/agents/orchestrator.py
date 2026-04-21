"""Agent orchestrator: multi-agent coordination and message routing."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

from anvil.agents.base import AgentConfig, BaseAgent, load_agent_from_yaml
from anvil.agents.checkpoints import CheckpointStore
from anvil.agents.coder import create_coder_agent
from anvil.agents.researcher import create_researcher_agent
from anvil.llm.client import OllamaClient
from anvil.tools import BUILTIN_TOOLS
from anvil.utils.fileio import atomic_write
from anvil.utils.logging import get_logger

log = get_logger("agents.orchestrator")

__all__ = [
    "AgentOrchestrator",
    "BUILTIN_AGENT_NAMES",
    "RESERVED_NAMES",
    "MAX_AGENT_NAME_LENGTH",
]

# Built-in agent names that cannot be overwritten by user agents
BUILTIN_AGENT_NAMES = {"assistant", "coder", "researcher"}

# Reserved names that cannot be used for agents (conflict with commands/system)
RESERVED_NAMES = {"system", "help", "quit", "exit", "agent", "agents", "model", "models",
                  "mcp", "tools", "config", "status", "stats", "clear", "reset"}

# Maximum agent name length
MAX_AGENT_NAME_LENGTH = 50


class AgentOrchestrator:
    """Coordinates multiple agents, routes messages, manages lifecycle.

    Supports both single-agent and multi-agent workflows:
    - Single agent: user chats with one agent directly
    - Multi-agent: orchestrator routes tasks to specialized agents
    """

    def __init__(self, client: OllamaClient, working_dir: str = "."):
        self.client = client
        self.working_dir = working_dir
        self.agents: dict[str, BaseAgent] = {}
        self.active_agent: str = ""
        self.checkpoints = CheckpointStore(working_dir)

        # Load built-in agents
        self._register_builtin_agents()

        # Load user-defined agents from agents/ directory
        self._load_user_agents()

        # Wire handoff tools now that every agent is registered. Each
        # agent that loaded the ``handoff`` tool gets access to the full
        # registry (minus itself) and a callback that flips the active
        # agent.
        self._setup_handoff_tools()

    def _register_builtin_agents(self) -> None:
        """Register the built-in agents."""
        # Base assistant (default)
        self.register_agent(BaseAgent(
            client=self.client,
            config=AgentConfig(
                name="assistant",
                system_prompt=(
                    "You are a helpful AI assistant running locally via Ollama. "
                    "You can search the web, read/write files, and run commands. "
                    "Be concise and practical. When the user asks you to do something, "
                    "use your tools to actually do it — don't just explain how."
                ),
                tools=["filesystem", "shell", "git", "web", "codebase", "diff", "image"],
                description="General-purpose assistant with all tools",
            ),
            working_dir=self.working_dir,
        ))

        # Coder agent — uses dedicated factory with detailed system prompt
        self.register_agent(create_coder_agent(self.client, self.working_dir))

        # Researcher agent — uses dedicated factory with detailed system prompt
        self.register_agent(create_researcher_agent(self.client, self.working_dir))

        # Set default active agent
        self.active_agent = "assistant"

    def _load_user_agents(self) -> None:
        """Load user-defined agents from YAML files in agents/ directory."""
        agents_dir = Path(self.working_dir) / "agents"
        if not agents_dir.exists():
            return

        for yaml_file in sorted(agents_dir.glob("*.yaml")) + sorted(agents_dir.glob("*.yml")):
            try:
                agent = load_agent_from_yaml(str(yaml_file), self.client, self.working_dir)
                # Warn about unknown tools (agent still loads with available tools)
                unknown_tools = [t for t in agent.config.tools if t not in BUILTIN_TOOLS]
                if unknown_tools:
                    log.warning(
                        "Agent %s from %s references unknown tools: %s",
                        agent.config.name, yaml_file.name, unknown_tools,
                    )
                self.register_agent(agent)
                log.info("Loaded user agent: %s from %s", agent.config.name, yaml_file.name)
            except (OSError, ValueError, KeyError, AttributeError) as e:
                log.error("Failed to load agent from %s: %s", yaml_file, e)

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator."""
        self.agents[agent.config.name] = agent
        # Late-binding re-wire: if handoff tools are already set up, the
        # newcomer needs to appear in every other agent's registry.
        if self.agents and hasattr(self, "_handoff_ready") and self._handoff_ready:
            self._setup_handoff_tools()

    def _setup_handoff_tools(self) -> None:
        """Inject the agent registry + switch callback into every HandoffTool.

        Walks each agent, checks whether it loaded the ``handoff`` tool,
        and binds its registry/switch callback. Each agent's handoff
        target list excludes itself so the LLM never sees a no-op
        transfer option.
        """
        from anvil.tools.handoff import HandoffTool

        for agent_name, agent in self.agents.items():
            tool = agent._tools.get("handoff")  # type: ignore[attr-defined]
            if isinstance(tool, HandoffTool):
                tool.bind(
                    agent_registry=dict(self.agents),
                    switch_fn=self.switch_agent,
                    exclude=agent_name,
                )
                # Force tool-def cache rebuild so the LLM sees the
                # updated available-agents list.
                agent._cached_tool_defs = None  # type: ignore[attr-defined]
        self._handoff_ready = True

    def switch_agent(self, name: str) -> str:
        """Switch the active agent (case-insensitive)."""
        name = name.strip()
        # Case-insensitive lookup
        if name not in self.agents:
            lower_map = {k.lower(): k for k in self.agents}
            if name.lower() in lower_map:
                name = lower_map[name.lower()]
            else:
                available = ", ".join(self.agents.keys())
                return f"Unknown agent: {name}. Available: {available}"

        self.active_agent = name
        return f"Switched to agent: {name} — {self.agents[name].config.description}"

    def chat(self, message: str, images: list[str] | None = None, think: bool = False) -> str:
        """Route a message to the active agent.

        Args:
            message: User's input text.
            images: Optional image paths or base64 strings for vision models.
            think: If True, enable thinking/reasoning mode.
        """
        # Check for agent switching commands
        if message.startswith("/agent "):
            name = message[7:].strip()
            return self.switch_agent(name)

        if message == "/agents":
            return self._list_agents()

        # Session checkpoint commands. Dispatched here (rather than inside
        # BaseAgent) because saving/restoring spans agent identity too —
        # /checkpoint load can come back to a different active agent.
        if message.startswith("/checkpoint"):
            return self._handle_checkpoint(message)

        agent = self.agents.get(self.active_agent)
        if not agent:
            return "No active agent. Use /agent <name> to select one."

        return agent.chat(message, images=images, think=think)

    def stream_chat(self, message: str, images: list[str] | None = None, think: bool = False) -> Generator[dict[str, Any], None, None]:
        """Stream a response from the active agent, yielding typed events.

        Yields dicts with 'type' key: 'text', 'tool_call', 'tool_result',
        'done', 'error'. Falls back to non-streaming for slash commands.

        Args:
            message: User's input text.
            images: Optional image paths or base64 strings for vision models.
            think: If True, enable thinking mode — yields 'thinking' events.
        """
        # Slash commands are not streamable — return as single text event
        if message.startswith("/checkpoint"):
            yield {"type": "text", "content": self._handle_checkpoint(message)}
            yield {"type": "done"}
            return

        if message.startswith("/agent "):
            name = message[7:].strip()
            yield {"type": "text", "content": self.switch_agent(name)}
            yield {"type": "done"}
            return
        if message == "/agents":
            yield {"type": "text", "content": self._list_agents()}
            yield {"type": "done"}
            return

        agent = self.agents.get(self.active_agent)
        if not agent:
            yield {"type": "error", "content": "No active agent. Use /agent <name> to select one."}
            return

        yield from agent.stream_chat(message, images=images, think=think)

    def create_agent(
        self,
        name: str,
        description: str,
        system_prompt: str,
        tools: list[str] | None = None,
        model: str = "",
        temperature: float = 0.7,
        save: bool = True,
    ) -> str:
        """Create a new agent from parameters.

        Validates inputs before creating. If save=True, writes a YAML
        definition to the agents/ directory using atomic write.
        """
        # Validate inputs
        errors = self._validate_agent_params(name, description, system_prompt, tools, temperature)
        if errors:
            return f"Cannot create agent: {'; '.join(errors)}"

        # Case-insensitive duplicate check
        lower_existing = {k.lower() for k in self.agents}
        if name.lower() in lower_existing:
            existing = next(k for k in self.agents if k.lower() == name.lower())
            return f"Agent '{existing}' already exists (case-insensitive match). Choose a different name."

        config = AgentConfig(
            name=name,
            model=model,
            system_prompt=system_prompt,
            tools=tools or ["filesystem", "shell", "web"],
            temperature=temperature,
            description=description,
        )

        # Atomic: save YAML first (if requested), then register
        if save:
            agents_dir = Path(self.working_dir) / "agents"
            agents_dir.mkdir(exist_ok=True)
            yaml_path = agents_dir / f"{name}.yaml"
            data = {
                "name": name,
                "description": description,
                "model": model,
                "system_prompt": system_prompt,
                "tools": config.tools,
                "temperature": temperature,
                "max_context": config.max_context,
            }
            # Atomic write: write to temp file then rename
            _tmp_path = None
            try:
                import yaml
            except ImportError:
                log.error("PyYAML not installed, cannot save agent YAML")
                return "Cannot create agent: PyYAML not installed (pip install pyyaml)"
            try:
                content = yaml.dump(data, default_flow_style=False, sort_keys=False)
                atomic_write(yaml_path, content)
            except OSError as e:
                log.error("Failed to save agent YAML for '%s': %s", name, e)
                return f"Cannot create agent: failed to save YAML ({e})"

        agent = BaseAgent(client=self.client, config=config, working_dir=self.working_dir)
        self.register_agent(agent)

        if save:
            log.info("Created agent '%s' and saved to %s", name, yaml_path)
            return f"Agent '{name}' created and saved to {yaml_path}"

        return f"Agent '{name}' created (in-memory only)"

    def delete_agent(self, name: str) -> str:
        """Delete an agent (case-insensitive lookup)."""
        # Case-insensitive lookup (matches switch_agent behavior)
        if name not in self.agents:
            lower_map = {k.lower(): k for k in self.agents}
            if name.lower() in lower_map:
                name = lower_map[name.lower()]
            else:
                return f"Agent '{name}' not found"

        if name.lower() in {n.lower() for n in BUILTIN_AGENT_NAMES}:
            return f"Cannot delete built-in agent: {name}"

        del self.agents[name]

        # Remove YAML file if it exists
        yaml_path = Path(self.working_dir) / "agents" / f"{name}.yaml"
        if yaml_path.exists():
            yaml_path.unlink()

        if self.active_agent == name:
            self.active_agent = "assistant"

        return f"Agent '{name}' deleted"

    def _handle_checkpoint(self, message: str) -> str:
        """Route ``/checkpoint ...`` subcommands.

        Syntax:
          /checkpoint save <name>      — snapshot the active agent
          /checkpoint load <name>      — restore state into its original agent
          /checkpoint list             — show every saved checkpoint
          /checkpoint delete <name>    — remove a checkpoint
          /checkpoint                  — show usage (same as list)
        """
        parts = message.strip().split(maxsplit=2)
        if len(parts) == 1:
            return self.checkpoints.format_list()
        sub = parts[1].lower()
        name = parts[2].strip() if len(parts) >= 3 else ""

        if sub == "list":
            return self.checkpoints.format_list()

        if sub == "save":
            agent = self.agents.get(self.active_agent)
            if agent is None:
                return "No active agent to checkpoint."
            ok, msg = self.checkpoints.save(name, agent)
            return msg

        if sub == "load":
            # Peek at the checkpoint to discover which agent owns it; switch
            # to that agent before restoring so saved state lands in the
            # right conversation.
            snapshots = self.checkpoints.list()
            match = next((c for c in snapshots if c.name == name), None)
            if match is None:
                return f"Error: no checkpoint named '{name}'"
            if match.agent_name and match.agent_name in self.agents:
                self.active_agent = match.agent_name
            agent = self.agents.get(self.active_agent)
            if agent is None:
                return "No active agent to restore into."
            ok, msg = self.checkpoints.load(name, agent)
            if ok and match.agent_name:
                return f"{msg}\nActive agent: {self.active_agent}"
            return msg

        if sub == "delete":
            ok, msg = self.checkpoints.delete(name)
            return msg

        return (
            "Unknown /checkpoint subcommand. Use: "
            "save <name> | load <name> | list | delete <name>"
        )

    def _list_agents(self) -> str:
        """List all registered agents."""
        lines = ["Registered agents:\n"]
        for name, agent in self.agents.items():
            marker = " *" if name == self.active_agent else "  "
            lines.append(f"  {marker} {name:20s} {agent.config.description}")
        lines.append(f"\nActive: {self.active_agent}")
        lines.append("Use /agent <name> to switch")
        return "\n".join(lines)

    def delegate(self, agent_name: str, message: str, images: list[str] | None = None, think: bool = False) -> str:
        """Send a single message to a named agent without switching the active agent.

        Args:
            agent_name: Name of the agent to delegate to (case-insensitive).
            message: The message to send.
            images: Optional image paths for vision models.
            think: If True, enable thinking mode.

        Returns:
            The agent's response, or an error message if the agent is not found.
        """
        # Case-insensitive lookup
        resolved = agent_name
        if resolved not in self.agents:
            lower_map = {k.lower(): k for k in self.agents}
            if resolved.lower() in lower_map:
                resolved = lower_map[resolved.lower()]
            else:
                available = ", ".join(self.agents.keys())
                return f"Unknown agent: {agent_name}. Available: {available}"

        agent = self.agents[resolved]
        return agent.chat(message, images=images, think=think)

    def chain(self, agents: list[str], message: str, think: bool = False) -> str:
        """Chain multiple agents: each agent's output becomes the next agent's input.

        Args:
            agents: List of agent names to chain (in order).
            message: Initial message for the first agent.
            think: If True, enable thinking mode.

        Returns:
            Final response from the last agent in the chain.
        """
        current_message = message
        for agent_name in agents:
            response = self.delegate(agent_name, current_message, think=think)
            if response.startswith("Unknown agent:"):
                return response  # Stop on error
            current_message = response
        return current_message

    def get_all_stats(self) -> dict[str, Any]:
        """Get stats for all agents."""
        return {name: agent.get_stats() for name, agent in self.agents.items()}

    @staticmethod
    def _validate_agent_params(
        name: str,
        description: str,
        system_prompt: str,
        tools: list[str] | None,
        temperature: float,
    ) -> list[str]:
        """Validate agent creation parameters. Returns list of errors."""
        from anvil.tools import BUILTIN_TOOLS

        errors: list[str] = []

        if not name or not name.strip():
            errors.append("name cannot be empty")
        elif not name.replace("-", "").replace("_", "").isalnum():
            errors.append("name must be alphanumeric (hyphens and underscores allowed)")
        elif len(name) > MAX_AGENT_NAME_LENGTH:
            errors.append(f"name must be at most {MAX_AGENT_NAME_LENGTH} characters")
        elif name.lower() in RESERVED_NAMES:
            errors.append(f"'{name}' is a reserved name and cannot be used for agents")
        elif name.lower() in BUILTIN_AGENT_NAMES:
            errors.append(f"'{name}' is a built-in agent name and cannot be overwritten")

        if not system_prompt or not system_prompt.strip():
            errors.append("system_prompt cannot be empty")

        if not (0.0 <= temperature <= 2.0):
            errors.append(f"temperature must be 0.0-2.0, got {temperature}")

        if tools:
            unknown = [t for t in tools if t not in BUILTIN_TOOLS]
            if unknown:
                errors.append(f"unknown tools: {', '.join(unknown)} (available: {', '.join(BUILTIN_TOOLS.keys())})")

        return errors
