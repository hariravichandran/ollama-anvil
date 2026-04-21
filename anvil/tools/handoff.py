"""Handoff tool: agents can hand the conversation to another agent.

The pattern comes from OpenAI's Swarm / Agents SDK (``handoff`` is a tool
call that swaps the active agent) and from Anthropic's agent routing
guidance. In forge, every agent that loads the ``handoff`` tool can
call::

    transfer_to(agent="coder", reason="code edit needed")

which flips the orchestrator's active agent to ``"coder"``. The current
turn finishes with a short transfer-confirmation message; the next user
message is handled by the new agent.

Keeping this as a plain tool rather than a hidden orchestration protocol
means the LLM sees it in the standard tool list, the decision shows up
in chat transcripts, and it composes naturally with other patterns
(hooks can observe PreToolUse on ``transfer_to`` just like any other
tool).

Registry + switch callback injection happens after all agents are
loaded; see :meth:`forge.agents.orchestrator.AgentOrchestrator._setup_handoff_tools`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from anvil.utils.logging import get_logger

log = get_logger("tools.handoff")

__all__ = ["HandoffTool"]


class HandoffTool:
    """Expose ``transfer_to(agent, reason)`` as a tool call.

    The ``agent_registry`` and ``switch_fn`` are optional at construction
    — they are injected after all agents are registered so this tool can
    be built via the same ``_instantiate_tool`` machinery as every other
    tool.
    """

    name = "handoff"
    description = "Hand off the conversation to another specialist agent"

    def __init__(
        self,
        working_dir: str = ".",
        agent_registry: dict[str, Any] | None = None,
        switch_fn: Callable[[str], str] | None = None,
    ):
        self.working_dir = working_dir
        self._registry: dict[str, Any] = dict(agent_registry or {})
        self._switch_fn: Callable[[str], str] | None = switch_fn

    def __repr__(self) -> str:
        return f"HandoffTool(agents={sorted(self._registry)})"

    def bind(
        self,
        agent_registry: dict[str, Any],
        switch_fn: Callable[[str], str],
        exclude: str | None = None,
    ) -> None:
        """Populate the handoff target list and the switch callback.

        Called by the orchestrator once all agents are registered.
        ``exclude`` lets us skip the current agent (an agent handing off
        to itself is a no-op at best).
        """
        self._registry = {k: v for k, v in agent_registry.items() if k != exclude}
        self._switch_fn = switch_fn

    def available_agents(self) -> list[tuple[str, str]]:
        """Return ``(name, description)`` pairs for the handoff targets."""
        pairs: list[tuple[str, str]] = []
        for name, agent in sorted(self._registry.items()):
            cfg = getattr(agent, "config", None)
            desc = getattr(cfg, "description", "") if cfg else ""
            pairs.append((name, desc))
        return pairs

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return the Ollama tool definition for ``transfer_to``."""
        targets = self.available_agents()
        if targets:
            target_list = "; ".join(
                f"'{name}' ({desc})" if desc else f"'{name}'"
                for name, desc in targets
            )
            desc_text = (
                "Hand off the conversation to a specialist agent. "
                f"Available: {target_list}. Use when another agent is better "
                "equipped for the next step."
            )
        else:
            desc_text = (
                "Hand off the conversation to a specialist agent. "
                "(No other agents are currently registered.)"
            )
        return [
            {
                "type": "function",
                "function": {
                    "name": "transfer_to",
                    "description": desc_text,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent": {
                                "type": "string",
                                "description": "Name of the agent to transfer to",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Short reason for the handoff (one sentence)",
                            },
                        },
                        "required": ["agent"],
                    },
                },
            },
        ]

    def execute(self, function_name: str, args: dict[str, Any]) -> str:
        """Dispatch a handoff tool call."""
        if function_name != "transfer_to":
            return f"Unknown function: {function_name}"

        agent = str(args.get("agent", "")).strip()
        reason = str(args.get("reason", "")).strip()

        if not agent:
            return "Error: agent name is required"

        # Case-insensitive match against the registry (matches
        # AgentOrchestrator.switch_agent's behaviour).
        if agent not in self._registry:
            lower_map = {k.lower(): k for k in self._registry}
            if agent.lower() in lower_map:
                agent = lower_map[agent.lower()]
            else:
                available = ", ".join(sorted(self._registry.keys())) or "(none)"
                return f"Error: unknown agent '{agent}'. Available: {available}"

        if self._switch_fn is None:
            # Registry is populated (since we found the agent above) but
            # the switch hook isn't wired — fail loud rather than silently
            # dropping the handoff.
            return f"Error: handoff target '{agent}' exists but no switch callback is wired"

        try:
            message = self._switch_fn(agent)
        except Exception as e:  # noqa: BLE001 — defensive; callback is user-supplied
            log.warning("Handoff switch callback raised: %s", e)
            return f"Error: handoff to '{agent}' failed: {e}"

        if reason:
            return f"Transferred to '{agent}'. Reason: {reason}. ({message})"
        return f"Transferred to '{agent}'. ({message})"
