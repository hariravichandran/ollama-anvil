"""Base agent: chat loop with tool use, memory, and context compression."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

from anvil.agents.dry_run import DryRunConfig, build_preview
from anvil.agents.permissions import PermissionManager
from anvil.agents.rules import build_glob_rules_fragment, load_glob_rules, load_project_rules
from anvil.agents.skills import SkillLibrary
from anvil.agents.tool_validation import format_errors, validate_arguments
from anvil.hooks import HookDecision, HookEvent, HookManager, load_hooks
from anvil.llm.client import OllamaClient
from anvil.llm.context import ContextCompressor
from anvil.observability import current_tracer
from anvil.tools import BUILTIN_TOOLS
from anvil.utils.logging import get_logger

log = get_logger("agents.base")

__all__ = [
    "AgentConfig",
    "BaseAgent",
    "load_agent_from_yaml",
    "MAX_USER_MESSAGE_LENGTH",
    "MAX_CONVERSATION_MESSAGES",
    "MAX_TOOL_RESULT_LENGTH",
    "MAX_ERROR_MESSAGE_LENGTH",
    "CIRCUIT_BREAKER_RESET_TIME",
    "CIRCUIT_BREAKER_CLEANUP_INTERVAL",
    "MAX_CIRCUIT_BREAKER_ENTRIES",
    "LLM_TRANSIENT_RETRIES",
    "LLM_RETRY_BASE_DELAY",
    "LOG_PREVIEW_LENGTH",
    "TOOL_EXECUTION_TIMEOUT",
    "TOOL_CACHE_MAX_SIZE",
    "CACHEABLE_TOOL_FUNCTIONS",
    "MAX_PARALLEL_TOOL_CALLS",
]

# Safety limits
MAX_USER_MESSAGE_LENGTH = 50_000  # 50K chars per message
MAX_CONVERSATION_MESSAGES = 1000  # Max messages before forced compression
MAX_TOOL_RESULT_LENGTH = 50_000  # Truncate oversized tool results
MAX_ERROR_MESSAGE_LENGTH = 500  # Truncate error messages
CIRCUIT_BREAKER_RESET_TIME = 300  # Auto-reset circuit breaker after 5 minutes
CIRCUIT_BREAKER_CLEANUP_INTERVAL = 50  # Clean stale entries every N tool calls
MAX_CIRCUIT_BREAKER_ENTRIES = 100  # Max tracked tool functions
LLM_TRANSIENT_RETRIES = 2  # Retry transient LLM errors before giving up
LLM_RETRY_BASE_DELAY = 1.0  # Base delay (seconds) for exponential backoff
LOG_PREVIEW_LENGTH = 200  # Truncate log previews (tool args, errors)
TOOL_EXECUTION_TIMEOUT = 120  # Max seconds for a single tool call
TOOL_CACHE_MAX_SIZE = 200  # Max cached tool results per session
MAX_PARALLEL_TOOL_CALLS = 8  # Max tool calls to run concurrently in one round

# Read-only tool functions whose results can be cached within a session
CACHEABLE_TOOL_FUNCTIONS = frozenset({
    "read_file", "list_directory", "find_files",
    "codebase_search", "find_symbol", "project_overview", "file_summary",
    "git_status", "git_log", "git_diff",
    "web_search",
})


@dataclass(slots=True)
class AgentConfig:
    """Configuration for an agent."""

    name: str = "assistant"
    model: str = ""  # uses default from hardware profile if empty
    system_prompt: str = (
        "You are a helpful AI assistant running locally via Ollama. "
        "You can use tools to help answer questions and complete tasks. "
        "Be concise and practical."
    )
    tools: list[str] = field(default_factory=lambda: ["filesystem", "shell", "web"])
    temperature: float = 0.7
    max_context: int = 8192
    description: str = "General-purpose assistant"


class BaseAgent:
    """Core agent with chat loop, tool dispatch, and context management.

    This is the foundation for all agents. Users interact with it directly
    for general queries, or it delegates to specialized agents via the orchestrator.
    """

    def __init__(
        self,
        client: OllamaClient,
        config: AgentConfig | None = None,
        working_dir: str = ".",
        permissions: PermissionManager | None = None,
        hooks: HookManager | None = None,
        skills: SkillLibrary | None = None,
        dry_run: bool | DryRunConfig = False,
    ):
        self.client = client
        self.config = config or AgentConfig()
        self.working_dir = working_dir
        self.permissions = permissions or PermissionManager()
        self.hooks = hooks if hooks is not None else HookManager(configs=load_hooks(working_dir))
        self.skills = skills if skills is not None else SkillLibrary.discover(working_dir)
        # Dry-run accepts either a bool shortcut or a full DryRunConfig.
        # When True/False we build a default config matching that flag.
        if isinstance(dry_run, DryRunConfig):
            self.dry_run = dry_run
        else:
            self.dry_run = DryRunConfig(enabled=bool(dry_run))
        self.messages: list[dict[str, str]] = []
        self.compressor = ContextCompressor(
            client=client,
            max_tokens=self.config.max_context,
        )

        # Load monolithic project rules (.anvil-rules / CLAUDE.md) once.
        rules = load_project_rules(working_dir)
        if rules:
            self._system_prompt = (
                f"{self.config.system_prompt}\n\n"
                f"--- Project Rules ---\n{rules}"
            )
        else:
            self._system_prompt = self.config.system_prompt

        # Glob-scoped rules are loaded once but matched per turn — keeps
        # the injection set tight to what's actually relevant right now.
        self._glob_rules = load_glob_rules(working_dir)

        # Initialize tools
        self._tools: dict[str, Any] = {}
        for tool_name in self.config.tools:
            tool_class = BUILTIN_TOOLS.get(tool_name)
            if tool_class:
                self._tools[tool_name] = self._instantiate_tool(
                    tool_class, working_dir, client,
                )

        # Bind tools that need a reference to the agent's live SkillLibrary
        # so saved skills are visible immediately (matches HandoffTool's
        # late-binding pattern).
        skill_author = self._tools.get("skill_author")
        if skill_author is not None and hasattr(skill_author, "bind"):
            skill_author.bind(self.skills)

        # Cache tool definitions (rebuilt only when tools change)
        self._cached_tool_defs: list[dict[str, Any]] | None = None

        # Build function_name → tool instance map for O(1) dispatch and
        # function_name → parameter-schema map for argument validation.
        self._function_tool_map: dict[str, Any] = {}
        self._function_schema_map: dict[str, dict[str, Any]] = {}
        for tool in self._tools.values():
            for defn in tool.get_tool_definitions():
                func_def = defn.get("function")
                if isinstance(func_def, dict):
                    func_name = func_def.get("name", "")
                    if func_name:
                        self._function_tool_map[func_name] = tool
                        params = func_def.get("parameters")
                        if isinstance(params, dict):
                            self._function_schema_map[func_name] = params

        # Circuit breaker: track consecutive failures per tool function
        self._tool_failure_counts: dict[str, int] = {}
        self._tool_failure_times: dict[str, float] = {}  # last failure time per tool
        self._tool_circuit_threshold = 3  # open circuit after N consecutive failures
        self._tool_call_count = 0  # total tool calls for periodic cleanup
        self._circuit_breaker_lock = threading.Lock()

        # Tool result cache for read-only operations
        self._tool_cache: dict[str, str] = {}
        self._tool_cache_hits = 0
        self._tool_cache_misses = 0

        # Tool call history for debugging and review
        self._tool_history: list[dict[str, Any]] = []

    @staticmethod
    def _instantiate_tool(tool_class: type, working_dir: str, client: Any) -> Any:
        """Instantiate a tool, passing the right constructor args.

        Inspects the ``__init__`` signature to determine which kwargs to pass:
        client-based tools get ``client``, dir-based tools get the directory
        param (``working_dir`` or ``project_dir``).
        """
        import inspect

        sig = inspect.signature(tool_class.__init__)
        params = sig.parameters
        kwargs: dict[str, Any] = {}
        if "client" in params:
            kwargs["client"] = client
        if "working_dir" in params:
            kwargs["working_dir"] = working_dir
        elif "project_dir" in params:
            kwargs["project_dir"] = working_dir
        return tool_class(**kwargs)

    def __repr__(self) -> str:
        return f"BaseAgent(name={self.config.name!r}, model={self.config.model!r}, tools={len(self._tools)})"

    def __enter__(self) -> BaseAgent:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> bool:
        """Exit context manager, closing all tools that support close()."""
        self.close()
        return False

    def close(self) -> None:
        """Close all tools that have a close() method."""
        for tool in self._tools.values():
            close_fn = getattr(tool, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except OSError as e:
                    log.debug("Error closing tool %s: %s", getattr(tool, "name", "?"), e)

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions for Ollama tool calling (cached)."""
        if self._cached_tool_defs is None:
            definitions = []
            for tool in self._tools.values():
                definitions.extend(tool.get_tool_definitions())
            self._cached_tool_defs = definitions
        return self._cached_tool_defs

    def chat(self, user_message: str, images: list[str] | None = None, think: bool = False) -> str:
        """Process a user message and return the agent's response.

        Handles tool calls automatically — loops until the agent produces
        a final text response (no more tool calls).

        Args:
            user_message: The user's input text.
            images: Optional list of image paths or base64 strings for vision models.
            think: If True, enable thinking/reasoning mode for chain-of-thought.
        """
        # Validate and truncate oversized messages
        if len(user_message) > MAX_USER_MESSAGE_LENGTH:
            log.warning("User message truncated from %d to %d chars", len(user_message), MAX_USER_MESSAGE_LENGTH)
            user_message = user_message[:MAX_USER_MESSAGE_LENGTH] + "\n... (truncated)"

        # UserPromptSubmit hook — may rewrite the prompt or block it entirely.
        submit = self.hooks.run(
            HookEvent.USER_PROMPT_SUBMIT,
            {"prompt": user_message, "agent": self.config.name},
        )
        if submit.decision == HookDecision.DENY:
            denial = submit.message or "The user prompt was blocked by a UserPromptSubmit hook."
            self.messages.append({"role": "assistant", "content": denial})
            return denial
        if submit.updated_input is not None:
            new_prompt = submit.updated_input.get("prompt")
            if isinstance(new_prompt, str) and new_prompt:
                user_message = new_prompt

        self.messages.append({"role": "user", "content": user_message})

        # Enforce conversation history limit
        if len(self.messages) > MAX_CONVERSATION_MESSAGES:
            log.warning("Conversation exceeded %d messages, trimming oldest", MAX_CONVERSATION_MESSAGES)
            self.messages = self.messages[-MAX_CONVERSATION_MESSAGES:]

        # Compress context if needed
        compressed = self.compressor.compress(self.messages)

        # Build messages with system prompt (includes project rules if any).
        # Two per-turn injections: glob-scoped rules (whose patterns match
        # the working dir or file paths in the user message) and skills
        # (matched against the message via triggers + token overlap).
        turn_prompt = self._system_prompt
        rules_frag = build_glob_rules_fragment(self._glob_rules, user_message, self.working_dir) \
            if self._glob_rules else ""
        if rules_frag:
            turn_prompt = f"{turn_prompt}\n\n{rules_frag}"
        skill_frag = self.skills.build_injection(user_message) if self.skills else ""
        if skill_frag:
            turn_prompt = f"{turn_prompt}\n\n{skill_frag}"
        messages = [{"role": "system", "content": turn_prompt}] + compressed

        # Get tool definitions
        tools = self.get_tool_definitions()

        max_tool_rounds = 10
        for round_idx in range(max_tool_rounds):
            # Only send images on the first round
            round_images = images if round_idx == 0 else None
            result = self._chat_with_retry(messages, tools, self.config.temperature, images=round_images, think=think)

            if "error" in result:
                error_text = str(result['error'])[:MAX_ERROR_MESSAGE_LENGTH]
                error_msg = f"LLM error: {error_text}"
                log.error(error_msg)
                self.messages.append({"role": "assistant", "content": error_msg})
                return error_msg

            # Check for tool calls
            tool_calls = result.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                # Add assistant message with tool calls
                assistant_msg = result.get("response", "")
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})

                # Execute all tool calls in this round (parallel when >1)
                for name, tool_result in self._execute_tool_calls(tool_calls):
                    log.info("Tool result: %s (%d chars)", name, len(tool_result))
                    messages.append({"role": "tool", "content": tool_result})
            else:
                # Final response — no tool calls
                response = result.get("response", "")
                self.messages.append({"role": "assistant", "content": response})
                return response

        return "Max tool rounds reached. Please try a simpler request."

    def stream_chat(self, user_message: str, images: list[str] | None = None, think: bool = False) -> Generator[dict[str, Any], None, None]:
        """Stream a response, yielding typed event dicts.

        Yields dicts with 'type' key: 'text', 'tool_call', 'done', 'error'.
        Handles tool calls inline — executes tools and includes results in stream.

        Args:
            user_message: The user's input text.
            images: Optional list of image paths or base64 strings for vision models.
            think: If True, enable thinking mode — yields 'thinking' events.
        """
        # Validate and truncate oversized messages (parity with chat())
        if len(user_message) > MAX_USER_MESSAGE_LENGTH:
            log.warning("User message truncated from %d to %d chars", len(user_message), MAX_USER_MESSAGE_LENGTH)
            user_message = user_message[:MAX_USER_MESSAGE_LENGTH] + "\n... (truncated)"

        # UserPromptSubmit hook — may rewrite the prompt or block it entirely.
        submit = self.hooks.run(
            HookEvent.USER_PROMPT_SUBMIT,
            {"prompt": user_message, "agent": self.config.name},
        )
        if submit.decision == HookDecision.DENY:
            denial = submit.message or "The user prompt was blocked by a UserPromptSubmit hook."
            self.messages.append({"role": "assistant", "content": denial})
            yield {"type": "text", "content": denial}
            yield {"type": "done"}
            return
        if submit.updated_input is not None:
            new_prompt = submit.updated_input.get("prompt")
            if isinstance(new_prompt, str) and new_prompt:
                user_message = new_prompt

        self.messages.append({"role": "user", "content": user_message})

        # Enforce conversation history limit (parity with chat())
        if len(self.messages) > MAX_CONVERSATION_MESSAGES:
            log.warning("Conversation exceeded %d messages, trimming oldest", MAX_CONVERSATION_MESSAGES)
            self.messages = self.messages[-MAX_CONVERSATION_MESSAGES:]

        compressed = self.compressor.compress(self.messages)
        turn_prompt = self._system_prompt
        rules_frag = build_glob_rules_fragment(self._glob_rules, user_message, self.working_dir) \
            if self._glob_rules else ""
        if rules_frag:
            turn_prompt = f"{turn_prompt}\n\n{rules_frag}"
        skill_frag = self.skills.build_injection(user_message) if self.skills else ""
        if skill_frag:
            turn_prompt = f"{turn_prompt}\n\n{skill_frag}"
        messages = [{"role": "system", "content": turn_prompt}] + compressed
        tools = self.get_tool_definitions()

        full_response = []
        tool_results = []
        for event in self.client.stream_chat(messages, tools=tools if tools else None, images=images, think=think):
            event_type = event.get("type", "")

            if event_type == "text":
                full_response.append(event.get("content", ""))
                yield event
            elif event_type == "tool_call":
                # Execute all tool calls in this event (parallel when >1)
                batch = event.get("tool_calls", [])
                for name, tool_result in self._execute_tool_calls(batch):
                    tool_results.append({"name": name, "result": tool_result})
                    yield {"type": "tool_result", "name": name, "result": tool_result}
                yield event
            elif event_type == "done":
                yield event
                break
            elif event_type == "error":
                yield event
                break
            else:
                yield event

        # Store tool results in conversation history so they persist across turns
        for tr in tool_results:
            self.messages.append({"role": "tool", "content": tr["result"]})
        self.messages.append({"role": "assistant", "content": "".join(full_response)})

    def _chat_with_retry(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        images: list[str] | None = None,
        think: bool = False,
    ) -> dict[str, Any]:
        """Call the LLM with retry for transient errors.

        Retries on LLM-level errors (model busy, temporary failures) with
        exponential backoff. Connection-level retries are handled by
        OllamaClient; this handles higher-level transient failures.
        """
        tracer = current_tracer()
        with tracer.span("llm.chat", kind="CLIENT", attributes={
            "agent.name": self.config.name,
            "llm.model": self.client.model,
            "llm.temperature": temperature,
            "llm.think": bool(think),
            "llm.messages": len(messages),
            "llm.tools": len(tools) if tools else 0,
        }) as span:
            for attempt in range(1 + LLM_TRANSIENT_RETRIES):
                result = self.client.chat(
                    messages=messages,
                    tools=tools if tools else None,
                    temperature=temperature,
                    images=images,
                    think=think,
                )
                if "error" not in result:
                    tool_calls = result.get("tool_calls") or []
                    span.set_attribute("llm.response_tool_calls", len(tool_calls))
                    span.set_attribute("llm.response_chars", len(result.get("response", "")))
                    return result
                # Don't retry on final attempt
                if attempt >= LLM_TRANSIENT_RETRIES:
                    span.set_attribute("llm.error", str(result.get("error", ""))[:200])
                    span.status = "ERROR"
                    return result
                delay = LLM_RETRY_BASE_DELAY * (2 ** attempt)
                log.warning(
                    "Transient LLM error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, 1 + LLM_TRANSIENT_RETRIES, delay,
                    str(result["error"])[:LOG_PREVIEW_LENGTH],
                )
                time.sleep(delay)
            return result  # unreachable, but satisfies type checker

    def _execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """Execute a batch of tool calls from one assistant turn.

        When the model emits a single tool call we dispatch it directly to
        avoid thread-pool overhead. When it emits 2+ calls we run them
        concurrently using a bounded thread pool, preserving input order in
        the returned list so the LLM sees results in the order it requested.

        ``_execute_tool`` is already thread-safe (circuit breaker state is
        lock-protected and each tool runs in its own ThreadPoolExecutor for
        timeout enforcement), so fan-out is safe.

        Returns a list of ``(function_name, result_text)`` tuples.
        """
        parsed: list[tuple[str, dict[str, Any]]] = []
        for tc in tool_calls:
            func = tc.get("function", {}) if isinstance(tc, dict) else {}
            name = func.get("name", "") if isinstance(func, dict) else ""
            args = func.get("arguments", {}) if isinstance(func, dict) else {}
            if not isinstance(args, dict):
                args = {}
            parsed.append((name, args))

        if not parsed:
            return []

        if len(parsed) == 1:
            name, args = parsed[0]
            log.info("Tool call: %s(%s)", name, json.dumps(args)[:LOG_PREVIEW_LENGTH])
            return [(name, self._execute_tool(name, args))]

        # Multiple calls in one round — run them concurrently.
        from concurrent.futures import ThreadPoolExecutor

        log.info("Parallel tool batch: %d calls", len(parsed))
        max_workers = min(len(parsed), MAX_PARALLEL_TOOL_CALLS)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="anvil-tool") as executor:
            futures = [
                executor.submit(self._execute_tool, name, args)
                for name, args in parsed
            ]
            # ``future.result()`` preserves submission order, so zip gives
            # us aligned (name, result) tuples even if completion reorders.
            results = [f.result() for f in futures]

        return [(name, result) for (name, _args), result in zip(parsed, results, strict=True)]

    def _execute_tool(self, function_name: str, args: dict[str, Any]) -> str:
        """Route a tool call to the appropriate tool handler.

        Implements a circuit breaker pattern: after N consecutive failures
        for the same tool function, the circuit "opens" and the tool is
        temporarily skipped to prevent cascading failures.

        Checks permissions before executing. If the user denies
        the action, returns a message indicating denial.
        Wraps execution in try/except to prevent tool errors from
        crashing the chat loop.
        """
        with self._circuit_breaker_lock:
            self._tool_call_count += 1
            # Periodic cleanup of stale circuit breaker entries
            if self._tool_call_count % CIRCUIT_BREAKER_CLEANUP_INTERVAL == 0:
                self._cleanup_stale_circuit_breakers()

            # Circuit breaker: skip tools that have failed too many times in a row
            failure_count = self._tool_failure_counts.get(function_name, 0)
            if failure_count >= self._tool_circuit_threshold:
                # Auto-reset if enough time has passed since last failure
                last_fail = self._tool_failure_times.get(function_name, 0)
                if time.time() - last_fail > CIRCUIT_BREAKER_RESET_TIME:
                    log.info("Circuit breaker auto-reset for '%s' (%.0fs elapsed)", function_name,
                             time.time() - last_fail)
                    self._tool_failure_counts[function_name] = 0
                    failure_count = 0
                else:
                    log.warning(
                        "Circuit breaker open for '%s' (%d consecutive failures), skipping",
                        function_name, failure_count,
                    )
                    return (
                        f"Tool '{function_name}' is temporarily unavailable after "
                        f"{failure_count} consecutive failures. Try a different approach."
                    )

        # PreToolUse hooks — may rewrite args or deny the call outright.
        pre = self.hooks.run(
            HookEvent.PRE_TOOL_USE,
            {"tool": {"function": function_name, "arguments": dict(args)}, "agent": self.config.name},
            tool_name=function_name,
        )
        if pre.decision == HookDecision.DENY:
            denial = pre.message or f"Tool '{function_name}' was blocked by a PreToolUse hook."
            log.info("PreToolUse denied %s: %s", function_name, denial[:LOG_PREVIEW_LENGTH])
            return denial
        if pre.updated_input is not None:
            log.debug("PreToolUse rewrote args for %s", function_name)
            args = pre.updated_input

        # Validate args against the tool's advertised JSON Schema. Any
        # error becomes a structured tool result; the agent loop picks
        # this up and the model retries with corrected arguments.
        schema = self._function_schema_map.get(function_name)
        if schema is not None:
            issues = validate_arguments(schema, args)
            if issues:
                log.info("Tool arg validation failed for %s: %d issue(s)", function_name, len(issues))
                return format_errors(issues, tool_name=function_name)

        # Dry-run: intercept destructive tool calls before they hit the
        # tool. Read-only operations fall through untouched so the agent
        # can still explore the codebase.
        if self.dry_run.enabled and self.dry_run.is_destructive(function_name):
            log.info("Dry-run intercept: %s", function_name)
            return build_preview(function_name, args)

        # Check permission before executing
        if not self.permissions.check(function_name, context=args):
            log.info("Permission denied for %s", function_name)
            return f"Action '{function_name}' was denied by the user."

        # Check tool result cache for read-only operations
        cache_key = ""
        is_cacheable = function_name in CACHEABLE_TOOL_FUNCTIONS
        if is_cacheable:
            import json as _json
            cache_key = f"{function_name}:{_json.dumps(args, sort_keys=True)}"
            if cache_key in self._tool_cache:
                log.debug("Tool cache hit: %s", function_name)
                self._tool_cache_hits += 1
                return self._tool_cache[cache_key]
            self._tool_cache_misses += 1

        # Use cached function→tool mapping for O(1) lookup
        tool = self._function_tool_map.get(function_name)
        if tool:
            call_start = time.time()
            tracer = current_tracer()
            trace_span = tracer.span("tool.execute", kind="INTERNAL", attributes={
                "tool.function": function_name,
                "tool.name": getattr(tool, "name", ""),
                "agent.name": self.config.name,
            })
            try:
                result = self._run_tool_with_timeout(tool, function_name, args)
                trace_span.set_attribute("tool.result_chars", len(result))
                trace_span.end()
                # Success — reset failure counter
                with self._circuit_breaker_lock:
                    self._tool_failure_counts[function_name] = 0
                # Record in history
                self._tool_history.append({
                    "function": function_name,
                    "args": args,
                    "status": "ok",
                    "time_s": round(time.time() - call_start, 3),
                })
                # Truncate oversized results to prevent context overflow
                if len(result) > MAX_TOOL_RESULT_LENGTH:
                    log.debug("Truncating tool result from %d to %d chars", len(result), MAX_TOOL_RESULT_LENGTH)
                    result = result[:MAX_TOOL_RESULT_LENGTH] + "\n... (output truncated)"
                # PostToolUse hooks — observability / post-processing.
                post = self.hooks.run(
                    HookEvent.POST_TOOL_USE,
                    {
                        "tool": {"function": function_name, "arguments": dict(args)},
                        "agent": self.config.name,
                        "result": result,
                    },
                    tool_name=function_name,
                )
                if post.updated_input is not None:
                    # A PostToolUse hook can return ``{"updatedInput": {"result": "..."}}``
                    # to rewrite the tool output before the LLM sees it
                    # (e.g., redact secrets).
                    maybe = post.updated_input.get("result")
                    if isinstance(maybe, str):
                        result = maybe
                # Cache read-only results
                if is_cacheable and len(self._tool_cache) < TOOL_CACHE_MAX_SIZE:
                    self._tool_cache[cache_key] = result
                return result
            except (TypeError, ValueError, OSError, KeyError) as e:
                log.error("Tool execution error for %s: %s", function_name, e)
                trace_span.record_exception(e)
                trace_span.end()
                self._tool_history.append({
                    "function": function_name,
                    "args": args,
                    "status": "error",
                    "error": str(e)[:MAX_ERROR_MESSAGE_LENGTH],
                    "time_s": round(time.time() - call_start, 3),
                })
                with self._circuit_breaker_lock:
                    # Re-read count under lock to avoid TOCTOU race with concurrent calls
                    self._tool_failure_counts[function_name] = self._tool_failure_counts.get(function_name, 0) + 1
                    self._tool_failure_times[function_name] = time.time()
                error_str = str(e)[:MAX_ERROR_MESSAGE_LENGTH]
                return f"Tool error in '{function_name}': {error_str}"

        return f"Unknown tool function: {function_name}"

    @staticmethod
    def _run_tool_with_timeout(tool: Any, function_name: str, args: dict[str, Any]) -> str:
        """Execute a tool with a timeout to prevent hanging.

        Uses concurrent.futures.ThreadPoolExecutor to enforce the timeout.
        If the tool takes longer than TOOL_EXECUTION_TIMEOUT, returns an
        error message instead of blocking indefinitely.
        """
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FuturesTimeout

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(tool.execute, function_name, args)
            try:
                return future.result(timeout=TOOL_EXECUTION_TIMEOUT)
            except FuturesTimeout:
                log.warning(
                    "Tool '%s' timed out after %ds", function_name, TOOL_EXECUTION_TIMEOUT,
                )
                return (
                    f"Tool '{function_name}' timed out after {TOOL_EXECUTION_TIMEOUT}s. "
                    f"Try a simpler request or break it into smaller steps."
                )

    def _cleanup_stale_circuit_breakers(self) -> None:
        """Remove stale circuit breaker entries that have expired."""
        now = time.time()
        stale_keys = [
            k for k, t in self._tool_failure_times.items()
            if now - t > CIRCUIT_BREAKER_RESET_TIME
        ]
        for k in stale_keys:
            self._tool_failure_counts.pop(k, None)
            self._tool_failure_times.pop(k, None)
        if stale_keys:
            log.debug("Cleaned up %d stale circuit breaker entries", len(stale_keys))

        # Enforce hard cap: evict oldest entries when over limit
        while len(self._tool_failure_times) > MAX_CIRCUIT_BREAKER_ENTRIES:
            oldest = min(self._tool_failure_times, key=self._tool_failure_times.get)
            self._tool_failure_counts.pop(oldest, None)
            self._tool_failure_times.pop(oldest, None)

    def reset_circuit_breaker(self, function_name: str | None = None) -> None:
        """Reset circuit breaker for a specific tool or all tools.

        Call this when the underlying issue has been resolved
        (e.g., a service restarted, a file restored).
        """
        with self._circuit_breaker_lock:
            if function_name:
                self._tool_failure_counts.pop(function_name, None)
            else:
                self._tool_failure_counts.clear()

    def reset(self) -> None:
        """Clear conversation history, context cache, and tool cache."""
        self.messages.clear()
        self.compressor.reset()
        self._tool_cache.clear()
        self._tool_cache_hits = 0
        self._tool_cache_misses = 0
        self._tool_history.clear()

    def get_tool_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent tool call history.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of dicts with function, args, status, time_s, and optionally error.
        """
        return self._tool_history[-limit:]

    def summarize(self) -> str:
        """Generate a summary of the current conversation using the LLM.

        Returns a concise summary or an empty string if there are no messages.
        """
        if not self.messages:
            return ""

        # Build a condensed transcript
        lines: list[str] = []
        for msg in self.messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "tool":
                continue  # Skip tool results for brevity
            if content:
                # Truncate long messages
                preview = content[:500] + "..." if len(content) > 500 else content
                lines.append(f"{role}: {preview}")

        transcript = "\n".join(lines)
        prompt = (
            "Summarize this conversation in 3-5 bullet points. "
            "Focus on key topics discussed, decisions made, and any action items.\n\n"
            f"{transcript}"
        )
        try:
            result = self.client.chat(
                messages=[{"role": "user", "content": prompt}],
                system=self.config.system_prompt,
            )
            if "error" in result:
                log.warning("Failed to summarize conversation: %s", result.get("error"))
                return ""
            return result.get("response", "")
        except (OSError, TimeoutError, ValueError) as e:
            log.warning("Summarization failed: %s", e)
            return ""

    def compact(self, threshold_pct: float = 80.0) -> dict[str, Any]:
        """Compact conversation history when context usage exceeds threshold.

        Summarizes older messages and replaces them with the summary,
        keeping only the most recent messages. This frees context window
        space for continued conversation on memory-constrained systems.

        Args:
            threshold_pct: Context usage percentage above which to compact.

        Returns:
            Dict with 'compacted' (bool), 'old_count', 'new_count',
            'freed_tokens', and 'summary' (str if compacted).
        """
        est_tokens = self.compressor.estimate_tokens(self.messages)
        max_tokens = self.config.max_context
        usage_pct = (est_tokens / max(1, max_tokens)) * 100

        if usage_pct < threshold_pct or len(self.messages) < 6:
            return {
                "compacted": False,
                "old_count": len(self.messages),
                "new_count": len(self.messages),
                "freed_tokens": 0,
                "summary": "",
            }

        old_count = len(self.messages)

        # Generate summary of older messages
        summary_text = self.summarize()
        if not summary_text:
            return {
                "compacted": False,
                "old_count": old_count,
                "new_count": old_count,
                "freed_tokens": 0,
                "summary": "",
            }

        # Keep the most recent 4 messages (2 exchanges)
        keep_recent = 4
        recent = self.messages[-keep_recent:]

        # Replace history with summary + recent messages
        self.messages = [
            {"role": "system", "content": f"[Conversation summary]: {summary_text}"},
        ] + recent

        new_tokens = self.compressor.estimate_tokens(self.messages)
        freed = est_tokens - new_tokens

        log.info(
            "Compacted conversation: %d→%d messages, freed ~%d tokens",
            old_count, len(self.messages), freed,
        )

        return {
            "compacted": True,
            "old_count": old_count,
            "new_count": len(self.messages),
            "freed_tokens": freed,
            "summary": summary_text,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get agent usage statistics."""
        est_tokens = self.compressor.estimate_tokens(self.messages)
        total_cache_requests = self._tool_cache_hits + self._tool_cache_misses
        return {
            "name": self.config.name,
            "model": self.client.model,
            "messages": len(self.messages),
            "context": {
                "estimated_tokens": est_tokens,
                "max_tokens": self.config.max_context,
                "usage_pct": round(est_tokens / max(1, self.config.max_context) * 100, 1),
            },
            "tools": {
                "total_calls": self._tool_call_count,
                "cache_hits": self._tool_cache_hits,
                "cache_misses": self._tool_cache_misses,
                "cache_hit_rate": round(
                    self._tool_cache_hits / max(1, total_cache_requests) * 100, 1
                ),
                "cached_entries": len(self._tool_cache),
            },
            "llm_stats": {
                "total_calls": self.client.stats.total_calls,
                "total_tokens": self.client.stats.total_tokens,
                "avg_time_s": round(self.client.stats.avg_time_s, 2),
                "errors": self.client.stats.errors,
            },
        }


def load_agent_from_yaml(yaml_path: str, client: OllamaClient, working_dir: str = ".") -> BaseAgent:
    """Load an agent from a YAML configuration file."""
    from pathlib import Path

    import yaml

    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Agent config not found: {yaml_path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    config = AgentConfig(
        name=data.get("name", path.stem),
        model=data.get("model", ""),
        system_prompt=data.get("system_prompt", AgentConfig.system_prompt),
        tools=data.get("tools", ["filesystem", "shell", "web"]),
        temperature=data.get("temperature", 0.7),
        max_context=data.get("max_context", 8192),
        description=data.get("description", ""),
    )

    if config.model:
        client.switch_model(config.model)

    return BaseAgent(client=client, config=config, working_dir=working_dir)
