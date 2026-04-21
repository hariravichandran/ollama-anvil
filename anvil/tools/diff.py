"""Diff explainer tool — LLM-powered code change explanation.

Takes a unified diff (from git or file comparison) and uses the LLM
to explain the changes in plain language. Useful for code reviews,
understanding PRs, and documenting changes.

Usage::

    from anvil.llm.client import OllamaClient
    from anvil.tools.diff import DiffTool

    client = OllamaClient(model="qwen2.5-coder:7b")
    tool = DiffTool(client)

    # Explain a git diff
    result = tool.execute("diff_explain", {"diff": diff_text})
    print(result)  # "This change adds input validation to the login function..."

    # Summarize changes in a file
    result = tool.execute("diff_summarize", {"diff": diff_text})
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anvil.utils.logging import get_logger

if TYPE_CHECKING:
    from anvil.llm.client import OllamaClient

log = get_logger("tools.diff")

__all__ = [
    "DiffTool",
    "MAX_DIFF_LENGTH",
    "EXPLAIN_SYSTEM_PROMPT",
    "SUMMARIZE_SYSTEM_PROMPT",
    "REVIEW_SYSTEM_PROMPT",
]

# Maximum diff size to send to LLM (characters)
MAX_DIFF_LENGTH = 30_000

EXPLAIN_SYSTEM_PROMPT = (
    "You are a code change explainer. Given a unified diff, explain what "
    "changed and why in plain language. Be specific about:\n"
    "- What files were modified\n"
    "- What was added, removed, or changed\n"
    "- The likely intent behind the changes\n"
    "- Any potential issues or concerns\n"
    "Keep the explanation concise but thorough."
)

SUMMARIZE_SYSTEM_PROMPT = (
    "You are a code change summarizer. Given a unified diff, produce a "
    "brief summary (2-5 bullet points) of the key changes. Focus on the "
    "'what' and 'why', not the 'how'. Use plain language. Group related "
    "changes together."
)

REVIEW_SYSTEM_PROMPT = (
    "You are a code reviewer. Analyze this diff for:\n"
    "1. **Bugs**: Logic errors, off-by-one, null/undefined issues\n"
    "2. **Security**: Injection, auth bypass, data exposure\n"
    "3. **Performance**: N+1 queries, unnecessary allocations, blocking I/O\n"
    "4. **Style**: Naming, readability, consistency\n"
    "5. **Missing**: Error handling, edge cases, tests\n\n"
    "Rate each issue as CRITICAL, HIGH, MEDIUM, or LOW. "
    "If the code looks good, say so."
)


class DiffTool:
    """Explain and review code diffs using LLM.

    Args:
        client: OllamaClient instance.
    """

    name = "diff"
    description = "Explain, summarize, and review code diffs using AI"

    def __init__(self, client: OllamaClient):
        self._client = client

    def __repr__(self) -> str:
        return f"DiffTool(model={self._client.model!r})"

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return Ollama tool-calling definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "diff_explain",
                    "description": "Explain a code diff in plain language — what changed and why",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "diff": {
                                "type": "string",
                                "description": "Unified diff text (from git diff or similar)",
                            },
                        },
                        "required": ["diff"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "diff_summarize",
                    "description": "Produce a brief bullet-point summary of code changes",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "diff": {
                                "type": "string",
                                "description": "Unified diff text",
                            },
                        },
                        "required": ["diff"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "diff_review",
                    "description": "Review a code diff for bugs, security issues, and quality",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "diff": {
                                "type": "string",
                                "description": "Unified diff text",
                            },
                        },
                        "required": ["diff"],
                    },
                },
            },
        ]

    def execute(self, function_name: str, arguments: dict[str, Any]) -> str:
        """Execute a diff analysis function.

        Args:
            function_name: One of diff_explain, diff_summarize, diff_review.
            arguments: Must include 'diff' text.

        Returns:
            Analysis result text.
        """
        diff_text = arguments.get("diff", "")
        if not diff_text:
            return "Error: 'diff' argument is required"

        # Truncate large diffs
        if len(diff_text) > MAX_DIFF_LENGTH:
            diff_text = diff_text[:MAX_DIFF_LENGTH] + "\n\n... (diff truncated)"
            log.warning("Diff truncated from %d to %d characters", len(arguments["diff"]), MAX_DIFF_LENGTH)

        # Select system prompt
        prompts = {
            "diff_explain": EXPLAIN_SYSTEM_PROMPT,
            "diff_summarize": SUMMARIZE_SYSTEM_PROMPT,
            "diff_review": REVIEW_SYSTEM_PROMPT,
        }
        system = prompts.get(function_name)
        if not system:
            return f"Error: Unknown function '{function_name}'"

        # Send to LLM
        try:
            result = self._client.chat(
                messages=[{"role": "user", "content": f"```diff\n{diff_text}\n```"}],
                system=system,
                timeout=120,
            )
            return result.get("response", "No response from model")
        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
            return f"Error analyzing diff: {exc}"

    def explain(self, diff_text: str) -> str:
        """Convenience: explain a diff in plain language."""
        return self.execute("diff_explain", {"diff": diff_text})

    def summarize(self, diff_text: str) -> str:
        """Convenience: summarize a diff as bullet points."""
        return self.execute("diff_summarize", {"diff": diff_text})

    def review(self, diff_text: str) -> str:
        """Convenience: review a diff for issues."""
        return self.execute("diff_review", {"diff": diff_text})
