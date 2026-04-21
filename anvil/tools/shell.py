"""Shell tool: execute commands with user approval."""

from __future__ import annotations

import re
import subprocess
import time
from typing import Any

from anvil.utils.logging import get_logger

log = get_logger("tools.shell")

__all__ = [
    "ShellTool",
    "DEFAULT_MAX_OUTPUT",
    "MAX_COMMAND_HISTORY",
    "DEFAULT_TIMEOUT",
    "MIN_TIMEOUT",
    "MAX_TIMEOUT",
    "COMMAND_SUMMARY_LENGTH",
]

# Default output truncation limit (characters)
DEFAULT_MAX_OUTPUT = 10000

# Max command history entries to retain (prevents unbounded memory growth)
MAX_COMMAND_HISTORY = 500

# Timeout bounds (seconds)
DEFAULT_TIMEOUT = 30
MIN_TIMEOUT = 1
MAX_TIMEOUT = 300
COMMAND_SUMMARY_LENGTH = 80  # Truncate command strings in logs/metrics

# Pre-compiled dangerous command regex patterns (avoids recompilation per call)
_DANGEROUS_REGEXES = [
    re.compile(pat) for pat in [
        r"curl\s+.*\|\s*(sh|bash)",     # curl URL | sh/bash
        r"wget\s+.*\|\s*(sh|bash)",     # wget URL | sh/bash
        r"chmod\s+-r\s+777\s+/",        # chmod -R 777 /
        r"\beval\s+[\"']",              # eval "..." — arbitrary code execution
        r"\bexec\s+\d*[<>]",            # exec redirections (fd manipulation)
        r"`[^`]+`",                     # backtick command substitution in dangerous context
    ]
]


class ShellTool:
    """Execute shell commands with safety checks."""

    name = "shell"
    description = "Execute shell commands with user approval"

    def __init__(self, working_dir: str = ".", auto_approve: bool = False):
        self.working_dir = working_dir
        self.auto_approve = auto_approve
        # Track command execution durations for observability
        self._command_durations: list[tuple[str, float]] = []  # (cmd_summary, seconds)

    def __repr__(self) -> str:
        return f"ShellTool(working_dir={self.working_dir!r})"

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return Ollama tool-calling definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Execute a shell command and return its output",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "The shell command to execute"},
                            "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
                        },
                        "required": ["command"],
                    },
                },
            },
        ]

    def execute(self, function_name: str, args: dict[str, Any]) -> str:
        """Execute a shell tool function."""
        if function_name != "run_command":
            return f"Unknown function: {function_name}"

        command = args.get("command", "").strip()
        try:
            timeout = int(args.get("timeout", DEFAULT_TIMEOUT))
        except (TypeError, ValueError):
            timeout = DEFAULT_TIMEOUT
        timeout = max(MIN_TIMEOUT, min(timeout, MAX_TIMEOUT))

        if not command:
            return "Error: empty command"

        # Safety check: block destructive commands
        if self._is_dangerous(command):
            return f"Blocked: '{command}' appears destructive. Use with caution."

        # Safety check: block interactive commands that would hang
        if self._is_interactive(command):
            return f"Blocked: '{command}' requires interactive input and would hang. Use a non-interactive alternative."

        return self._run(command, timeout)

    def _run(self, command: str, timeout: int = 30) -> str:
        """Run a shell command and return output.

        Uses Popen for proper process lifecycle management. Tracks execution
        duration and uses graceful shutdown (SIGTERM then SIGKILL) for
        timed-out processes to prevent orphaned child processes.
        """
        log.info("Running: %s", command)
        start = time.time()
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.working_dir,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                elapsed = time.time() - start
                self._command_durations.append((command[:COMMAND_SUMMARY_LENGTH], elapsed))
                # Create a TimeoutExpired with the real PID for tree kill
                err = subprocess.TimeoutExpired(command, timeout)
                err.pid = proc.pid  # type: ignore[attr-defined]
                self._kill_process_tree(err)
                # Ensure the process is fully reaped
                proc.kill()
                proc.wait(timeout=5)
                return f"Command timed out after {timeout}s: {command}"

            elapsed = time.time() - start
            cmd_summary = command[:COMMAND_SUMMARY_LENGTH]
            self._command_durations.append((cmd_summary, elapsed))
            if len(self._command_durations) > MAX_COMMAND_HISTORY:
                self._command_durations = self._command_durations[-MAX_COMMAND_HISTORY:]
            log.debug("Command completed in %.2fs: %s", elapsed, cmd_summary)

            # Warn if command took more than half the timeout
            if elapsed > timeout / 2:
                log.warning(
                    "Command used %.0f%% of timeout (%.1fs / %ds): %s",
                    elapsed / timeout * 100, elapsed, timeout, cmd_summary,
                )

            output = stdout
            if stderr:
                output += f"\n[stderr]: {stderr}"
            if proc.returncode != 0:
                output += f"\n[exit code: {proc.returncode}]"

            # Truncate very long output
            if len(output) > DEFAULT_MAX_OUTPUT:
                keep_start = DEFAULT_MAX_OUTPUT // 2
                keep_end = DEFAULT_MAX_OUTPUT // 5
                output = output[:keep_start] + f"\n... ({len(output)} chars total, truncated) ...\n" + output[-keep_end:]

            return output.strip() or "(no output)"
        except OSError as e:
            elapsed = time.time() - start
            self._command_durations.append((command[:COMMAND_SUMMARY_LENGTH], elapsed))
            return f"Error running command: {e}"

    @staticmethod
    def _kill_process_tree(timeout_error: subprocess.TimeoutExpired) -> None:
        """Attempt graceful shutdown of timed-out process and its children.

        Uses /proc traversal on Linux to find child processes, sends SIGTERM
        first, then SIGKILL after a grace period if they're still alive.
        """
        import os
        import signal

        pid = getattr(timeout_error, "pid", None)
        if pid is None:
            log.debug("No PID on TimeoutExpired, OS will clean up")
            return

        # Collect child PIDs via /proc (Linux) before killing the parent
        child_pids: list[int] = []
        try:
            proc_dir = f"/proc/{pid}/task/{pid}/children"
            if os.path.exists(proc_dir):
                with open(proc_dir) as f:
                    child_pids = [int(p) for p in f.read().split() if p.strip()]
        except (OSError, ValueError):
            pass  # /proc not available (non-Linux) or already exited

        # Send SIGTERM to parent and children
        targets = [pid] + child_pids
        for target_pid in targets:
            try:
                os.kill(target_pid, signal.SIGTERM)
                log.debug("Sent SIGTERM to PID %d", target_pid)
            except ProcessLookupError:
                pass  # already exited
            except PermissionError:
                log.debug("No permission to kill PID %d", target_pid)

        # Brief grace period then SIGKILL any survivors
        time.sleep(0.5)
        for target_pid in targets:
            try:
                os.kill(target_pid, signal.SIGKILL)
                log.debug("Sent SIGKILL to PID %d", target_pid)
            except ProcessLookupError:
                pass  # exited after SIGTERM
            except PermissionError:
                pass

    def get_stats(self) -> dict[str, Any]:
        """Get command execution duration statistics."""
        if not self._command_durations:
            return {"total_commands": 0, "total_time_s": 0, "avg_time_s": 0}
        durations = [d for _, d in self._command_durations]
        return {
            "total_commands": len(durations),
            "total_time_s": round(sum(durations), 2),
            "avg_time_s": round(sum(durations) / len(durations), 2),
            "max_time_s": round(max(durations), 2),
            "recent": self._command_durations[-5:],
        }

    @staticmethod
    def _is_dangerous(command: str) -> bool:
        """Check if a command is potentially destructive."""
        dangerous_patterns = [
            "rm -rf /",
            "rm -rf ~",
            "rm -rf /*",
            "mkfs.",
            "dd if=",
            ":(){",  # fork bomb
            "> /dev/sd",
            "chmod 000",
            "sudo rm",
            "sudo mkfs",
            "sudo dd",
            "sudo chmod",
            "sudo chown /",
            "shred ",
            "wipefs",
            "> /dev/null 2>&1 &",  # silent background execution
            "nohup rm",
            "xargs rm",
            "find / -delete",
            "find / -exec rm",
            "truncate -s 0 /",
            "systemctl disable",
            "systemctl mask",
        ]
        cmd_lower = command.lower().strip()
        if any(pattern in cmd_lower for pattern in dangerous_patterns):
            return True

        # Use pre-compiled regex patterns for dangerous command detection
        return any(pat.search(cmd_lower) for pat in _DANGEROUS_REGEXES)

    @staticmethod
    def _is_interactive(command: str) -> bool:
        """Check if a command requires interactive input (would hang)."""
        # Single-word commands that are interactive
        interactive_exact = {
            "top", "htop", "btop", "ipython", "irb", "python", "python3",
            "node", "bpython", "mysql", "psql", "sqlite3", "mongo", "redis-cli",
        }

        # Prefixes that indicate interactive mode
        interactive_prefixes = [
            "git rebase -i",
            "git add -i",
            "git add --interactive",
            "vim ", "nano ", "emacs ", "vi ", "nvim ", "micro ",
            "less ", "more ",
            "python3 -i", "python -i",
            "ssh ", "telnet ", "ftp ", "sftp ",
            "docker exec -it", "docker run -it",
            "kubectl exec -it",
            "nslookup",
        ]

        cmd_lower = command.lower().strip()
        if cmd_lower in interactive_exact:
            return True
        return any(cmd_lower.startswith(p) for p in interactive_prefixes)
