"""Self-improvement agent: iterates on ollama-anvil using LLM + web research.

This agent:
1. Reads community ideas and evaluates them
2. Searches the web for latest AI/LLM advances
3. Proposes improvements to the codebase
4. Tests changes before committing
5. Contributors: creates a GitHub PR for review
6. Maintainers: commits directly to main, may promote to stable
7. Respects stable branch policy: max 1 push per 2 days

Contribution modes:
- contributor (default): Forks + creates PRs via `gh` CLI. No direct push access.
- maintainer: Direct push to main + stable branch promotion. Only for repo owner
  and their AI agents.

This feature is OPT-IN: disabled by default. Users must explicitly enable it
to donate spare CPU/GPU resources for self-improvement.

Branch policy:
- main: experimental changes, concurrent contributors welcome
- stable: tested improvements only, max 1 push per 2 days (maintainer only)

Designed to handle concurrent usage: uses file-based locking and
pull-before-push to avoid conflicts.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anvil.agents.qa import QAAgent
from anvil.community.ideas import IdeaCollector
from anvil.llm.client import OllamaClient
from anvil.utils.fileio import atomic_write
from anvil.utils.logging import get_logger

log = get_logger("community.self_improve")

__all__ = [
    "ImprovementResult",
    "SelfImproveAgent",
    "STABLE_PUSH_COOLDOWN",
    "UPSTREAM_REPO",
    "SAFE_TEST_PATTERNS",
    "DANGEROUS_TEST_PATTERNS",
    "LOCK_FILE_NAME",
    "LOG_DESCRIPTION_LENGTH",
    "LOG_REVIEW_LENGTH",
    "LOG_STDERR_LENGTH",
    "TIMEOUT_TEST",
    "TIMEOUT_GH_AUTH",
    "TIMEOUT_GH_PR",
    "TIMEOUT_GIT",
]

# Stable branch: max 1 push per 2 days (172800 seconds)
STABLE_PUSH_COOLDOWN = 2 * 24 * 3600

# Upstream repo for PRs (community contributors fork from this)
UPSTREAM_REPO = "hariravichandran/ollama-anvil"

# Lock file for serializing concurrent improvement runs
LOCK_FILE_NAME = "self_improve.lock"

# Truncation limits for logging and display
LOG_DESCRIPTION_LENGTH = 200  # candidate description in LLM prompt
LOG_REVIEW_LENGTH = 300  # code review message
LOG_STDERR_LENGTH = 500  # test failure stderr

# Subprocess timeout values (seconds)
TIMEOUT_TEST = 120       # running test suite
TIMEOUT_GH_AUTH = 10     # GitHub CLI auth check
TIMEOUT_GH_PR = 30       # creating a PR via GitHub CLI
TIMEOUT_GIT = 60         # git operations (commit, checkout, etc.)

# Safe test command patterns (whitelist approach)
SAFE_TEST_PATTERNS = [
    "python -m pytest",
    "pytest",
    "python -m unittest",
    "cargo test",
    "npm test",
    "npm run test",
    "go test",
    "make test",
]

# Dangerous patterns in test commands — these should never execute
DANGEROUS_TEST_PATTERNS = [
    "rm ", "rm\t", "rmdir",
    "curl ", "wget ",
    "sudo ", "su ",
    "chmod ", "chown ",
    "mkfs", "dd if=",
    "> /dev/", "| sh", "| bash",
    "eval ", "exec ",
]


@dataclass(slots=True)
class ImprovementResult:
    """Result of an improvement attempt."""

    idea_id: str
    success: bool
    description: str
    files_changed: list[str]
    tests_passed: bool
    committed: bool
    commit_hash: str = ""
    pr_url: str = ""


class SelfImproveAgent:
    """Iterates on ollama-anvil, evaluating ideas and proposing improvements.

    Two modes of operation:

    **Contributor mode** (default):
    - Creates a feature branch locally
    - Commits changes to the feature branch
    - Uses `gh pr create` to submit a PR to the upstream repo
    - Does NOT have direct push access to main or stable
    - Requires `gh` CLI to be installed and authenticated

    **Maintainer mode** (opt-in via config):
    - Pushes directly to main branch
    - Can promote tested changes to stable branch (max 1 per 2 days)
    - Only for the repo owner (hariravichandran) and their AI agents

    The improvement loop:
    1. Pull latest from main
    2. Read community ideas + search for latest advances
    3. Evaluate which ideas are feasible and beneficial
    4. Implement the most promising improvement
    5. Run tests
    6. If tests pass: create PR (contributor) or push to main (maintainer)
    7. Maintainer only: maybe promote to stable
    """

    def __init__(
        self,
        client: OllamaClient,
        idea_collector: IdeaCollector,
        repo_dir: str = ".",
        maintainer: bool = False,
    ):
        self.client = client
        self.ideas = idea_collector
        self.repo_dir = Path(repo_dir)
        self.maintainer = maintainer
        self.qa = QAAgent(client=client, repo_dir=repo_dir)
        self.state_file = self.repo_dir / ".anvil_state" / "self_improve_state.json"
        self.state = self._load_state()

    @property
    def is_contributor(self) -> bool:
        """True if running in contributor mode (creates PRs, no direct push)."""
        return not self.maintainer

    def run_iteration(self) -> ImprovementResult | None:
        """Run one improvement iteration.

        Uses file-based locking to prevent concurrent runs from
        conflicting with each other.

        Returns ImprovementResult if a change was made, None otherwise.
        """
        # Acquire lock to prevent concurrent runs
        if not self._acquire_lock():
            log.warning("Another self-improve iteration is running. Skipping.")
            return None

        try:
            return self._run_iteration_locked()
        finally:
            self._release_lock()

    def _run_iteration_locked(self) -> ImprovementResult | None:
        """The actual iteration logic (called while holding the lock)."""
        # Verify gh CLI is available for contributors
        if self.is_contributor and not self._gh_available():
            log.error(
                "GitHub CLI (gh) is required for contributor mode. "
                "Install: https://cli.github.com/ then run: gh auth login"
            )
            return None

        # Step 1: Pull latest
        self._git_pull()

        # Step 2: Gather improvement candidates
        candidates = self._gather_candidates()
        if not candidates:
            log.info("No improvement candidates found")
            return None

        # Step 3: Evaluate and pick the best candidate
        best = self._evaluate_candidates(candidates)
        if not best:
            log.info("No viable candidates after evaluation")
            return None

        # Step 4: Propose implementation
        proposal = self._propose_implementation(best)
        if not proposal:
            log.info("Could not generate viable proposal for: %s", best.get("title"))
            return None

        # Step 5: For contributors, create a feature branch before applying changes
        branch_name = None
        if self.is_contributor:
            branch_name = self._create_feature_branch(best)

        # Step 6: Apply and test
        result = self._apply_and_test(best, proposal)

        # Step 7: Commit and submit
        if result.success and result.tests_passed:
            if self.is_contributor:
                result.committed, result.pr_url = self._submit_pr(result, branch_name)
                # Return to main after PR submission
                self._git_run("checkout", "main")
            else:
                result.committed = self._push_to_main(result)

        # Step 8: Maintainer only — maybe promote to stable
        if result.committed and self.maintainer:
            self._maybe_promote_to_stable()

        # Clean up feature branch on failure
        if not result.committed and branch_name:
            try:
                self._git_run("checkout", "main")
                self._git_run("branch", "-D", branch_name)
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
                log.debug("Branch cleanup failed (non-critical): %s", e)

        return result

    def _gather_candidates(self) -> list[dict[str, Any]]:
        """Gather improvement candidates from community ideas and web research."""
        candidates = []

        # Community ideas
        new_ideas = self.ideas.get_new_ideas()
        for idea in new_ideas:
            candidates.append({
                "source": "community",
                "idea_id": idea.id,
                "title": idea.title,
                "description": idea.description,
                "category": idea.category,
                "votes": idea.votes,
            })

        # Web research for latest advances
        web_candidates = self._search_latest_advances()
        candidates.extend(web_candidates)

        log.info("Gathered %d improvement candidates (%d community, %d research)",
                 len(candidates), len(new_ideas), len(web_candidates))
        return candidates

    @staticmethod
    def _search_latest_advances() -> list[dict[str, Any]]:
        """Search the web for latest LLM/AI advances relevant to the project."""
        try:
            from anvil.mcp.web_search import WebSearchMCP
            search = WebSearchMCP()

            queries = [
                f"ollama new features improvements {time.strftime('%Y')}",
                "local LLM optimization techniques",
                "MCP model context protocol new servers",
                "context window compression techniques LLM",
            ]

            candidates = []
            for query in queries:
                results = search.search(query, max_results=3)
                for r in results:
                    candidates.append({
                        "source": "research",
                        "title": r.get("title", ""),
                        "description": r.get("body", ""),
                        "url": r.get("href", ""),
                        "category": "improvement",
                        "votes": 0,
                    })
            return candidates
        except (ImportError, OSError, ValueError) as e:
            log.info("Web research unavailable: %s", e)
            return []

    def _evaluate_candidates(self, candidates: list[dict]) -> dict | None:
        """Use LLM to evaluate which candidate is most promising."""
        if not candidates:
            return None

        # Format candidates for LLM
        candidates_text = "\n".join(
            f"{i+1}. [{c['category']}] {c['title']} (votes: {c.get('votes', 0)})\n   {c['description'][:LOG_DESCRIPTION_LENGTH]}"
            for i, c in enumerate(candidates[:15])
        )

        result = self.client.generate(
            prompt=(
                f"Evaluate these improvement candidates for an open-source local AI framework (ollama-anvil).\n"
                f"Pick the ONE most impactful and feasible improvement.\n\n"
                f"Candidates:\n{candidates_text}\n\n"
                f"Return JSON: {{\"pick\": <number>, \"reason\": \"...\", \"feasibility\": \"high/medium/low\"}}"
            ),
            system="You evaluate software improvement proposals. Be practical — favor high-impact, low-risk changes.",
            json_mode=True,
            temperature=0.3,
        )

        try:
            data = json.loads(result.get("response", "{}"))
            pick_idx = data.get("pick", 1) - 1
            if 0 <= pick_idx < len(candidates):
                chosen = candidates[pick_idx]
                chosen["evaluation_reason"] = data.get("reason", "")
                return chosen
        except (json.JSONDecodeError, ValueError) as e:
            log.debug("Failed to parse LLM candidate evaluation: %s", e)

        return candidates[0] if candidates else None

    def _propose_implementation(self, candidate: dict) -> dict | None:
        """Ask LLM to propose a concrete implementation."""
        result = self.client.generate(
            prompt=(
                f"Propose a concrete implementation for this improvement to ollama-anvil:\n\n"
                f"Title: {candidate['title']}\n"
                f"Description: {candidate['description']}\n\n"
                f"Return JSON: {{\n"
                f"  \"files_to_modify\": [\"path/to/file.py\"],\n"
                f"  \"changes\": [{{\"file\": \"...\", \"description\": \"...\", \"old_code\": \"...\", \"new_code\": \"...\"}}],\n"
                f"  \"tests_to_run\": [\"pytest tests/test_*.py\"],\n"
                f"  \"risk\": \"low/medium/high\"\n"
                f"}}"
            ),
            system="You propose minimal, safe code changes. Prefer small improvements over large refactors.",
            json_mode=True,
            temperature=0.3,
        )

        try:
            data = json.loads(result.get("response", "{}"))
            # Validate required keys in the proposal
            required_keys = {"files_to_modify", "changes", "tests_to_run", "risk"}
            if not isinstance(data, dict) or not all(k in data for k in required_keys):
                log.warning(
                    "LLM proposal missing required fields (need %s, got %s)",
                    required_keys, set(data.keys()) if isinstance(data, dict) else type(data).__name__,
                )
                return None
            return data
        except json.JSONDecodeError:
            return None

    def _apply_and_test(self, candidate: dict, proposal: dict) -> ImprovementResult:
        """Apply proposed changes, run QA validation, and test.

        The QA pipeline:
        1. Apply the proposed code changes
        2. Run existing tests (regression check)
        3. QA agent generates new tests specific to the changes
        4. Run generated tests to verify new behavior
        5. QA agent reviews code for common issues
        6. Rollback everything if any step fails
        """
        files_changed = []
        idea_id = candidate.get("idea_id", "research")

        # Apply changes
        repo_resolved = self.repo_dir.resolve()
        for change in proposal.get("changes", []):
            file_path = self.repo_dir / change.get("file", "")
            # Prevent path traversal outside repo directory
            if not str(file_path.resolve()).startswith(str(repo_resolved)):
                log.warning("Path traversal attempt blocked: %s", change.get("file", ""))
                continue
            if not file_path.exists():
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
                old_code = change.get("old_code", "")
                new_code = change.get("new_code", "")
                if old_code and old_code in content:
                    content = content.replace(old_code, new_code, 1)
                    file_path.write_text(content, encoding="utf-8")
                    files_changed.append(str(file_path))
            except OSError as e:
                log.error("Failed to apply change to %s: %s", file_path, e)

        if not files_changed:
            return ImprovementResult(
                idea_id=idea_id,
                success=False,
                description=candidate.get("title", ""),
                files_changed=[],
                tests_passed=False,
                committed=False,
            )

        # Get diff for QA context
        try:
            diff = self._git_run("diff")
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
            log.debug("Could not get git diff for QA context: %s", e)
            diff = ""

        # Run LLM-proposed test commands if any
        proposed_tests = proposal.get("tests_to_run", [])
        if proposed_tests and not self._run_tests(proposed_tests):
            log.warning("LLM-proposed tests failed, rolling back")
            self._git_checkout_files(files_changed)
            return ImprovementResult(
                idea_id=idea_id,
                success=False,
                description=candidate.get("title", ""),
                files_changed=[],
                tests_passed=False,
                committed=False,
            )

        # QA validation: existing tests + generated tests + code review
        qa_result = self.qa.validate_changes(
            files_changed=files_changed,
            change_description=candidate.get("title", ""),
            diff=diff,
        )

        tests_passed = qa_result.passed

        if not tests_passed:
            log.warning("QA validation failed: %s", qa_result.summary)

        # Code review (non-blocking but logged)
        review = self.qa.review_code(files_changed, diff=diff)
        if review and "LGTM" not in review.upper():
            log.info("QA review concerns: %s", review[:LOG_REVIEW_LENGTH])
            # If review finds security issues or critical bugs, fail the change
            review_lower = review.lower()
            if any(w in review_lower for w in ["security", "injection", "hardcoded secret", "critical bug"]):
                log.warning("QA review flagged critical issue — rolling back")
                tests_passed = False

        # Rollback if QA failed
        if not tests_passed and files_changed:
            log.warning("QA failed, rolling back changes")
            self._git_checkout_files(files_changed)
            files_changed = []

        return ImprovementResult(
            idea_id=idea_id,
            success=bool(files_changed),
            description=candidate.get("title", ""),
            files_changed=files_changed,
            tests_passed=tests_passed,
            committed=False,
        )

    def _run_tests(self, test_commands: list[str]) -> bool:
        """Run test commands with sanitization. Returns True if all pass.

        Validates commands against a whitelist of safe test patterns
        and rejects commands containing dangerous patterns.
        """
        if not test_commands:
            # Default: run pytest if it exists
            test_commands = ["python -m pytest tests/ -x --tb=short"]

        executed_any = False
        for cmd in test_commands:
            # Sanitize: reject dangerous commands
            if not self._is_safe_test_command(cmd):
                log.warning("Blocked unsafe test command: %s", cmd)
                continue

            executed_any = True
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,
                    timeout=TIMEOUT_TEST, cwd=str(self.repo_dir),
                )
                if result.returncode != 0:
                    log.warning("Test failed: %s\n%s", cmd, result.stderr[:LOG_STDERR_LENGTH])
                    return False
            except subprocess.TimeoutExpired:
                log.warning("Test timed out: %s", cmd)
                return False

        if not executed_any:
            log.warning("No safe test commands were executed — treating as failure")
            return False
        return True

    @staticmethod
    def _is_safe_test_command(cmd: str) -> bool:
        """Check if a test command is safe to execute.

        Validates against a whitelist of known test runners and
        rejects commands containing dangerous patterns.
        """
        cmd_lower = cmd.lower().strip()

        # Check for dangerous patterns
        for pattern in DANGEROUS_TEST_PATTERNS:
            if pattern in cmd_lower:
                return False

        # Check if command starts with a known safe test runner
        for safe in SAFE_TEST_PATTERNS:
            if cmd_lower.startswith(safe):
                return True

        # Unknown test command — reject by default
        log.warning("Unknown test command pattern, rejecting: %s", cmd)
        return False

    # ─── Contributor mode: fork + PR ─────────────────────────────────────────

    @staticmethod
    def _gh_available() -> bool:
        """Check if GitHub CLI is installed and authenticated."""
        if not shutil.which("gh"):
            return False
        try:
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True, text=True, timeout=TIMEOUT_GH_AUTH,
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired) as e:
            log.debug("GitHub auth check failed: %s", e)
            return False

    def _create_feature_branch(self, candidate: dict) -> str:
        """Create a feature branch for the improvement."""
        # Generate branch name from candidate title
        title_slug = candidate.get("title", "improvement")[:40]
        title_slug = "".join(c if c.isalnum() or c == "-" else "-" for c in title_slug.lower())
        title_slug = title_slug.strip("-")
        branch_name = f"self-improve/{title_slug}-{int(time.time()) % 100000}"

        self._git_run("checkout", "-b", branch_name)
        log.info("Created feature branch: %s", branch_name)
        return branch_name

    def _submit_pr(self, result: ImprovementResult, branch_name: str | None) -> tuple[bool, str]:
        """Create a GitHub PR for the improvement (contributor mode).

        Returns (committed, pr_url) tuple.
        """
        if not branch_name:
            return False, ""

        try:
            # Stage and commit changes
            for f in result.files_changed:
                self._git_run("add", f)

            msg = (
                f"self-improve: {result.description}\n\n"
                f"Idea: {result.idea_id}\n"
                f"Files: {', '.join(result.files_changed)}\n\n"
                f"Generated by ollama-anvil self-improvement agent."
            )
            self._git_run("commit", "-m", msg)

            # Push the feature branch to the user's fork (origin)
            self._git_run("push", "-u", "origin", branch_name)

            # Create PR against upstream repo
            pr_result = subprocess.run(
                [
                    "gh", "pr", "create",
                    "--repo", UPSTREAM_REPO,
                    "--title", f"self-improve: {result.description[:60]}",
                    "--body", (
                        f"## Self-Improvement PR\n\n"
                        f"**Idea:** {result.idea_id}\n"
                        f"**Description:** {result.description}\n\n"
                        f"### Changes\n"
                        f"{chr(10).join('- ' + f for f in result.files_changed)}\n\n"
                        f"### Testing\n"
                        f"- All tests passed locally\n\n"
                        f"---\n"
                        f"Generated by ollama-anvil self-improvement agent."
                    ),
                    "--head", branch_name,
                ],
                capture_output=True, text=True, timeout=TIMEOUT_GH_PR,
                cwd=str(self.repo_dir),
            )

            if pr_result.returncode == 0:
                pr_url = pr_result.stdout.strip()
                log.info("Created PR: %s", pr_url)
                return True, pr_url

            log.error("Failed to create PR: %s", pr_result.stderr)
            return False, ""

        except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
            log.error("Failed to submit PR: %s", e)
            return False, ""

    # ─── Maintainer mode: direct push ────────────────────────────────────────

    def _push_to_main(self, result: ImprovementResult) -> bool:
        """Commit improvement directly to main branch (maintainer only)."""
        if not self.maintainer:
            log.error("Direct push requires maintainer mode")
            return False

        try:
            # Stage changed files
            for f in result.files_changed:
                self._git_run("add", f)

            # Commit
            msg = (
                f"auto-improve: {result.description}\n\n"
                f"Idea: {result.idea_id}\n"
                f"Files: {', '.join(result.files_changed)}"
            )
            self._git_run("commit", "-m", msg)

            # Pull-rebase then push (safe for concurrent usage)
            self._git_run("pull", "--rebase", "origin", "main")
            self._git_run("push", "origin", "main")

            log.info("Committed improvement to main: %s", result.description)
            return True
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
            log.error("Failed to commit: %s", e)
            return False

    def _maybe_promote_to_stable(self) -> bool:
        """Promote main to stable if cooldown has elapsed (max 1 per 2 days).

        Maintainer only — contributors cannot promote to stable.
        """
        if not self.maintainer:
            return False

        last_stable_push = self.state.get("last_stable_push", 0)
        elapsed = time.time() - last_stable_push

        if elapsed < STABLE_PUSH_COOLDOWN:
            remaining_hours = (STABLE_PUSH_COOLDOWN - elapsed) / 3600
            log.info("Stable push cooldown: %.1f hours remaining", remaining_hours)
            return False

        try:
            # Merge main into stable
            self._git_run("checkout", "stable")
            self._git_run("pull", "origin", "stable")
            self._git_run("merge", "main", "--no-edit")
            self._git_run("push", "origin", "stable")
            self._git_run("checkout", "main")

            self.state["last_stable_push"] = time.time()
            self._save_state()

            log.info("Promoted main to stable branch")
            return True
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
            log.error("Failed to promote to stable: %s", e)
            # Make sure we're back on main
            try:
                self._git_run("checkout", "main")
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as e2:
                log.debug("Checkout to main failed during error recovery: %s", e2)
            return False

    # ─── Git helpers ─────────────────────────────────────────────────────────

    def _git_pull(self) -> None:
        """Pull latest from origin."""
        try:
            self._git_run("pull", "--rebase", "origin", "main")
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
            log.warning("Git pull failed: %s", e)

    def _git_checkout_files(self, files: list[str]) -> None:
        """Restore files from git (rollback)."""
        for f in files:
            try:
                self._git_run("checkout", "--", f)
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
                log.warning("Failed to restore file %s: %s", f, e)

    def _git_run(self, *args: str) -> str:
        """Run a git command."""
        result = subprocess.run(
            ["git"] + list(args),
            capture_output=True, text=True, timeout=TIMEOUT_GIT,
            cwd=str(self.repo_dir),
        )
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr[:LOG_STDERR_LENGTH]}")
        return result.stdout.strip()

    # ─── Locking ─────────────────────────────────────────────────────────────

    def _acquire_lock(self) -> bool:
        """Acquire a file-based lock to prevent concurrent runs.

        Returns True if lock acquired, False if another process holds it.
        Uses exclusive file creation (O_CREAT|O_EXCL) for atomic lock
        acquisition. Stale locks (older than 1 hour) are automatically
        cleaned up.
        """
        lock_path = self.repo_dir / ".anvil_state" / LOCK_FILE_NAME
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Try atomic exclusive creation first
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"locked_at": time.time(), "pid": os.getpid()}, f)
            return True
        except FileExistsError:
            pass  # Lock held by another process; fall through to stale check
        except OSError:
            return False

        # Lock file exists — check for stale lock (older than 1 hour)
        try:
            lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
            lock_time = lock_data.get("locked_at", 0)
            if time.time() - lock_time > 3600:
                log.warning("Removing stale lock (age: %.0f minutes)", (time.time() - lock_time) / 60)
                lock_path.unlink(missing_ok=True)
                # Retry atomic acquisition
                return self._acquire_lock()
            return False
        except (json.JSONDecodeError, OSError):
            lock_path.unlink(missing_ok=True)
            return self._acquire_lock()

    def _release_lock(self) -> None:
        """Release the file-based lock."""
        lock_path = self.repo_dir / ".anvil_state" / LOCK_FILE_NAME
        lock_path.unlink(missing_ok=True)

    def _load_state(self) -> dict[str, Any]:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Failed to load self-improve state: %s", e)
        return {}

    def _save_state(self) -> None:
        atomic_write(self.state_file, json.dumps(self.state, indent=2))
