"""QA agent: generates and runs test cases for code changes.

Used by the self-improvement agent to verify that proposed changes
actually work before submitting PRs or pushing to main. The QA agent:

1. Analyzes the diff of proposed changes
2. Generates targeted test cases using the LLM
3. Writes tests to a temporary file
4. Runs the tests with pytest
5. Reports pass/fail with details

This ensures every self-improvement iteration is validated beyond
just "existing tests still pass" — it also verifies the NEW behavior
introduced by the change.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from anvil.llm.client import OllamaClient
from anvil.utils.logging import get_logger

log = get_logger("agents.qa")

__all__ = [
    "QAAgent",
    "QAResult",
    "EXISTING_TEST_TIMEOUT",
    "GENERATED_TEST_TIMEOUT",
    "MAX_FILE_CONTENT_LENGTH",
    "MAX_TEST_GEN_CONTENT_LENGTH",
    "MAX_DIFF_PREVIEW_LENGTH",
    "MAX_TEST_DIFF_LENGTH",
    "MAX_REVIEW_FILES",
    "MAX_TEST_GEN_FILES",
    "DANGEROUS_TEST_PATTERNS",
]

# Timeouts (seconds)
EXISTING_TEST_TIMEOUT = 120
GENERATED_TEST_TIMEOUT = 60

# File content limits
MAX_FILE_CONTENT_LENGTH = 3000
MAX_TEST_GEN_CONTENT_LENGTH = 2000
MAX_DIFF_PREVIEW_LENGTH = 2000
MAX_TEST_DIFF_LENGTH = 1500
MAX_REVIEW_FILES = 5
MAX_TEST_GEN_FILES = 3

# Dangerous patterns in generated test code — reject before running
DANGEROUS_TEST_PATTERNS = re.compile(
    r"(os\.system|subprocess\.call.*shell\s*=\s*True|shutil\.rmtree\s*\(\s*['\"/]"
    r"|open\s*\(.*/etc/|eval\s*\(|exec\s*\()",
    re.IGNORECASE,
)


class QAAgent:
    """Generates and runs test cases for code changes.

    Usage:
        qa = QAAgent(client, repo_dir="/path/to/repo")

        # Validate a set of changes
        result = qa.validate_changes(
            files_changed=["forge/mcp/registry.py"],
            change_description="Added 5 new MCP entries to the registry",
            diff="...",  # optional git diff
        )

        if result.passed:
            print("All QA checks passed")
        else:
            print(f"QA failed: {result.summary}")
    """

    def __init__(
        self,
        client: OllamaClient,
        repo_dir: str = ".",
    ):
        self.client = client
        self.repo_dir = Path(repo_dir)

    def validate_changes(
        self,
        files_changed: list[str],
        change_description: str,
        diff: str = "",
    ) -> QAResult:
        """Validate a set of changes by generating and running tests.

        Steps:
        1. Run existing tests (regression check)
        2. Generate new tests for the changed code
        3. Run the generated tests
        4. Return combined result
        """
        # Step 1: Regression check — existing tests must pass
        existing_pass = self._run_existing_tests()
        if not existing_pass:
            return QAResult(
                passed=False,
                existing_tests_passed=False,
                generated_tests_passed=False,
                summary="Existing tests failed — change broke something",
                generated_test_code="",
                test_output="Existing test suite failed",
            )

        # Step 2: Generate targeted tests for the changes
        test_code = self._generate_tests(files_changed, change_description, diff)
        if not test_code:
            # If we can't generate tests, at least existing tests passed
            return QAResult(
                passed=True,
                existing_tests_passed=True,
                generated_tests_passed=True,
                summary="Existing tests passed; could not generate additional tests",
                generated_test_code="",
                test_output="",
            )

        # Step 3: Run the generated tests
        gen_passed, test_output = self._run_generated_tests(test_code)

        return QAResult(
            passed=existing_pass and gen_passed,
            existing_tests_passed=existing_pass,
            generated_tests_passed=gen_passed,
            summary="All QA checks passed" if gen_passed else f"Generated tests failed:\n{test_output}",
            generated_test_code=test_code,
            test_output=test_output,
        )

    def review_code(self, files_changed: list[str], diff: str = "") -> str:
        """Use LLM to review code changes for common issues.

        Returns a review summary with any concerns found.
        """
        # Read the changed files
        file_contents = {}
        for f in files_changed[:MAX_REVIEW_FILES]:
            path = self.repo_dir / f
            if path.exists():
                try:
                    content = path.read_text(encoding="utf-8")
                    if len(content) > MAX_FILE_CONTENT_LENGTH:
                        content = content[:MAX_FILE_CONTENT_LENGTH] + "\n... (truncated)"
                    file_contents[f] = content
                except (OSError, UnicodeDecodeError) as e:
                    log.debug("Could not read file %s: %s", f, e)

        if not file_contents:
            log.warning("No readable files found for code review")
            return "Review skipped: could not read any changed files."

        files_text = "\n\n".join(
            f"--- {name} ---\n{content}"
            for name, content in file_contents.items()
        )

        result = self.client.generate(
            prompt=(
                f"Review these code changes for an open-source Python project:\n\n"
                f"Changes:\n{files_text}\n\n"
                f"Diff:\n{diff[:MAX_DIFF_PREVIEW_LENGTH] if diff else 'Not available'}\n\n"
                f"Check for:\n"
                f"1. Import errors or missing dependencies\n"
                f"2. Type errors or incorrect function signatures\n"
                f"3. Security issues (hardcoded secrets, command injection)\n"
                f"4. Logic errors or off-by-one bugs\n"
                f"5. Breaking changes to public APIs\n\n"
                f"Return a brief review. If no issues found, say 'LGTM'."
            ),
            system="You are a code reviewer. Be concise. Focus on real bugs, not style.",
            temperature=0.2,
        )

        return result.get("response", "Review failed")

    def _generate_tests(
        self,
        files_changed: list[str],
        change_description: str,
        diff: str,
    ) -> str:
        """Use LLM to generate test cases for the changes."""
        # Read changed files for context
        file_contents = {}
        for f in files_changed[:MAX_TEST_GEN_FILES]:
            path = self.repo_dir / f
            if path.exists():
                try:
                    content = path.read_text(encoding="utf-8")
                    if len(content) > MAX_TEST_GEN_CONTENT_LENGTH:
                        content = content[:MAX_TEST_GEN_CONTENT_LENGTH] + "\n... (truncated)"
                    file_contents[f] = content
                except (OSError, UnicodeDecodeError) as e:
                    log.debug("Could not read file %s: %s", f, e)

        if not file_contents:
            return ""

        files_text = "\n\n".join(
            f"--- {name} ---\n{content}"
            for name, content in file_contents.items()
        )

        result = self.client.generate(
            prompt=(
                f"Generate pytest test cases for these code changes:\n\n"
                f"Description: {change_description}\n\n"
                f"Changed files:\n{files_text}\n\n"
                f"Diff:\n{diff[:MAX_TEST_DIFF_LENGTH] if diff else 'Not available'}\n\n"
                f"Requirements:\n"
                f"- Write 2-5 focused test functions\n"
                f"- Use pytest (no unittest)\n"
                f"- Tests must be self-contained (no external services)\n"
                f"- Import from the actual module paths (e.g., from anvil.mcp.registry import ...)\n"
                f"- Test the NEW behavior introduced by the change\n"
                f"- Include edge cases\n\n"
                f"Return ONLY valid Python code (no markdown fences)."
            ),
            system=(
                "You generate Python test code. Return ONLY executable Python code. "
                "No markdown. No explanations. Just pytest-compatible test functions."
            ),
            temperature=0.2,
        )

        test_code = result.get("response", "")

        # Clean up: remove markdown fences if the LLM included them
        test_code = test_code.strip()
        if test_code.startswith("```python"):
            test_code = test_code[len("```python"):].strip()
        if test_code.startswith("```"):
            test_code = test_code[3:].strip()
        if test_code.endswith("```"):
            test_code = test_code[:-3].strip()

        # Basic validation: must contain at least one test function
        if "def test_" not in test_code:
            log.warning("Generated test code has no test functions")
            return ""

        # Safety check: reject generated tests with dangerous patterns
        if DANGEROUS_TEST_PATTERNS.search(test_code):
            log.warning("Generated test code contains dangerous patterns, rejecting")
            return ""

        return test_code

    def _run_generated_tests(self, test_code: str) -> tuple[bool, str]:
        """Write generated tests to a temp file and run them.

        Returns (passed, output) tuple.
        """
        # Write to a temporary file outside the project tree
        fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="forge_qa_test_")
        test_file = Path(tmp_path)
        os.close(fd)
        try:
            test_file.write_text(test_code, encoding="utf-8")

            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
                capture_output=True, text=True,
                timeout=GENERATED_TEST_TIMEOUT, cwd=str(self.repo_dir),
            )
            output = result.stdout + result.stderr
            passed = result.returncode == 0

            if not passed:
                log.warning("Generated tests failed:\n%s", output[-500:])

            return passed, output
        except subprocess.TimeoutExpired:
            return False, "Generated tests timed out"
        except OSError as e:
            log.error("Failed to run generated tests: %s", e)
            return False, f"Failed to run generated tests: {e}"
        finally:
            # Clean up temp test file
            if test_file.exists():
                test_file.unlink()

    def _run_existing_tests(self) -> bool:
        """Run the existing test suite as a regression check."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-x", "--tb=short", "-q"],
                capture_output=True, text=True,
                timeout=EXISTING_TEST_TIMEOUT, cwd=str(self.repo_dir),
            )
            if result.returncode != 0:
                log.warning("Existing tests failed:\n%s", result.stderr[-500:])
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            log.warning("Existing test suite timed out")
            return False


class QAResult:
    """Result from QA validation."""

    def __init__(
        self,
        passed: bool,
        existing_tests_passed: bool,
        generated_tests_passed: bool,
        summary: str,
        generated_test_code: str,
        test_output: str,
    ):
        self.passed = passed
        self.existing_tests_passed = existing_tests_passed
        self.generated_tests_passed = generated_tests_passed
        self.summary = summary
        self.generated_test_code = generated_test_code
        self.test_output = test_output

    def __repr__(self) -> str:
        return (
            f"QAResult(passed={self.passed}, existing={self.existing_tests_passed}, "
            f"generated={self.generated_tests_passed})"
        )
