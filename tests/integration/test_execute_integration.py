"""Integration tests for ContainerSandbox.execute against a real container."""

from __future__ import annotations

import shlex
import time

import pytest


def test_echo_returns_output_and_exit_code_zero(sandbox):
    result = sandbox.execute("echo hello")

    assert result.exit_code == 0
    assert result.output == "hello\n"
    assert result.truncated is False


def test_execute_on_stopped_container_returns_error_response(container, sandbox):
    container.stop(timeout=2)

    result = sandbox.execute("echo hello")  # must not raise

    assert result.exit_code == 1
    assert isinstance(result.output, str), type(result.output)
    assert result.output.startswith("Sandbox error:")
    assert result.truncated is False


def test_execute_on_removed_container_returns_error_response(
    container, sandbox, provisioner
):
    provisioner.delete(container)

    result = sandbox.execute("echo hello")  # must not raise

    assert result.exit_code == 1
    assert isinstance(result.output, str), type(result.output)
    assert result.output.startswith("Sandbox error:")
    assert result.truncated is False


def test_nonzero_exit_code_is_propagated(sandbox):
    result = sandbox.execute("exit 3")

    assert result.exit_code == 3
    assert result.output == ""


def test_stdout_and_stderr_are_merged(sandbox):
    result = sandbox.execute("echo out; echo err >&2")

    assert result.exit_code == 0
    assert "out" in result.output
    assert "err" in result.output


def test_commands_run_in_the_working_directory(sandbox):
    assert sandbox.execute("pwd").output.strip() == "/workspace"


def test_relative_paths_resolve_in_the_working_directory(sandbox):
    sandbox.execute("echo data > rel.txt")

    result = sandbox.execute("cat /workspace/rel.txt")

    assert result.output == "data\n"


def test_commands_reading_stdin_do_not_hang(sandbox):
    """Agents often run interactive commands by accident; none may block."""
    start = time.monotonic()
    result = sandbox.execute("cat")
    elapsed = time.monotonic() - start

    assert elapsed < 5
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "text",
    ["it's", 'say "hi"', "$HOME `id` $(id)", "line1\nline2", "héllo wörld ✓", "back\\slash"],
)
def test_command_text_passes_through_sh_c_intact(sandbox, text):
    result = sandbox.execute(f"printf %s {shlex.quote(text)}")

    assert result.output == text
