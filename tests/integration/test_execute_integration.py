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


# === Timeout ==================================================================
#
# Conventions for this group:
# - Exit codes 124 (SIGTERM at the deadline) and 137 (SIGKILL after the kill
#   delay) are GNU coreutils `timeout` behavior. Busybox/Alpine can differ, so
#   a failure on another image is a finding about the README, not a test bug.
# - Timing assertions use wide windows: the lower bound proves the command
#   really ran for about the timeout, the upper bound only catches hangs.
#   Don't tighten them to exact seconds; slow runners would make them flaky.
# - Leftover processes are found by scanning /proc, because slim images may
#   ship neither `ps` nor `pgrep`.


_FIND_SLEEP_299 = (
    r"for p in /proc/[0-9]*; do tr '\0' ' ' < $p/cmdline 2>/dev/null; echo; done"
    " | grep '[s]leep 299'"
)
# The [s] trick keeps this scanning command from matching itself.
# grep exits 0 if it found a process, 1 if none is left.


def test_timeout_kills_a_cooperative_command(sandbox):
    start = time.monotonic()
    result = sandbox.execute("sleep 30", timeout=2)
    elapsed = time.monotonic() - start

    assert result.exit_code == 124
    assert "timed out after 2 seconds" in result.output
    assert 1.5 <= elapsed < 8, elapsed


def test_timeout_escalates_to_sigkill_when_sigterm_is_ignored(sandbox):
    """Ignored signals are inherited, so `sleep` ignores SIGTERM too."""
    start = time.monotonic()
    result = sandbox.execute("trap '' TERM; sleep 30", timeout=2)
    elapsed = time.monotonic() - start

    assert result.exit_code == 137
    assert 3.5 <= elapsed < 10, elapsed  # timeout + 2 s kill delay


def test_timeout_kills_child_processes(sandbox):
    sandbox.execute("sleep 299 & wait", timeout=2)

    leftovers = sandbox.execute(_FIND_SLEEP_299)

    assert leftovers.exit_code == 1, leftovers.output


def test_setsid_child_survives_the_timeout(sandbox):
    """Characterization of a documented limitation (README: "Processes that
    detach (setsid, double fork) can outlive timeout").

    If this test starts failing, the behavior changed. Decide on purpose
    whether the README or the code should change.
    """
    sandbox.execute("setsid sleep 299 & sleep 30", timeout=2)

    leftovers = sandbox.execute(_FIND_SLEEP_299)

    assert leftovers.exit_code == 0, "the detached process was killed after all"
