"""Integration tests for ContainerSandbox.execute against a real container."""

from __future__ import annotations


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