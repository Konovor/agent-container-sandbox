"""Integration tests for the BaseSandbox helpers (glob, ls, grep, ...) running through
ContainerSandbox.execute()."""

from __future__ import annotations


def _many_files(sandbox, count: int = 500) -> None:
    sandbox.execute(
        "mkdir -p /workspace/many && cd /workspace/many && "
        f"for i in $(seq 1 {count}); do echo needle > f$i.txt; done"
    )


def test_glob_survives_output_over_the_limit(make_sandbox):
    """Regression test: an appended truncation note made deepagents' glob parser
    reject the torn final record. The truncated flag alone must be enough."""
    sandbox = make_sandbox(max_output_size=1024)
    _many_files(sandbox)

    result = sandbox.glob("*", "/workspace/many")

    assert result.error is None, result.error
    assert result.truncated is True
    assert result.truncation_reason == "transport"
