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


def test_ls_returns_a_partial_listing_without_any_signal_when_output_is_clipped(make_sandbox):
    """Characterization of an upstream limitation, reported as
    https://github.com/langchain-ai/deepagents/issues/7480

    deepagents' ls() drops ExecuteResponse.truncated and LsResult has no truncation
    field, so a clipped listing looks complete. If this test starts failing,
    deepagents changed: update it and the matching README limitation.
    """
    sandbox = make_sandbox(max_output_size=1024)
    _many_files(sandbox)

    result = sandbox.ls("/workspace/many")

    assert result.error is None
    assert 0 < len(result.entries) < 500


def test_grep_returns_partial_matches_flagged_as_complete_when_output_is_clipped(make_sandbox):
    """Characterization of an upstream limitation, reported as
    https://github.com/langchain-ai/deepagents/issues/7480

    deepagents' grep() does not consult ExecuteResponse.truncated, so clipped
    matches come back with truncated=False. If this test starts failing,
    deepagents changed: update it and the matching README limitation.
    """
    sandbox = make_sandbox(max_output_size=1024)
    _many_files(sandbox)

    result = sandbox.grep("needle", "/workspace/many")

    assert result.error is None
    assert 0 < len(result.matches) < 500
    assert result.truncated is False
