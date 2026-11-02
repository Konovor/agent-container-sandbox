"""Tests for the exception hierarchy and the reuse errors: no Docker daemon required."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent_container_sandbox.errors import (
    ContainerNotRunningError,
    ContainerReuseError,
    DockerImageNotFound,
    MissingPythonError,
    MissingTimeoutError,
    SandboxError,
    SpecMismatchError,
)


@pytest.mark.parametrize(
    ("error", "parent"),
    [
        (DockerImageNotFound, SandboxError),
        (MissingTimeoutError, SandboxError),
        (MissingPythonError, SandboxError),
        (ContainerReuseError, SandboxError),
        (SpecMismatchError, ContainerReuseError),
        (ContainerNotRunningError, ContainerReuseError),
    ],
)
def test_error_hierarchy(error, parent):
    """Every library error sits where the documented tree says it does."""
    assert issubclass(error, parent)


@pytest.mark.parametrize("reason", ["not_created_by_library", "no_fingerprint", "different_spec"])
def test_spec_mismatch_error_carries_reason_container_and_recovery(reason):
    """The error exposes what failed and names the way out in its message."""
    container = MagicMock()
    err = SpecMismatchError("box", container, reason)
    assert err.reason == reason
    assert err.name == "box"
    assert err.container is container
    assert "'box'" in str(err)
    assert "e.container" in str(err)


def test_spec_mismatch_error_messages_differ_per_reason():
    """Each reason gets its own explanation, not one generic message."""
    messages = {
        str(SpecMismatchError("box", MagicMock(), reason))
        for reason in ("not_created_by_library", "no_fingerprint", "different_spec")
    }
    assert len(messages) == 3


def test_container_not_running_error_carries_status_and_both_ways_out():
    container = MagicMock()
    err = ContainerNotRunningError("box", container, "exited")
    assert err.status == "exited"
    assert err.container is container
    assert "exited" in str(err)
    assert "e.container.start()" in str(err)
    assert "delete" in str(err)
    assert "max_lifetime_seconds" in str(err)


def test_reuse_errors_are_caught_by_the_base_class():
    """One `except ContainerReuseError` covers both failure kinds."""
    for err in (
        SpecMismatchError("box", MagicMock(), "different_spec"),
        ContainerNotRunningError("box", MagicMock(), "exited"),
    ):
        with pytest.raises(ContainerReuseError):
            raise err
