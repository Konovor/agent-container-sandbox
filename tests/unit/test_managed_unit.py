"""Unit tests for ManagedContainerSandbox: no Docker daemon required.

These run the real ContainerProvisioner and ContainerSandbox over a mocked
Docker client/container, so they check the managed layer's wiring and
lifecycle guarantees without duplicating the provisioner's or sandbox's own
unit tests.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import docker
import pytest

from agent_container_sandbox import ContainerSandbox, ContainerSpec
from agent_container_sandbox.errors import MissingTimeoutError
from agent_container_sandbox.managed import ManagedContainerSandbox
from agent_container_sandbox.sandbox import _KILL_DELAY, _TIMEOUT_PROBE

MANAGED_LOGGER = "agent_container_sandbox.managed"

# -- Fixtures -----------------------------------------------------------------


@pytest.fixture
def mock_container():
    """A fake container. The first exec_run call is provision()'s mkdir; any
    later call is the verify_timeout() probe or a sandbox command."""
    container = MagicMock()
    container.short_id = "abc123"
    container.status = "running"
    container.exec_run.return_value = (0, b"")
    container.attrs = {"Config": {"WorkingDir": "/from-container"}}
    return container


@pytest.fixture
def mock_client(mock_container):
    """A fake Docker client: image check passes, containers.run hands out the
    fake container."""
    client = MagicMock()
    client.images.get.return_value = MagicMock()
    client.containers.run.return_value = mock_container
    return client


@pytest.fixture
def spec():
    return ContainerSpec(image="test:image", name="test-container")


@pytest.fixture
def managed(mock_client, spec):
    return ManagedContainerSandbox(mock_client, spec)


# === Construction ==============================================================


def test_construction_makes_no_docker_calls(mock_client, spec):
    """Building the object is side-effect free: nothing touches Docker until
    __enter__."""
    ManagedContainerSandbox(mock_client, spec)

    mock_client.images.get.assert_not_called()
    mock_client.containers.run.assert_not_called()


def test_public_surface_is_empty(managed):
    """The convenience layer exposes no public attributes, so the raw
    Container (and the provisioner) cannot be reached or tampered with."""
    public = [name for name in dir(managed) if not name.startswith("_")]

    assert public == []


@pytest.mark.parametrize(
    "kwargs",
    [{"command_timeout": 0}, {"command_timeout": -1},
     {"max_output_size": 0}, {"max_output_size": -1}],
)
def test_managed_rejects_bad_limits_before_touching_docker(mock_client, kwargs):
    with pytest.raises(ValueError):
        ManagedContainerSandbox(mock_client, ContainerSpec(image="test:image"), **kwargs)

    mock_client.images.get.assert_not_called()
    mock_client.containers.run.assert_not_called()


# === __enter__ =================================================================


def test_enter_returns_sandbox_over_provisioned_container(managed, mock_client):
    """Entering provisions exactly one container and yields a ContainerSandbox
    for it, not the raw container."""
    with managed as sandbox:
        assert isinstance(sandbox, ContainerSandbox)
        assert sandbox.id == "abc123"

    mock_client.containers.run.assert_called_once()


def test_enter_provisions_from_the_given_spec(mock_client):
    """The spec's name and the marker label reach containers.run."""
    spec = ContainerSpec(image="test:image", name="my-sandbox", labels={"team": "x"})

    with ManagedContainerSandbox(mock_client, spec):
        pass

    run_kwargs = mock_client.containers.run.call_args.kwargs
    assert run_kwargs["name"] == "my-sandbox"
    assert run_kwargs["labels"]["team"] == "x"
    assert run_kwargs["labels"]["created-by"] == "agent-container-sandbox"


def test_sandbox_discovers_container_working_dir_by_default(managed, mock_container):
    """With no working_dir override, commands run in the directory the
    container was configured with."""
    with managed as sandbox:
        sandbox.execute("pwd")

    assert mock_container.exec_run.call_args.kwargs["workdir"] == "/from-container"


def test_sandbox_settings_are_passed_through(mock_client, mock_container, spec):
    """working_dir, command_timeout and max_output_size reach the sandbox."""
    mock_container.exec_run.side_effect = [(0, b""), (0, b"x" * 50)]
    managed = ManagedContainerSandbox(
        mock_client,
        spec,
        working_dir="/custom",
        command_timeout=42,
        max_output_size=10,
        verify_timeout=False,
    )

    with managed as sandbox:
        result = sandbox.execute("echo hi")

    call_kwargs = mock_container.exec_run.call_args.kwargs
    assert call_kwargs["cmd"] == [
        "timeout",
        "-k",
        str(_KILL_DELAY),
        "42",
        "sh",
        "-c",
        "echo hi",
    ]
    assert call_kwargs["workdir"] == "/custom"
    assert result.truncated is True


# -- verify_timeout ------------------------------------------------------------


def test_enter_verifies_timeout_by_default(managed, mock_container):
    """By default the timeout probe runs once after provisioning."""
    with managed:
        pass

    assert mock_container.exec_run.call_count == 2  # mkdir, then the probe
    probe_cmd = mock_container.exec_run.call_args.kwargs["cmd"]
    assert probe_cmd == ["sh", "-c", _TIMEOUT_PROBE]


def test_enter_skips_verification_when_disabled(mock_client, mock_container, spec):
    """verify_timeout=False means only provision()'s own mkdir runs."""
    managed = ManagedContainerSandbox(mock_client, spec, verify_timeout=False)

    with managed:
        pass

    assert mock_container.exec_run.call_count == 1  # mkdir only


# -- Failures during __enter__ -------------------------------------------------


def test_enter_deletes_container_when_verification_fails(managed, mock_container):
    """If the probe fails after the container exists, it is deleted before
    the error propagates, because __exit__ never runs."""
    mock_container.exec_run.side_effect = [
        (0, b""),  # provision's mkdir
        (127, b"timeout: not found"),  # verify_timeout probe
    ]

    with pytest.raises(MissingTimeoutError):
        managed.__enter__()

    mock_container.stop.assert_called_once()
    mock_container.remove.assert_called_once()


def test_enter_deletes_container_when_sandbox_construction_fails(
    managed, mock_container, monkeypatch
):
    """Any failure after provisioning triggers cleanup, and the original
    exception propagates unchanged."""
    monkeypatch.setattr(
        "agent_container_sandbox.managed.ContainerSandbox",
        MagicMock(side_effect=RuntimeError("cannot build sandbox")),
    )

    with pytest.raises(RuntimeError, match="cannot build sandbox"):
        managed.__enter__()

    mock_container.stop.assert_called_once()
    mock_container.remove.assert_called_once()


def test_enter_cleans_up_on_keyboard_interrupt(managed, mock_container, monkeypatch):
    """Cleanup covers BaseException, not just Exception."""
    monkeypatch.setattr(
        "agent_container_sandbox.managed.ContainerSandbox",
        MagicMock(side_effect=KeyboardInterrupt),
    )

    with pytest.raises(KeyboardInterrupt):
        managed.__enter__()

    mock_container.stop.assert_called_once()
    mock_container.remove.assert_called_once()


def test_enter_does_not_delete_when_provisioning_itself_fails(managed, mock_client, mock_container):
    """If containers.run fails there is no container, so the managed layer
    must not try to delete anything."""
    mock_client.containers.run.side_effect = docker.errors.APIError("boom")

    with pytest.raises(docker.errors.APIError):
        managed.__enter__()

    mock_container.stop.assert_not_called()
    mock_container.remove.assert_not_called()


def test_enter_does_not_double_delete_when_provision_cleans_up_itself(managed, mock_container):
    """provision() already deletes the container when its mkdir fails; the
    managed layer must not delete it a second time."""
    mock_container.exec_run.return_value = (1, b"mkdir: permission denied")

    with pytest.raises(RuntimeError, match="mkdir"):
        managed.__enter__()

    mock_container.stop.assert_called_once()
    mock_container.remove.assert_called_once()


# === __exit__ ==================================================================


def test_exit_deletes_container_on_normal_exit(managed, mock_container):
    """Leaving the block deletes the container: stop, then remove."""
    with managed:
        mock_container.stop.assert_not_called()  # still alive inside the block

    mock_container.stop.assert_called_once()
    mock_container.remove.assert_called_once()


def test_exit_deletes_container_when_body_raises(managed, mock_container):
    """The container is deleted even if the body raised, and the body's
    exception propagates."""
    with pytest.raises(ValueError, match="body failed"):
        with managed:
            raise ValueError("body failed")

    mock_container.stop.assert_called_once()
    mock_container.remove.assert_called_once()


def test_exit_never_suppresses_exceptions(managed):
    """__exit__ returns a falsy value so exceptions are never swallowed."""
    managed.__enter__()

    result = managed.__exit__(ValueError, ValueError("x"), None)

    assert not result


def test_body_exception_wins_when_deletion_also_fails(managed, mock_container, caplog):
    """A failing delete must not mask the exception from the body."""
    mock_container.stop.side_effect = RuntimeError("daemon unreachable")

    with caplog.at_level(logging.WARNING, logger=MANAGED_LOGGER):
        with pytest.raises(ValueError, match="body failed"):
            with managed:
                raise ValueError("body failed")

    assert any("abc123" in record.getMessage() for record in caplog.records)


def test_deletion_failure_after_successful_body_is_logged_not_raised(
    managed, mock_container, caplog
):
    """If the body succeeded and only deletion fails, the block still
    finishes normally and the failure is logged as a warning."""
    mock_container.stop.side_effect = RuntimeError("daemon unreachable")

    with caplog.at_level(logging.WARNING, logger=MANAGED_LOGGER):
        with managed:
            pass  # must not raise on exit

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "abc123" in warnings[0].getMessage()


def test_container_already_gone_on_exit_is_not_an_error(managed, mock_container, caplog):
    """A NotFound during delete is the provisioner's tolerated end state, so
    nothing is logged as a failure."""
    mock_container.stop.side_effect = docker.errors.NotFound("gone")

    with caplog.at_level(logging.WARNING, logger=MANAGED_LOGGER):
        with managed:
            pass

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_cleanup_is_idempotent(managed, mock_container):
    """Cleaning up twice deletes the container only once."""
    with managed:
        pass

    managed._cleanup()

    mock_container.stop.assert_called_once()
    mock_container.remove.assert_called_once()


# === Client ownership ==========================================================


def test_client_is_never_closed_on_normal_exit(managed, mock_client):
    with managed:
        pass

    mock_client.close.assert_not_called()


def test_client_is_never_closed_when_body_raises(managed, mock_client):
    with pytest.raises(ValueError):
        with managed:
            raise ValueError("body failed")

    mock_client.close.assert_not_called()


def test_client_is_never_closed_when_enter_fails(managed, mock_client, mock_container):
    mock_container.exec_run.side_effect = [(0, b""), (127, b"timeout: not found")]

    with pytest.raises(MissingTimeoutError):
        managed.__enter__()

    mock_client.close.assert_not_called()


# === Single use ================================================================


def test_nested_reentry_raises_without_touching_the_running_container(
    managed, mock_client, mock_container
):
    """Entering an instance that is already active raises, provisions
    nothing new, and leaves the active container alone."""
    with managed:
        with pytest.raises(RuntimeError, match="single-use"):
            managed.__enter__()

        mock_container.stop.assert_not_called()

    assert mock_client.containers.run.call_count == 1
    mock_container.stop.assert_called_once()


def test_reentry_after_exit_raises(managed, mock_client):
    """Instances are single-use: a second block raises even after the first
    one has exited."""
    with managed:
        pass

    with pytest.raises(RuntimeError, match="single-use"):
        with managed:
            pass

    assert mock_client.containers.run.call_count == 1


def test_instance_is_burned_even_when_first_enter_fails(managed, mock_container):
    """A failed first __enter__ still counts as use, so the instance cannot
    be retried."""
    mock_container.exec_run.side_effect = [(0, b""), (127, b"timeout: not found")]

    with pytest.raises(MissingTimeoutError):
        managed.__enter__()

    with pytest.raises(RuntimeError, match="single-use"):
        managed.__enter__()
