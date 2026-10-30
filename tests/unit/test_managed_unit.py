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

from agent_container_sandbox import (
    ContainerSandbox,
    ContainerSpec,
    MissingPythonError,
    MissingTimeoutError,
)
from agent_container_sandbox.managed import ManagedContainerSandbox
from agent_container_sandbox.sandbox import _KILL_DELAY, _PYTHON_PROBE, _TIMEOUT_PROBE

MANAGED_LOGGER = "agent_container_sandbox.managed"

# -- Fixtures -----------------------------------------------------------------


@pytest.fixture
def mock_container():
    """A fake container. The first exec_run call is provision()'s mkdir; any
    later call is a verify probe (timeout, then python3) or a sandbox command."""
    container = MagicMock()
    container.short_id = "abc123"
    container.status = "running"
    container.exec_run.return_value = (0, b"")
    container.attrs = {"Config": {"WorkingDir": "/from-container"}}
    return container


@pytest.fixture
def mock_client(mock_container):
    """A fake Docker client: image check passes, containers.create hands out the
    fake container."""
    client = MagicMock()
    client.images.get.return_value = MagicMock()
    client.containers.create.return_value = mock_container
    return client


@pytest.fixture
def spec():
    return ContainerSpec(image="test:image", name="test-container")


@pytest.fixture
def managed(mock_client, spec):
    return ManagedContainerSandbox(mock_client, spec)


# -- Helpers ------------------------------------------------------------------


def _probes(mock_container):
    """The probe commands that ran inside the container, in order."""
    commands = [c.kwargs["cmd"][-1] for c in mock_container.exec_run.call_args_list]
    return [c for c in commands if c in (_TIMEOUT_PROBE, _PYTHON_PROBE)]


def _fail_probe(mock_container, probe, exit_code, output):
    """Make exec_run fail for one probe command and succeed for everything else.

    Keyed by command instead of by call position, so adding or reordering
    probes does not silently break the tests that use it.
    """

    def exec_run(**kwargs):
        if kwargs["cmd"][-1] == probe:
            return (exit_code, output)
        return (0, b"")

    mock_container.exec_run.side_effect = exec_run


# === Construction ==============================================================


def test_construction_makes_no_docker_calls(mock_client, spec):
    """Building the object is side-effect free: nothing touches Docker until
    __enter__."""
    ManagedContainerSandbox(mock_client, spec)

    mock_client.images.get.assert_not_called()
    mock_client.containers.create.assert_not_called()


def test_public_surface_is_empty(managed):
    """The convenience layer exposes no public attributes, so the raw
    Container (and the provisioner) cannot be reached or tampered with."""
    public = [name for name in dir(managed) if not name.startswith("_")]

    assert public == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"command_timeout": 0},
        {"command_timeout": -1},
        {"max_output_size": 0},
        {"max_output_size": -1},
    ],
)
def test_managed_rejects_bad_limits_before_touching_docker(mock_client, kwargs):
    with pytest.raises(ValueError):
        ManagedContainerSandbox(mock_client, ContainerSpec(image="test:image"), **kwargs)

    mock_client.images.get.assert_not_called()
    mock_client.containers.create.assert_not_called()


# === __enter__ =================================================================


def test_enter_returns_sandbox_over_provisioned_container(managed, mock_client):
    """Entering provisions exactly one container and yields a ContainerSandbox
    for it, not the raw container."""
    with managed as sandbox:
        assert isinstance(sandbox, ContainerSandbox)
        assert sandbox.id == "abc123"

    mock_client.containers.create.assert_called_once()


def test_enter_provisions_from_the_given_spec(mock_client):
    """The spec's name and the marker label reach containers.create."""
    spec = ContainerSpec(image="test:image", name="my-sandbox", labels={"team": "x"})

    with ManagedContainerSandbox(mock_client, spec):
        pass

    run_kwargs = mock_client.containers.create.call_args.kwargs
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
        verify_python=False,
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


# -- verify_timeout / verify_python --------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, [_TIMEOUT_PROBE, _PYTHON_PROBE]),  # both by default, timeout first
        ({"verify_timeout": False}, [_PYTHON_PROBE]),
        ({"verify_python": False}, [_TIMEOUT_PROBE]),
        ({"verify_timeout": False, "verify_python": False}, []),
    ],
)
def test_enter_runs_only_the_enabled_checks_in_order(
    mock_client, mock_container, spec, kwargs, expected
):
    with ManagedContainerSandbox(mock_client, spec, **kwargs):
        pass

    assert _probes(mock_container) == expected


def test_probes_run_after_provision_and_use_raw_sh(managed, mock_container):
    """The probes follow provision()'s mkdir and are not wrapped in `timeout`."""
    with managed:
        pass

    calls = [c.kwargs["cmd"] for c in mock_container.exec_run.call_args_list]
    assert calls == [
        ["sh", "-c", "mkdir -p /workspace"],
        ["sh", "-c", _TIMEOUT_PROBE],
        ["sh", "-c", _PYTHON_PROBE],
    ]


# -- Failures during __enter__ -------------------------------------------------


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
    """If containers.create fails there is no container, so the managed layer
    must not try to delete anything."""
    mock_client.containers.create.side_effect = docker.errors.APIError("boom")

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


@pytest.mark.parametrize(
    ("probe", "error", "output"),
    [
        pytest.param(_TIMEOUT_PROBE, MissingTimeoutError, b"timeout: not found", id="timeout"),
        pytest.param(_PYTHON_PROBE, MissingPythonError, b"python3: not found", id="python"),
    ],
)
def test_enter_deletes_container_when_a_check_fails(managed, mock_container, probe, error, output):
    """If a probe fails after the container exists, it is deleted before the
    error propagates, because __exit__ never runs."""
    _fail_probe(mock_container, probe, 127, output)

    with pytest.raises(error):
        managed.__enter__()

    mock_container.stop.assert_called_once()
    mock_container.remove.assert_called_once()


def test_enter_does_not_probe_python_when_the_timeout_check_fails(managed, mock_container):
    _fail_probe(mock_container, _TIMEOUT_PROBE, 127, b"timeout: not found")

    with pytest.raises(MissingTimeoutError):
        managed.__enter__()

    assert _probes(mock_container) == [_TIMEOUT_PROBE]


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
    _fail_probe(mock_container, _TIMEOUT_PROBE, 127, b"timeout: not found")

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

    assert mock_client.containers.create.call_count == 1
    mock_container.stop.assert_called_once()


def test_reentry_after_exit_raises(managed, mock_client):
    """Instances are single-use: a second block raises even after the first
    one has exited."""
    with managed:
        pass

    with pytest.raises(RuntimeError, match="single-use"):
        with managed:
            pass

    assert mock_client.containers.create.call_count == 1


def test_instance_is_burned_even_when_first_enter_fails(managed, mock_container):
    """A failed first __enter__ still counts as use, so the instance cannot
    be retried."""
    _fail_probe(mock_container, _TIMEOUT_PROBE, 127, b"timeout: not found")

    with pytest.raises(MissingTimeoutError):
        managed.__enter__()

    with pytest.raises(RuntimeError, match="single-use"):
        managed.__enter__()
