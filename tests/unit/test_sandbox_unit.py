"""Unit tests for sandbox: no Docker daemon required."""

import io
import stat
import tarfile
import threading
from unittest.mock import MagicMock

import docker
import pytest

from agent_container_sandbox import ContainerSandbox
from agent_container_sandbox.sandbox import _KILL_DELAY, _SANDBOX_ERROR_PREFIX

# -- Fixtures: creating the fake Docker world -------------------------------


@pytest.fixture
def mock_container():
    """A fake container. exec_run must return a (exit_code, output) 2-tuple,
    because execute() unpacks it."""
    container = MagicMock()
    container.short_id = "abc123"
    container.exec_run.return_value = (0, b"")
    container.attrs = {"Config": {"WorkingDir": "/workspace"}}
    return container


@pytest.fixture
def sandbox(mock_container):
    return ContainerSandbox(mock_container)


def _tar_bytes(name: str, content: bytes) -> bytes:
    """Build a single-file tar archive, for faking get_archive's return."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        info = tarfile.TarInfo(name=name)
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    return stream.getvalue()


def _fake_exec_run(directories=()):
    """exec_run stub: `test -d <path>` succeeds only for paths in `directories`."""
    calls = []

    def fake(cmd, **kwargs):
        text = " ".join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
        calls.append(text)
        if "test -d" in text:
            is_dir = any(d in text for d in directories)
            return (0 if is_dir else 1, b"")
        return (0, b"")  # mkdir -p and anything else

    fake.calls = calls
    return fake


# === execute() ================================================================


def test_execute_rejects_empty_command(sandbox, mock_container):
    """An empty command is rejected before touching the container."""
    result = sandbox.execute("")

    assert result.exit_code == 1
    assert "non-empty string" in result.output
    mock_container.exec_run.assert_not_called()


def test_execute_rejects_non_string_command(sandbox, mock_container):
    """A non-string command is rejected the same way as an empty one."""
    result = sandbox.execute(123)  # type: ignore[arg-type]

    assert result.exit_code == 1
    assert "non-empty string" in result.output
    mock_container.exec_run.assert_not_called()


def test_execute_happy_path(sandbox, mock_container):
    """A successful command returns its output and exit code."""
    mock_container.exec_run.return_value = (0, b"hello\n")

    result = sandbox.execute("echo hello")

    assert result.exit_code == 0
    assert result.output == "hello\n"
    assert result.truncated is False


def test_execute_container_side_timeout_returns_real_result(sandbox, mock_container):
    """When the container's own `timeout` kills the process, exec_run returns
    promptly with a real result -- this must NOT trigger the Python-side
    fallback path at all."""
    mock_container.exec_run.return_value = (124, b"partial output before kill")

    result = sandbox.execute("sleep 999", timeout=30)

    assert result.exit_code == 124
    assert result.output.startswith("partial output before kill")
    assert "timed out after 30 seconds" in result.output
    # a real command result, not a sandbox-generated failure
    assert _SANDBOX_ERROR_PREFIX not in result.output
    assert "no response from docker" not in result.output.lower()


def test_execute_wraps_command_with_container_side_timeout(sandbox, mock_container):
    """The command actually run inside the container must be wrapped with
    `timeout <effective_timeout>`, not sent bare."""
    sandbox.execute("echo hi", timeout=30)

    cmd = mock_container.exec_run.call_args.kwargs["cmd"]
    assert cmd == ["timeout", "-k", str(_KILL_DELAY), "30", "sh", "-c", "echo hi"]


def test_execute_falls_back_when_container_never_responds(sandbox, mock_container, monkeypatch):
    """If even the container-side timeout fails to make exec_run return --
    Docker itself hanging -- the Python-side fallback fires, with output
    distinguishable from a real timed-out command's output.

    Regression test for the race where the timeout branch read
    result_holder's fields separately after the background thread might
    already be writing to it; now also covers the two-tier timeout design.
    """
    # Shrink the grace period so this test doesn't need to sleep for real seconds.
    monkeypatch.setattr("agent_container_sandbox.sandbox._TIMEOUT_GRACE_SECONDS", 0.05)

    release_worker = threading.Event()

    def hanging_exec_run(**kwargs):
        release_worker.wait(timeout=5)  # blocks well past effective_timeout + grace
        return (124, b"finished late, container-side timeout apparently also stuck")

    mock_container.exec_run.side_effect = hanging_exec_run

    try:
        result = sandbox.execute("sleep 999", timeout=0.05)
    finally:
        release_worker.set()  # let the background thread finish so it doesn't linger

    assert result.exit_code == 124
    assert result.output.startswith(_SANDBOX_ERROR_PREFIX)
    assert "no response from docker" in result.output.lower()


def test_execute_timeout_messages_are_distinguishable(sandbox, mock_container, monkeypatch):
    """The container-side-timeout path and the Docker-hung fallback path must
    never produce output that could be confused for one another."""
    monkeypatch.setattr("agent_container_sandbox.sandbox._TIMEOUT_GRACE_SECONDS", 0.05)

    mock_container.exec_run.return_value = (124, b"command was killed by container timeout")
    real_timeout_result = sandbox.execute("sleep 999", timeout=30)

    release_worker = threading.Event()

    def hanging_exec_run(**kwargs):
        release_worker.wait(timeout=5)
        return (124, b"should never surface")

    mock_container.exec_run.side_effect = hanging_exec_run
    try:
        fallback_result = sandbox.execute("sleep 999", timeout=0.05)
    finally:
        release_worker.set()

    assert real_timeout_result.output != fallback_result.output
    assert "no response from docker" not in real_timeout_result.output.lower()
    assert "no response from docker" in fallback_result.output.lower()


def test_execute_returns_str_error_when_exec_run_raises(sandbox, mock_container):
    """A failing exec_run must produce a plain-string error, not a tuple.

    Regression test: a trailing comma once turned the message into a
    one-element tuple.
    """
    mock_container.exec_run.side_effect = RuntimeError("boom")

    result = sandbox.execute("echo hi")

    assert result.exit_code == 1
    assert isinstance(result.output, str)
    assert result.output.startswith(_SANDBOX_ERROR_PREFIX)
    assert "RuntimeError: boom" in result.output
    assert result.truncated is False


@pytest.mark.parametrize("bad_timeout", [0, -1, -0.5])
def test_execute_rejects_non_positive_timeout(sandbox, mock_container, bad_timeout):
    """GNU timeout reads 0 as "no limit" and fails on negatives, so these
    values are rejected before anything reaches the container."""
    result = sandbox.execute("echo hi", timeout=bad_timeout)

    assert result.exit_code == 1
    assert result.output.startswith(_SANDBOX_ERROR_PREFIX)
    assert "positive" in result.output
    assert str(bad_timeout) in result.output
    assert result.truncated is False
    mock_container.exec_run.assert_not_called()


def test_execute_accepts_small_positive_timeout(sandbox, mock_container):
    """Guards against an over-eager check: fractional timeouts stay valid."""
    mock_container.exec_run.return_value = (0, b"ok")

    result = sandbox.execute("echo ok", timeout=0.05)

    assert result.exit_code == 0
    mock_container.exec_run.assert_called_once()


# === upload_files() ============================================================


def test_upload_root_level_file_builds_valid_mkdir(sandbox, mock_container):
    """Uploading /file.txt must run `mkdir -p /`, not `mkdir -p ` (no operand)."""
    mock_container.exec_run.side_effect = _fake_exec_run()  # target is not a directory

    results = sandbox.upload_files([("/file.txt", b"data")])

    assert results[0].error is None
    mkdir_cmd = mock_container.exec_run.call_args_list[0].kwargs["cmd"]
    assert mkdir_cmd == [
        "timeout",
        "-k",
        str(_KILL_DELAY),
        str(sandbox._default_timeout),
        "sh",
        "-c",
        "mkdir -p /",
    ]
    mock_container.put_archive.assert_called_once()


def test_upload_quotes_path_with_space(sandbox, mock_container):
    """A directory path containing a space must reach the shell as a single
    quoted argument, not be split into two mkdir operands."""
    mock_container.exec_run.side_effect = _fake_exec_run()  # target is not a directory

    results = sandbox.upload_files([("/workspace/my dir/x.txt", b"data")])

    assert results[0].error is None
    mkdir_cmd = mock_container.exec_run.call_args_list[0].kwargs["cmd"]
    assert mkdir_cmd == [
        "timeout",
        "-k",
        str(_KILL_DELAY),
        str(sandbox._default_timeout),
        "sh",
        "-c",
        "mkdir -p '/workspace/my dir'",
    ]
    mock_container.put_archive.assert_called_once()


def test_upload_aborts_when_mkdir_fails(sandbox, mock_container):
    """If mkdir reports a non-zero exit code, put_archive must never run,
    and the failure is surfaced as a per-file error."""
    mock_container.exec_run.return_value = (1, b"mkdir: permission denied")

    results = sandbox.upload_files([("/workspace/x.txt", b"data")])

    assert results[0].error is not None
    assert "mkdir" in results[0].error.lower()
    mock_container.put_archive.assert_not_called()


def test_upload_onto_existing_directory_reports_error_and_does_not_write(
    sandbox, mock_container
):
    mock_container.exec_run.side_effect = _fake_exec_run(
        directories=["/workspace/adir"]
    )

    [response] = sandbox.upload_files([("adir", b"data")])

    mock_container.put_archive.assert_not_called()
    assert response.error is not None
    assert response.error.startswith("Failed to upload")
    assert "directory" in response.error


def test_upload_onto_non_directory_checks_and_writes(sandbox, mock_container):
    fake = _fake_exec_run(directories=[])
    mock_container.exec_run.side_effect = fake

    [response] = sandbox.upload_files([("file.txt", b"data")])

    assert any("test -d" in c for c in fake.calls)  # check ran (fails without fix)
    mock_container.put_archive.assert_called_once()
    assert response.error is None


def test_upload_batch_directory_target_does_not_stop_other_files(
    sandbox, mock_container
):
    mock_container.exec_run.side_effect = _fake_exec_run(
        directories=["/workspace/adir"]
    )

    first, second = sandbox.upload_files([("adir", b"a"), ("ok.txt", b"b")])

    assert first.path == "/workspace/adir"
    assert first.error is not None and "directory" in first.error
    assert second.path == "/workspace/ok.txt"
    assert second.error is None
    mock_container.put_archive.assert_called_once()


# === download_files() ==========================================================


def test_download_missing_file_returns_file_not_found(sandbox, mock_container):
    """A NotFound from the container maps to the file_not_found error code."""
    mock_container.get_archive.side_effect = docker.errors.NotFound("no such file")

    results = sandbox.download_files(["/workspace/nope.txt"])

    assert len(results) == 1
    assert results[0].content is None
    assert results[0].error == "file_not_found"


def test_download_directory_returns_path_is_directory(sandbox, mock_container):
    """A path whose stat reports a directory maps to path_is_directory."""
    tar_bytes = _tar_bytes("workspace/some_dir", b"")
    dir_stat = {"mode": stat.S_IFDIR | 0o755}
    mock_container.get_archive.return_value = (iter([tar_bytes]), dir_stat)

    results = sandbox.download_files(["/workspace/some_dir"])

    assert results[0].content is None
    assert results[0].error == "path_is_directory"


def test_download_happy_path(sandbox, mock_container):
    """A normal file round-trips through get_archive correctly."""
    tar_bytes = _tar_bytes("workspace/test.txt", b"hello world")
    mock_container.get_archive.return_value = (iter([tar_bytes]), {"mode": 0o644})

    results = sandbox.download_files(["/workspace/test.txt"])

    assert len(results) == 1
    assert results[0].content == b"hello world"
    assert results[0].error is None
    assert results[0].path == "/workspace/test.txt"  # echoes the requested path
