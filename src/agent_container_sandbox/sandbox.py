from __future__ import annotations

import codecs
import io
import logging
import shlex
import tarfile
import threading
import time
from typing import Any

import docker
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox
from docker.models.containers import Container

from .errors import MissingTimeoutError

logger = logging.getLogger(__name__)

_DEFAULT_COMMAND_TIMEOUT = 150  # seconds
_DEFAULT_MAX_OUTPUT_SIZE = 64 * 1024  # 64 KiB
_KILL_DELAY = 2  # seconds to wait after sending SIGTERM before sending SIGKILL
_GO_MODE_DIR = 1 << 31  # os.ModeDir: Docker's path stat uses Go's FileMode, not POSIX bits
_TIMEOUT_GRACE_SECONDS = 5  # headroom for the container-side kill to land before we give up waiting
_TIMEOUT_PROBE = "timeout -k 1 1 true"  # must match the form execute() uses
_POSIX_TIMEOUT_EXIT_CODE = 124
_SANDBOX_ERROR_PREFIX = "Sandbox error:"


def _decode_and_truncate(raw: bytes, limit: int) -> tuple[str, bool]:
    """Decode command output, cutting it at ``limit`` raw bytes."""
    if len(raw) <= limit:
        return raw.decode("utf-8", errors="replace"), False
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    text = decoder.decode(raw[:limit], final=False)  # holds back a split trailing character
    return f"{text}\n\n... Output truncated at {limit} bytes.", True


def _resolve_path(path: str, working_dir: str) -> str:
    return path if path.startswith("/") else f"{working_dir}/{path}"


def _build_single_file_tar(file_path: str, content: bytes) -> io.BytesIO:
    """Build a single-file tar archive ready for put_archive."""
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        info = tarfile.TarInfo(name=file_path.lstrip("/"))
        info.size = len(content)
        info.mtime = int(time.time())  # TarInfo defaults to 0, which dates files 1970
        tar.addfile(info, io.BytesIO(content))
    tar_stream.seek(0)
    return tar_stream


def _is_directory(path_stat: dict | None) -> bool:
    return path_stat is not None and bool(path_stat.get("mode", 0) & _GO_MODE_DIR)


def _extract_single_file(tar_data: io.BytesIO) -> tuple[bytes | None, str | None]:
    """Given a downloaded tar archive, return (content, error)."""
    with tarfile.open(fileobj=tar_data, mode="r") as tar:
        members = tar.getmembers()
        f = tar.extractfile(members[0]) if members else None
        if f is None:
            return None, "file_not_found"
        return f.read(), None


def _validate_limits(*, command_timeout: int, max_output_size: int) -> None:
    """Reject settings that have no sensible meaning. Raises ValueError."""
    if command_timeout <= 0:
        raise ValueError(f"command_timeout must be positive, got {command_timeout}.")
    if max_output_size <= 0:
        raise ValueError(f"max_output_size must be positive, got {max_output_size}.")


class ContainerSandbox(BaseSandbox):
    """Deep Agents sandbox backed by an existing, running Docker container.

    A thin adapter: it runs commands and transfers files inside a container
    you already have. It never creates, stops, or removes containers and never
    closes the Docker client. Use ``ContainerProvisioner`` for lifecycle.

    The container image must provide ``sh`` and a ``timeout`` binary.
    Commands exceeding their timeout are killed inside the container and
    return exit code 124.

    Args:
        container: A running container. The caller owns its lifecycle.
        working_dir: Directory used for commands and relative paths. If None,
            the container's configured working directory is used (``/`` if the
            image sets none). If given, it must already exist.
        command_timeout: Default per-command timeout in seconds.
        max_output_size: Output beyond this size is truncated.
    """

    def __init__(
        self,
        container: Container,
        *,
        working_dir: str | None = None,
        command_timeout: int = _DEFAULT_COMMAND_TIMEOUT,
        max_output_size: int = _DEFAULT_MAX_OUTPUT_SIZE,
    ) -> None:
        _validate_limits(command_timeout=command_timeout, max_output_size=max_output_size)
        self._container = container
        self._working_dir = working_dir or self._discover_working_dir()
        self._default_timeout = command_timeout
        self._max_output_bytes = max_output_size

    @property
    def id(self) -> str:
        return self._container.short_id

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output=f"{_SANDBOX_ERROR_PREFIX} Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )
        if timeout is not None and timeout <= 0:
            return ExecuteResponse(
                output=(
                    f"{_SANDBOX_ERROR_PREFIX} timeout must be a positive "
                    f"number of seconds (got {timeout})."
                ),
                exit_code=1,
                truncated=False,
            )
        effective_timeout = timeout if timeout is not None else self._default_timeout

        result_holder: dict[str, Any] = {}
        execution_done = threading.Event()

        def run_command() -> None:
            try:
                exit_code, output_bytes = self._container.exec_run(
                    cmd=[
                        "timeout",
                        "-k",
                        str(_KILL_DELAY),
                        str(effective_timeout),
                        "sh",
                        "-c",
                        command,
                    ],
                    stdout=True,
                    stderr=True,
                    demux=False,
                    workdir=self._working_dir,
                )
                output, truncated = _decode_and_truncate(
                    output_bytes or b"", self._max_output_bytes
                )
                if exit_code == _POSIX_TIMEOUT_EXIT_CODE:
                    output += (
                        f"\n\n[Command timed out after {effective_timeout} seconds and was killed.]"
                    )
                result_holder.update(exit_code=exit_code, output=output, truncated=truncated)
            except Exception as e:
                result_holder.update(
                    exit_code=1,
                    output=(
                        f"{_SANDBOX_ERROR_PREFIX} Error executing command: "
                        f"{type(e).__name__}: {e}"
                    ),
                    truncated=False,
                )
            finally:
                execution_done.set()

        thread = threading.Thread(target=run_command, daemon=True)
        thread.start()

        if not execution_done.wait(
            timeout=effective_timeout + _KILL_DELAY + _TIMEOUT_GRACE_SECONDS
        ):
            # The container's own `timeout` should have killed the command by
            # effective_timeout. Reaching here means even that didn't work --
            # Docker itself is unresponsive, not just the command running long.
            return ExecuteResponse(
                output=(
                    f"{_SANDBOX_ERROR_PREFIX} No response from Docker within "
                    f"{effective_timeout + _KILL_DELAY + _TIMEOUT_GRACE_SECONDS} seconds; "
                    f"the command's state is unknown. The sandbox may be unhealthy."
                ),
                exit_code=124,
                truncated=False,
            )

        return ExecuteResponse(
            output=result_holder["output"],
            exit_code=result_holder["exit_code"],
            truncated=result_holder["truncated"],
        )

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files to the container via tar archive."""
        responses: list[FileUploadResponse] = []

        for file_path, content in files:
            try:
                file_path = _resolve_path(file_path, self._working_dir)
                dir_path = file_path.rsplit("/", 1)[0] or "/"

                tar_stream = _build_single_file_tar(file_path, content)

                mkdir_result = self.execute(f"mkdir -p {shlex.quote(dir_path)}")
                if mkdir_result.exit_code != 0:
                    raise RuntimeError(f"mkdir -p {dir_path} failed: {mkdir_result.output}")

                # put_archive would silently replace a directory with a file (observed
                # on Podman), deleting its content. Not atomic with the write below.
                check = self.execute(f"test -d {shlex.quote(file_path)}")
                if check.exit_code == 0:
                    raise IsADirectoryError(
                        f"{file_path} is an existing directory; refusing to replace it"
                    )

                self._container.put_archive("/", tar_stream)

                responses.append(FileUploadResponse(path=file_path, error=None))
                logger.debug("Uploaded file: %s (%d bytes)", file_path, len(content))

            except Exception as e:
                error_msg = f"Failed to upload {file_path}: {type(e).__name__}: {e}"
                logger.error("%s", error_msg)
                responses.append(FileUploadResponse(path=file_path, error=error_msg))

        return responses

    def _download_one(self, file_path: str) -> FileDownloadResponse:
        path = _resolve_path(file_path, self._working_dir)
        try:
            tar_stream, path_stat = self._container.get_archive(path)
            if _is_directory(path_stat):
                # Return before iterating: a large directory is never loaded into memory.
                return FileDownloadResponse(path=file_path, content=None, error="is_directory")
            tar_data = io.BytesIO()
            for chunk in tar_stream:
                tar_data.write(chunk)
            tar_data.seek(0)
            content, error = _extract_single_file(tar_data)
        except docker.errors.NotFound:
            content, error = None, self._not_found_error(path)
        except Exception as e:
            error_msg = f"Failed to download {path}: {type(e).__name__}: {e}"
            logger.error("%s", error_msg)
            content, error = None, error_msg
        return FileDownloadResponse(path=file_path, content=content, error=error)

    def _not_found_error(self, path: str) -> str:
        """The daemon raises NotFound for a missing file and for a missing container."""
        try:
            self._container.reload()
        except docker.errors.NotFound:
            return f"Failed to download {path}: container no longer exists"
        except Exception:
            logger.debug("reload() failed while classifying NotFound", exc_info=True)
        return "file_not_found"

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return [self._download_one(p) for p in paths]

    def verify_timeout(self) -> None:
        """Raise if the container cannot run commands under ``timeout``."""
        exit_code, output_bytes = self._container.exec_run(
            cmd=["sh", "-c", _TIMEOUT_PROBE],
            workdir="/",  # not working_dir: the probe shouldn't depend on it existing
        )
        if exit_code != 0:
            detail = (output_bytes or b"").decode("utf-8", errors="replace").strip()
            raise MissingTimeoutError(
                f"Container image cannot run `{_TIMEOUT_PROBE}` (exit {exit_code}): "
                f"{detail}. ContainerSandbox requires a `timeout` binary "
                f"(GNU coreutils or compatible)."
            )

    def _discover_working_dir(self) -> str:
        self._container.reload()  # refresh attrs from the daemon
        return self._container.attrs.get("Config", {}).get("WorkingDir") or "/"
