"""Context manager for scripts: provision, adapt, and always delete.

Convenience layer on top of ``ContainerProvisioner`` and ``ContainerSandbox``
for the case where the container lives exactly as long as a ``with`` block.
It adds no capability of its own and delegates everything. For anything more
complex (reuse, leaving containers behind, direct container access), use the
provisioner and sandbox directly.
"""

from __future__ import annotations

import logging
from types import TracebackType

import docker
from docker.models.containers import Container

from .provisioning import ContainerProvisioner, ContainerSpec
from .sandbox import (
    _DEFAULT_COMMAND_TIMEOUT,
    _DEFAULT_MAX_OUTPUT_SIZE,
    ContainerSandbox,
    _validate_limits
)

logger = logging.getLogger(__name__)


class ManagedContainerSandbox:
    """Provision a container, yield a sandbox for it, and always delete it.

    Contract: the container lives exactly as long as the ``with`` block.
    There is no option to keep it, no reuse of existing containers, and no
    access to the underlying ``Container`` object. The yielded sandbox exposes
    ``id`` (the short container ID) for ``docker logs`` / ``docker inspect``.

    Instances are single-use: any second ``__enter__`` raises, even after the
    first block has exited.

    Args:
        client: Docker client. Owned by the caller; never closed here.
        spec: Description of the container to provision. If ``spec.name`` is
            ``None``, the provisioner generates a unique name.
        working_dir: Optional override passed to ``ContainerSandbox``. If
            ``None``, the sandbox discovers the container's configured
            working directory.
        command_timeout: Default per-command timeout in seconds, passed to
            ``ContainerSandbox``.
        max_output_size: Output truncation limit, passed to
            ``ContainerSandbox``.
        verify_timeout: Run ``ContainerSandbox.verify_timeout()`` on entry so
            an image without a usable ``timeout`` binary fails early with a
            clear error.
    """

    def __init__(
        self,
        client: docker.DockerClient,
        spec: ContainerSpec,
        *,
        working_dir: str | None = None,
        command_timeout: int = _DEFAULT_COMMAND_TIMEOUT,
        max_output_size: int = _DEFAULT_MAX_OUTPUT_SIZE,
        verify_timeout: bool = True,
    ) -> None:
        """Store configuration only. No Docker calls happen until ``__enter__``."""
        _validate_limits(command_timeout=command_timeout, max_output_size=max_output_size)
        self._client = client
        self._spec = spec
        self._working_dir = working_dir
        self._command_timeout = command_timeout
        self._max_output_size = max_output_size
        self._verify_timeout = verify_timeout

        self._provisioner: ContainerProvisioner | None = None
        self._container: Container | None = None  # private; never exposed
        self._used = False  # set on first __enter__; never reset

    def __enter__(self) -> ContainerSandbox:
        """Provision the container and return a sandbox over it.

        Raises:
            RuntimeError: This instance was already entered (single-use).
            MissingTimeoutError: ``verify_timeout`` is enabled and the image
                cannot run commands under ``timeout``.

        If anything fails after the container has been created, the container
        is deleted before the exception propagates, because ``__exit__`` does
        not run when ``__enter__`` raises.
        """
        if self._used:
            raise RuntimeError(
                "ManagedContainerSandbox is single-use and was already entered. "
                "Create a new instance."
            )
        self._used = True

        # provision() cleans up after itself on failure, so if it raises there
        # is no container for us to delete and self._container stays None.
        self._provisioner = ContainerProvisioner(self._client)
        self._container = self._provisioner.provision(self._spec)

        try:
            sandbox = ContainerSandbox(
                self._container,
                working_dir=self._working_dir,
                command_timeout=self._command_timeout,
                max_output_size=self._max_output_size,
            )
            if self._verify_timeout:
                sandbox.verify_timeout()
        except BaseException:
            # __exit__ will not run, so delete the container here.
            self._cleanup()
            raise

        return sandbox

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Delete the container, even if the body raised.

        The original exception always propagates (this returns a falsy value).
        A failure during deletion is logged as a warning and never masks the
        body's exception.
        """
        self._cleanup()

    def _cleanup(self) -> None:
        """Delete the container if one exists, logging instead of raising.

        Shared by ``__enter__`` (failed setup) and ``__exit__``.
        """
        # Clear the reference first so a repeated call can never delete twice.
        container, self._container = self._container, None
        if container is None or self._provisioner is None:
            return

        try:
            self._provisioner.delete(container)
        except Exception:
            # Never mask the body's exception (or a failed __enter__'s cause).
            logger.warning(
                "Cleanup of container %s failed; it will stop on its own after "
                "max_lifetime_seconds. Find leftovers with: "
                "docker ps -a --filter label=created-by=agent-container-sandbox",
                container.short_id,
                exc_info=True,
            )
