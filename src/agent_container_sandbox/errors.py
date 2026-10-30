"""Exceptions raised by agent-container-sandbox.

This module only defines exception classes. It must not import anything from
the rest of the package, so that both ``sandbox`` and ``provisioning`` can
import from it without risk of circular imports.

Hierarchy::

    SandboxError
    ├── DockerImageNotFound
    ├── MissingTimeoutError
    └── MissingPythonError

"""

from __future__ import annotations


class SandboxError(Exception):
    """Base class for every error this library raises on purpose.

    Catch this to handle any library-specific failure in one place.
    """


class DockerImageNotFound(SandboxError):
    """Raised when an image is not available locally (and could not be pulled)."""

    def __init__(self, image: str, detail: str | None = None) -> None:
        self.image = image
        message = f"Image '{image}' is not available locally."
        if detail:
            message += f" {detail}"
        else:
            message += f" Run `docker pull {image}` first, or enable auto_pull."
        super().__init__(message)


class MissingTimeoutError(SandboxError):
    """Raised when a container cannot run commands under ``timeout``.

    ``ContainerSandbox`` wraps every command in ``timeout``, so the image
    must ship a ``timeout`` binary (GNU coreutils or compatible).
    """


class MissingPythonError(SandboxError):
    """Raised when a container has no usable ``python3``.

    The file helpers inherited from ``BaseSandbox`` (``read``, ``write``,
    ``edit``, ``ls``, ``grep``, ``glob``) run Python inside the container.
    ``execute`` and the file transfer methods do not need it.
    """
