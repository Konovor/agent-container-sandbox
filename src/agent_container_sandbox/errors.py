"""Exceptions raised by agent-container-sandbox.

This module only defines exception classes. It must not import anything from
the rest of the package, so that both ``sandbox`` and ``provisioning`` can
import from it without risk of circular imports.

Hierarchy::

    SandboxError
    ├── DockerImageNotFound
    ├── MissingTimeoutError
    ├── MissingPythonError
    └── ContainerReuseError
        ├── SpecMismatchError
        └── ContainerNotRunningError

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from docker.models.containers import Container

MismatchReason = Literal["not_created_by_library", "no_fingerprint", "different_spec"]


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
    ``edit``, ``ls``, ``glob``) run Python inside the container.
    ``execute`` and the file transfer methods do not need it.
    """


class ContainerReuseError(SandboxError):
    """Raised when ``get_or_create`` finds the name but will not hand the container out.

    Catch this to handle every reuse failure in one place. The container that
    failed the check is available as ``container``, so recovery is a one-liner:
    ``e.container.start()`` or ``provisioner.delete(e.container)``.
    """

    def __init__(self, name: str, container: Container, message: str) -> None:
        self.name = name
        self.container = container
        super().__init__(message)


class SpecMismatchError(ContainerReuseError):
    """Raised when the container found is not the one this spec describes.

    ``reason`` says which check failed:

    - ``not_created_by_library``: the library's marker label is missing, so
      the name belongs to something else.
    - ``no_fingerprint``: the marker label is present but the spec fingerprint
      is not, which means an earlier library version created it.
    - ``different_spec``: the container's fingerprint differs from the spec's.
    """

    def __init__(self, name: str, container: Container, reason: MismatchReason) -> None:
        self.reason = reason
        if reason == "not_created_by_library":
            message = (
                f"Container '{name}' exists but was not created by agent-container-sandbox, "
                "so get_or_create will not reuse it. Use another spec.name, or remove it "
                "with provisioner.delete(e.container) if you are sure it is unused."
            )
        elif reason == "no_fingerprint":
            message = (
                f"Container '{name}' was created by agent-container-sandbox but has no spec "
                "fingerprint, probably because an earlier library version created it, so "
                "get_or_create cannot tell whether it matches this spec. Delete it with "
                "provisioner.delete(e.container) to get a new one."
            )
        else:
            message = (
                f"Container '{name}' was created from a different spec (its spec "
                "fingerprint differs). Delete it with provisioner.delete(e.container) to "
                "get a fresh one from this spec, or use another spec.name."
            )
        super().__init__(name, container, message)


class ContainerNotRunningError(ContainerReuseError):
    """Raised when the container matches the spec but is not running.

    ``status`` is the status string the runtime reported. An expired
    ``max_lifetime_seconds`` is the common cause.
    """

    def __init__(self, name: str, container: Container, status: str) -> None:
        self.status = status
        super().__init__(
            name,
            container,
            f"Container '{name}' matches the spec but is not running (status: {status}). "
            "Start it with e.container.start(), or delete it with "
            "provisioner.delete(e.container) to get a fresh one. An expired "
            "max_lifetime_seconds is the common cause.",
        )
