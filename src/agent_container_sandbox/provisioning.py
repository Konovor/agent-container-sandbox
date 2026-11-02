"""Container provisioning: create, find, and delete containers.

Owns lifecycle only. ContainerSandbox (sandbox.py) owns the adapter logic and
never creates or deletes anything. The Docker client is injected and owned by
the caller; this module never closes it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shlex
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import docker
from docker.errors import APIError, ImageNotFound
from docker.models.containers import Container

from .errors import ContainerNotRunningError, DockerImageNotFound, SpecMismatchError

logger = logging.getLogger(__name__)

# Applied to every container this module creates, regardless of what the
# caller passes in ContainerSpec.labels. Lets a reaper script find orphans
# ("docker ps -a --filter label=created-by=agent-container-sandbox") even if
# the caller supplied no labels of their own.
_MARKER_LABEL_KEY = "created-by"
_MARKER_LABEL_VALUE = "agent-container-sandbox"

# Hash of the spec a container was created from (see _spec_fingerprint). Set
# after the marker and the caller's labels, so neither can override it.
_FINGERPRINT_LABEL_KEY = "agent-container-sandbox.spec-fingerprint"

_DEFAULT_WORKING_DIR = "/workspace"
_DEFAULT_MAX_LIFETIME_SECONDS = 3600  # hard upper bound; not an idle timeout


@dataclass(frozen=True, kw_only=True)
class ContainerSpec:
    """Immutable description of a container to provision.

    Fields:
        image: Image name.
        name: Container name. optional; None means the provisioner generates
        a unique name; a name is required for ``ContainerProvisioner.find`` /
        ``get_or_create``.
        volumes: Volume mounts, ``{"/host": {"bind": "/container", "mode": "rw"}}``.
        working_dir: Working directory inside the container. Created if missing.
        max_lifetime_seconds: Hard upper bound on how long the container's
            main process runs. The main process is ``sleep <seconds>``, so the
            container stops on its own even if the caller's process crashes.
            This is a maximum lifetime, not an idle timeout — Docker has no
            idle TTL.
        labels: Additional labels merged with the module's fixed marker
            label; callers cannot omit or override the marker this way.

    ``get_or_create`` reuses an existing container only if it carries the
    fingerprint of this spec (every field except ``name``) and is running.
    """

    image: str
    name: str | None = None
    volumes: dict[str, dict[str, str]] = field(default_factory=dict)
    working_dir: str = _DEFAULT_WORKING_DIR
    max_lifetime_seconds: int = _DEFAULT_MAX_LIFETIME_SECONDS
    labels: dict[str, str] = field(default_factory=dict)
    mem_limit: str | None = "2g"
    cpus: float | None = 2.0
    network_mode: str | None = "none"
    pids_limit: int | None = 512
    no_new_privileges: bool = True
    cap_drop_all: bool = True
    cap_add: tuple[str, ...] = ()


def _spec_fingerprint(spec: ContainerSpec) -> str:
    """SHA-256 of the spec's canonical JSON, excluding ``name``.

    The name is the lookup key, and ``provision`` generates one when it is
    None, so including it would stop a container created from a nameless spec
    from ever matching. ``cap_add`` is sorted because its order is irrelevant;
    dictionary keys are sorted by the JSON encoder.
    """
    data = asdict(spec)
    data.pop("name")
    data["cap_add"] = sorted(data["cap_add"])
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _hardening_kwargs(spec: ContainerSpec) -> dict[str, Any]:
    """Translate the spec's hardening fields into ``containers.create`` arguments.

    Disabled settings are omitted, except ``pids_limit``: it is sent as -1
    ("unlimited") so the daemon's own default cannot silently apply.
    """
    kwargs: dict[str, Any] = {}
    if spec.mem_limit is not None:
        kwargs["mem_limit"] = spec.mem_limit
        kwargs["memswap_limit"] = spec.mem_limit  # no swap beyond the cap
    if spec.cpus is not None:
        kwargs["nano_cpus"] = int(spec.cpus * 1_000_000_000)
    if spec.network_mode is not None:
        kwargs["network_mode"] = spec.network_mode
    # Always sent. Omitting it would let the daemon apply its own default
    # (2048 on Podman), so None would not actually mean "unlimited".
    kwargs["pids_limit"] = spec.pids_limit if spec.pids_limit is not None else -1
    if spec.no_new_privileges:
        kwargs["security_opt"] = ["no-new-privileges:true"]
    if spec.cap_drop_all:
        kwargs["cap_drop"] = ["ALL"]
    if spec.cap_add:
        kwargs["cap_add"] = list(spec.cap_add)
    return kwargs


class ContainerProvisioner:
    """Creates, finds, and deletes containers via an injected Docker client.

    Concurrency note: ``get_or_create`` is not safe against two callers
    racing to create a container with the same name at the same time; this
    is a documented limitation, not handled by retrying on a name conflict.
    """

    def __init__(self, client: docker.DockerClient) -> None:
        self._client = client

    def _check_image(self, image: str, auto_pull: bool = False) -> None:
        """Ensure ``image`` is available locally, optionally pulling it.

        Args:
            image: Image name and tag, e.g. ``"python:3.13-slim"``.
            auto_pull: If True, pull the image when it is not found locally.
                If False (the default), a missing image raises immediately.

        Raises:
            DockerImageNotFound: The image is missing and ``auto_pull`` is False,
                or the pull was attempted and failed (the cause is chained).
            docker.errors.APIError: The Docker daemon failed while checking for
                the image. Propagated unchanged.
        """
        try:
            self._client.images.get(image)
        except ImageNotFound:
            if not auto_pull:
                raise DockerImageNotFound(image) from None
            try:
                self._client.images.pull(image)
            except APIError as e:
                raise DockerImageNotFound(image, detail=f"Pulling it failed: {e}") from e

    def provision(self, spec: ContainerSpec) -> Container:
        """Create and start a new container from ``spec`` and prepare it for use.

        If anything fails after the container has been created (starting it or
        creating the working directory), the container is deleted before the
        exception propagates, so a failed call never leaves a container behind.
        If creation itself fails (for example a name conflict), nothing is
        deleted: no container was created by this call.
        """
        self._check_image(spec.image, auto_pull=True)
        name = spec.name or f"agent-container-sandbox-{uuid.uuid4().hex[:12]}"
        # The main process is only a placeholder that keeps the container alive;
        # real work arrives later via exec_run. The entrypoint is set explicitly so
        # an image's own ENTRYPOINT cannot interfere, and init=True runs it behind
        # tini so SIGTERM is forwarded and orphaned child processes are reaped.
        container = self._client.containers.create(
            image=spec.image,
            name=name,
            entrypoint=["sleep", str(spec.max_lifetime_seconds)],
            working_dir=spec.working_dir,
            volumes=spec.volumes,
            labels={
                **spec.labels,
                _MARKER_LABEL_KEY: _MARKER_LABEL_VALUE,
                _FINGERPRINT_LABEL_KEY: _spec_fingerprint(spec),
            },
            stdin_open=False,
            init=True,
            tty=False,
            **_hardening_kwargs(spec),
        )

        # Everything from here on may leave a created container behind, so start()
        # belongs inside the try. create() stays outside: if it raises, no container
        # was assigned and a same-named one belongs to someone else.
        try:
            container.start()
            exit_code, output_bytes = container.exec_run(
                cmd=["sh", "-c", f"mkdir -p {shlex.quote(spec.working_dir)}"],
                workdir=spec.working_dir,
            )
            if exit_code != 0:
                detail = (output_bytes or b"").decode("utf-8", errors="replace")
                raise RuntimeError(f"mkdir -p {spec.working_dir} failed: {detail}")
        except BaseException:
            try:
                self.delete(container)
            except Exception:
                logger.warning("Cleanup of %s failed", container.short_id, exc_info=True)
            raise

        return container

    def find(self, name: str) -> Container | None:
        """Look up an existing container by name.

        Returns:
            The container if one with this name exists, else ``None``.
            Does not distinguish "never existed" from "existed and was
            removed" — both return ``None``.
        """
        candidates = self._client.containers.list(all=True, filters={"name": name})
        exact = [c for c in candidates if c.name == name]
        return exact[0] if exact else None

    def _check_reusable(self, container: Container, spec: ContainerSpec) -> None:
        """Raise unless ``container`` is the running container ``spec`` describes.

        Identity before state: a stopped container with a different spec raises
        ``SpecMismatchError``, because "just start it" would be wrong advice.
        """
        labels = container.labels or {}
        name = spec.name or ""
        if labels.get(_MARKER_LABEL_KEY) != _MARKER_LABEL_VALUE:
            raise SpecMismatchError(name, container, "not_created_by_library")
        stored = labels.get(_FINGERPRINT_LABEL_KEY)
        if stored is None:
            raise SpecMismatchError(name, container, "no_fingerprint")
        if stored != _spec_fingerprint(spec):
            raise SpecMismatchError(name, container, "different_spec")
        if container.status != "running":
            raise ContainerNotRunningError(name, container, container.status)

    def get_or_create(self, spec: ContainerSpec) -> Container:
        """Return the running container created from ``spec``, or provision one.

        Looks the container up by ``spec.name``. If one exists, it must carry
        the library's marker label and the fingerprint of this exact spec, and
        it must be running. Nothing is started, stopped or deleted on failure.

        Raises:
            ValueError: ``spec.name`` is empty.
            SpecMismatchError: The container is not the one this spec describes
                (see its ``reason``).
            ContainerNotRunningError: It is the right container but not running.
        """
        if not spec.name:
            raise ValueError("get_or_create requires spec.name; it is the lookup key.")
        container = self.find(spec.name)
        if container:
            self._check_reusable(container, spec)
            return container
        return self.provision(spec)

    def delete(self, container: Container) -> None:
        """Stop and remove a container.

        Tolerates the container already being gone (``docker.errors.NotFound``,
        from either ``stop()`` or ``remove()``) without raising, since that's a
        successful end state, not a failure.
        """
        try:
            container.stop(timeout=5)
            container.remove(force=True)
        except docker.errors.NotFound:
            logger.debug("Container %s already gone", container.short_id)
            return

        logger.info("Container %s removed", container.short_id)
