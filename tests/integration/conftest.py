"""Shared fixtures for the integration tests."""

from __future__ import annotations

import io
import logging
import os
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import docker
import pytest
from docker.errors import APIError, ImageNotFound
from docker.models.containers import Container

from agent_container_sandbox import ContainerProvisioner, ContainerSandbox, ContainerSpec

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent

DEFAULT_IMAGE = "ghcr.io/astral-sh/uv:0.10.0-python3.13-trixie-slim"
MARKER_LABEL_FILTER = "created-by=agent-container-sandbox"
BACKSTOP_LIFETIME_SECONDS = 600  # a crashed run leaves nothing behind for long


# -- Collection hooks ----------------------------------------------------------


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    for item in items:
        if _HERE in item.path.parents:
            item.add_marker(pytest.mark.integration)


def pytest_collection_finish(session):
    if not any(item.get_closest_marker("integration") for item in session.items):
        return  # no integration tests selected, so don't touch Docker
    try:
        client = docker.from_env()
        try:
            client.ping()
        finally:
            client.close()
    except Exception as exc:
        pytest.exit(
            f"Integration tests were selected, but the Docker daemon is not "
            f"reachable ({type(exc).__name__}: {exc}). "
            "Start Docker or set DOCKER_HOST.",
            returncode=3,
        )


# -- Session fixtures ----------------------------------------------------------


@pytest.fixture(scope="session")
def client() -> Iterator[docker.DockerClient]:
    """One Docker client for the whole run. Closed by us, as the library expects."""
    docker_client = docker.from_env()
    yield docker_client
    docker_client.close()


@pytest.fixture(scope="session")
def run_id() -> str:
    """Unique per pytest run; ties every container to this run."""
    return uuid.uuid4().hex[:8]


@pytest.fixture(scope="session")
def name_prefix(run_id: str) -> str:
    return f"it-{run_id}-"


@pytest.fixture(scope="session")
def image(client: docker.DockerClient) -> str:
    """Name of the test image, pulled once if it is not present locally."""
    name = os.environ.get("AGENT_SANDBOX_TEST_IMAGE", DEFAULT_IMAGE)
    try:
        client.images.get(name)
    except ImageNotFound:
        try:
            client.images.pull(name)
        except APIError as exc:
            pytest.fail(f"Could not pull test image '{name}': {exc}", pytrace=False)
    return name


@pytest.fixture(scope="session")
def provisioner(client: docker.DockerClient) -> ContainerProvisioner:
    return ContainerProvisioner(client)


@pytest.fixture(scope="session", autouse=True)
def _sweep_leaked_containers(client: docker.DockerClient, name_prefix: str) -> Iterator[None]:
    """Last line of defense: remove containers of THIS run that tests leaked.

    Matches on the marker label AND this run's name prefix. The label alone
    would also hit other people's containers on a shared machine.
    """
    yield
    leaked = client.containers.list(
        all=True,
        filters={"label": MARKER_LABEL_FILTER, "name": name_prefix},
    )
    for container in leaked:
        if not container.name.startswith(name_prefix):
            continue  # Docker's name filter matches substrings
        try:
            container.remove(force=True)
            logger.warning("Swept leaked container %s (%s)", container.name, container.short_id)
        except Exception:
            logger.warning("Could not sweep %s", container.name, exc_info=True)


# -- Per-test factories --------------------------------------------------------


@pytest.fixture
def make_spec(image: str, name_prefix: str) -> Callable[..., ContainerSpec]:
    """Build a ContainerSpec with a prefixed unique name and a short lifetime.

    Use this for tests that hand the spec to ManagedContainerSandbox or the
    provisioner themselves. Any field can be overridden.
    """

    def _make(**overrides) -> ContainerSpec:
        params = {
            "image": image,
            "name": f"{name_prefix}{uuid.uuid4().hex[:8]}",
            "max_lifetime_seconds": BACKSTOP_LIFETIME_SECONDS,
            **overrides,
        }
        return ContainerSpec(**params)

    return _make


@pytest.fixture
def make_container(
    provisioner: ContainerProvisioner, make_spec: Callable[..., ContainerSpec]
) -> Iterator[Callable[..., Container]]:
    """Provision containers on demand; every one is deleted at test end.

    Each test gets its own containers, so tests that exhaust a limit cannot
    disturb the others.
    """
    created: list[Container] = []

    def _make(**spec_overrides) -> Container:
        container = provisioner.provision(make_spec(**spec_overrides))
        created.append(container)
        return container

    yield _make

    for container in created:
        try:
            provisioner.delete(container)
        except Exception:
            logger.warning("Cleanup of %s failed", container.short_id, exc_info=True)


@pytest.fixture
def container(make_container: Callable[..., Container]) -> Container:
    """A fresh container with default spec. Deleted at test end."""
    return make_container()


@pytest.fixture
def sandbox(container: Container) -> ContainerSandbox:
    """A sandbox over the `container` fixture.

    Request both fixtures in one test to get the sandbox and the container
    behind it, for example to stop or delete the container mid-test.
    """
    return ContainerSandbox(container)


@pytest.fixture
def make_sandbox(
    make_container: Callable[..., Container],
) -> Callable[..., ContainerSandbox]:
    """Build a sandbox with custom settings.

    ``spec`` holds ContainerSpec overrides; all other keyword arguments go to
    ContainerSandbox (``working_dir``, ``command_timeout``, ``max_output_size``).
    Example: ``make_sandbox(max_output_size=100)``.
    """

    def _make(*, spec: dict | None = None, **sandbox_kwargs) -> ContainerSandbox:
        return ContainerSandbox(make_container(**(spec or {})), **sandbox_kwargs)

    return _make


@pytest.fixture(scope="session")
def image_without_timeout(client: docker.DockerClient, image: str) -> Iterator[str]:
    """The test image with the `timeout` binary removed (built once per run).

    Yields the image ID, not a tag, so that it works the same on Docker and
    on Podman (which prefixes local tags with "localhost/").
    """
    dockerfile = f"""FROM {image}
RUN rm -f "$(command -v timeout)" && ! command -v timeout
"""
    built, _ = client.images.build(fileobj=io.BytesIO(dockerfile.encode()), rm=True)
    yield built.id
    try:
        client.images.remove(built.id, force=True)
    except Exception:
        logger.warning("Could not remove the test image %s", built.id, exc_info=True)
