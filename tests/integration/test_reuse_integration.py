"""Integration tests for get_or_create(): reuse is checked against a real daemon.

Containers are inspected through the daemon, not through the library's own
objects, so a passing test means the daemon's state is what we claim.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Iterator

import pytest
from docker.models.containers import Container

from agent_container_sandbox import (
    ContainerNotRunningError,
    ContainerSandbox,
    SpecMismatchError,
)

logger = logging.getLogger(__name__)

# What an earlier library version put on its containers: the marker, no fingerprint.
MARKER_LABELS = {"created-by": "agent-container-sandbox"}


@pytest.fixture
def make_foreign_container(client, image, name_prefix) -> Iterator[Callable[..., Container]]:
    """Start a running container that provision() did not create.

    The session sweeper only matches containers with the marker label, so
    this fixture removes everything it made, whatever the test outcome.
    """
    created: list[Container] = []

    def _make(labels: dict[str, str] | None = None) -> Container:
        container = client.containers.run(
            image,
            entrypoint=["sleep", "600"],
            name=f"{name_prefix}{uuid.uuid4().hex[:8]}",
            labels=labels or {},
            detach=True,
        )
        created.append(container)
        return container

    yield _make

    for container in created:
        try:
            container.remove(force=True)
        except Exception:
            logger.warning("Cleanup of %s failed", container.short_id, exc_info=True)


# === get_or_create(): reuse ===================================================


def test_same_spec_twice_returns_the_same_container(provisioner, make_spec):
    spec = make_spec()

    first = provisioner.get_or_create(spec)
    try:
        second = provisioner.get_or_create(spec)
        assert second.id == first.id
    finally:
        provisioner.delete(first)


# === get_or_create(): refusal =================================================


def test_different_spec_raises_and_leaves_the_container_alone(
    provisioner, make_container, make_spec
):
    spec = make_spec()
    container = make_container(name=spec.name)
    other = make_spec(name=spec.name, mem_limit="1g")

    with pytest.raises(SpecMismatchError) as exc_info:
        provisioner.get_or_create(other)

    assert exc_info.value.reason == "different_spec"
    assert exc_info.value.container.id == container.id
    container.reload()
    assert container.status == "running"


def test_stopped_container_raises_and_is_not_restarted(provisioner, make_container, make_spec):
    spec = make_spec()
    container = make_container(name=spec.name)
    container.stop(timeout=2)

    with pytest.raises(ContainerNotRunningError) as exc_info:
        provisioner.get_or_create(spec)

    assert exc_info.value.status != "running"
    assert exc_info.value.container.id == container.id
    container.reload()
    assert container.status != "running"


def test_container_without_labels_raises_not_created_by_library(
    provisioner, make_foreign_container, make_spec
):
    container = make_foreign_container()
    spec = make_spec(name=container.name)

    with pytest.raises(SpecMismatchError) as exc_info:
        provisioner.get_or_create(spec)

    assert exc_info.value.reason == "not_created_by_library"
    assert exc_info.value.container.id == container.id


def test_container_with_only_the_marker_raises_no_fingerprint(
    provisioner, make_foreign_container, make_spec
):
    container = make_foreign_container(labels=MARKER_LABELS)
    spec = make_spec(name=container.name)

    with pytest.raises(SpecMismatchError) as exc_info:
        provisioner.get_or_create(spec)

    assert exc_info.value.reason == "no_fingerprint"
    assert exc_info.value.container.id == container.id


# === find() and ContainerSandbox: bring your own container ====================


def test_find_and_sandbox_accept_a_container_the_library_did_not_create(
    provisioner, make_foreign_container
):
    """The unverified route stays open: find() returns the container that
    get_or_create() refuses, and ContainerSandbox runs commands in it."""
    container = make_foreign_container()

    found = provisioner.find(container.name)

    assert found is not None
    assert found.id == container.id
    assert ContainerSandbox(found).execute("echo hi").output == "hi\n"
