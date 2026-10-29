"""Integration tests for container lifecycle and cleanup.

The library's headline promise is that ManagedContainerSandbox deletes its
container no matter how the block ends. Every test checks that against the
daemon by name or ID, never through the library's own objects.
"""

from __future__ import annotations

import docker
import pytest

from agent_container_sandbox import ManagedContainerSandbox, MissingTimeoutError


def _exists(client, name: str) -> bool:
    """Exact-name lookup (the daemon's name filter matches substrings)."""
    return any(c.name == name for c in client.containers.list(all=True, filters={"name": name}))


# === ManagedContainerSandbox: cleanup =========================================


def test_container_lives_exactly_as_long_as_the_block(client, make_spec):
    spec = make_spec()

    with ManagedContainerSandbox(client, spec) as sandbox:
        assert _exists(client, spec.name)
        assert sandbox.execute("echo hi").output == "hi\n"

    assert not _exists(client, spec.name)


def test_container_with_generated_name_is_deleted_too(client, make_spec):
    with ManagedContainerSandbox(client, make_spec(name=None)) as sandbox:
        container_id = sandbox.id
        assert client.containers.get(container_id)

    with pytest.raises(docker.errors.NotFound):
        client.containers.get(container_id)


def test_container_is_deleted_when_the_body_raises(client, make_spec):
    spec = make_spec()

    with pytest.raises(RuntimeError, match="boom"):
        with ManagedContainerSandbox(client, spec):
            raise RuntimeError("boom")

    assert not _exists(client, spec.name)


def test_container_is_deleted_on_keyboard_interrupt(client, make_spec):
    """KeyboardInterrupt is a BaseException, so `except Exception` would miss it."""
    spec = make_spec()

    with pytest.raises(KeyboardInterrupt):
        with ManagedContainerSandbox(client, spec):
            raise KeyboardInterrupt

    assert not _exists(client, spec.name)


@pytest.mark.parametrize("action", ["stop", "remove"])
def test_exit_tolerates_a_container_stopped_or_removed_inside_the_block(client, make_spec, action):
    spec = make_spec()

    with ManagedContainerSandbox(client, spec) as sandbox:
        container = client.containers.get(sandbox.id)
        if action == "stop":
            container.stop(timeout=2)
        else:
            container.remove(force=True)

    assert not _exists(client, spec.name)  # and no exception from __exit__


# === ManagedContainerSandbox: failed setup ====================================


def test_missing_timeout_aborts_entry_and_leaves_no_container(
    client, make_spec, image_without_timeout
):
    spec = make_spec(image=image_without_timeout)

    with pytest.raises(MissingTimeoutError, match="timeout"):
        with ManagedContainerSandbox(client, spec):
            pytest.fail("the body must not run")

    assert not _exists(client, spec.name)


def test_without_the_check_a_missing_timeout_shows_up_on_every_command(
    client, make_spec, image_without_timeout
):
    """Checks the README claim: with verify_timeout=False every execute() fails
    with a runtime message about the missing executable. The exact exit code
    may differ between runtimes, so only the substance is asserted."""
    spec = make_spec(image=image_without_timeout)

    with ManagedContainerSandbox(client, spec, verify_timeout=False) as sandbox:
        result = sandbox.execute("echo hi")

    assert result.exit_code != 0
    assert "timeout" in result.output
    assert not _exists(client, spec.name)


def test_name_collision_fails_and_leaves_the_existing_container_untouched(
    client, make_spec, make_container
):
    existing = make_container()

    with pytest.raises(docker.errors.APIError):
        with ManagedContainerSandbox(client, make_spec(name=existing.name)):
            pytest.fail("the body must not run")

    existing.reload()
    assert existing.status == "running"


# === ManagedContainerSandbox: single use ======================================


def test_second_entry_while_active_raises_and_does_not_disturb_the_first(client, make_spec):
    spec = make_spec()
    managed = ManagedContainerSandbox(client, spec)

    with managed as sandbox:
        with pytest.raises(RuntimeError, match="single-use"):
            with managed:
                pass
        assert sandbox.execute("echo still-alive").output == "still-alive\n"

    assert not _exists(client, spec.name)


def test_entering_again_after_exit_raises_and_creates_nothing(client, make_spec):
    spec = make_spec()
    managed = ManagedContainerSandbox(client, spec)
    with managed:
        pass

    with pytest.raises(RuntimeError, match="single-use"):
        with managed:
            pass

    assert not _exists(client, spec.name)


# === ContainerProvisioner.delete ==============================================


def test_delete_twice_is_silent(provisioner, make_container):
    container = make_container()

    provisioner.delete(container)
    provisioner.delete(container)  # already gone: must not raise


def test_delete_after_external_removal_is_silent(provisioner, make_container):
    container = make_container()
    container.remove(force=True)

    provisioner.delete(container)


def test_delete_removes_a_stopped_container(client, provisioner, make_container):
    container = make_container()
    container.stop(timeout=2)

    provisioner.delete(container)

    assert not _exists(client, container.name)


def test_failed_start_leaves_no_container_behind(client, provisioner, make_spec):
    # working_dir="/proc/nope" makes the runtime fail during container init as it
    # cannot create the directory. The runtime fails after the container is created
    # but before the entrypoint runs, so the container exists briefly and needs to
    # be cleaned up. The test checks that the library does not leak the container
    # even if the runtime fails after the container was created. Verified on Podman;
    # runc-based Docker should behave the same.
    spec = make_spec(working_dir="/proc/nope")

    with pytest.raises(docker.errors.APIError):
        provisioner.provision(spec)

    assert not _exists(client, spec.name)
