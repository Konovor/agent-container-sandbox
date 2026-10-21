"""Integration tests: verify that a real Docker daemon enforces the hardening.

The unit tests mock the Docker client, so they only prove that we *pass* the
right arguments. These tests prove that Docker actually applies them.

Requirements:
    - A running Docker daemon (the whole module is skipped if there is none).
    - An image with ``sh``, ``sleep``, ``timeout`` and ``python3``. Override the
      default with the AGENT_SANDBOX_TEST_IMAGE environment variable.

Run:
    pytest tests/integration -m integration
Skip in unit-test runs:
    pytest -m "not integration"
"""

from __future__ import annotations

import logging
import os
import time
import uuid

import docker
import pytest

from agent_container_sandbox import (
    ContainerProvisioner,
    ContainerSandbox,
    ContainerSpec,
)

pytestmark = pytest.mark.integration

logger = logging.getLogger(__name__)

IMAGE = os.environ.get(
    "AGENT_SANDBOX_TEST_IMAGE",
    "ghcr.io/astral-sh/uv:0.10.0-python3.13-trixie-slim",
)

# -- Fixtures -----------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    try:
        docker_client = docker.from_env()
        docker_client.ping()
    except Exception as e:  # daemon not installed, not running, or not reachable
        pytest.skip(f"Docker daemon not available: {e}")
    yield docker_client
    docker_client.close()


@pytest.fixture
def provisioner(client):
    return ContainerProvisioner(client)


@pytest.fixture
def make_container(provisioner):
    """Factory that provisions containers and always deletes them afterwards.

    Each test gets its own container, so tests that exhaust a limit (such as
    the PID test) cannot disturb the others.
    """
    created = []

    def _make(**spec_overrides):
        spec = ContainerSpec(
            image=IMAGE,
            name=f"it-{uuid.uuid4().hex[:8]}",
            **spec_overrides,
        )
        container = provisioner.provision(spec)
        created.append(container)
        return container

    yield _make

    for container in created:
        try:
            provisioner.delete(container)
        except Exception:
            logger.warning("Cleanup of %s failed", container.short_id, exc_info=True)


def _proc_status(sandbox: ContainerSandbox) -> dict[str, str]:
    """Return CapEff and NoNewPrivs from /proc/self/status inside the container."""
    result = sandbox.execute("grep -E 'CapEff|NoNewPrivs' /proc/self/status")
    assert result.exit_code == 0, result.output
    fields = {}
    for line in result.output.splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


# === Defaults are enforced =====================================================


def test_default_container_has_no_capabilities_and_no_new_privileges(make_container):
    sandbox = ContainerSandbox(make_container())

    status = _proc_status(sandbox)

    assert int(status["CapEff"], 16) == 0
    assert status["NoNewPrivs"] == "1"


def test_default_memory_limit_kills_runaway_allocation(make_container):
    """Allocating 3 GiB under the 2g default must be OOM-killed (exit 137)."""
    sandbox = ContainerSandbox(make_container())

    result = sandbox.execute('python3 -c "x = bytearray(3 * 1024**3)"')

    assert result.exit_code == 137, result.output


def test_default_pids_limit_stops_fork_bomb(make_container):
    """600 background processes exceed the 512 default and must fail to fork."""
    sandbox = ContainerSandbox(make_container())

    result = sandbox.execute("for i in $(seq 1 600); do sleep 30 & done; wait")

    assert result.exit_code != 0
    output = result.output.lower()
    assert "fork" in output or "resource temporarily unavailable" in output


def test_container_config_reflects_defaults(make_container):
    """Read back what Docker actually stored for the container.

    This is how the CPU limit is checked: a CPU cap throttles instead of
    failing, so there is no simple command whose result proves it.
    """
    container = make_container()
    container.reload()
    host_config = container.attrs["HostConfig"]

    assert host_config["Init"] is True
    assert host_config["Memory"] == 2 * 1024**3
    assert host_config["MemorySwap"] == 2 * 1024**3
    assert host_config["NanoCpus"] == 2_000_000_000
    assert host_config["PidsLimit"] == 512
    assert "ALL" in (host_config["CapDrop"] or [])
    assert "no-new-privileges:true" in (host_config["SecurityOpt"] or [])


# === Overrides and opt-outs work ===============================================


def test_cap_add_restores_selected_capabilities(make_container):
    sandbox = ContainerSandbox(make_container(cap_add=("CHOWN",)))

    status = _proc_status(sandbox)

    assert int(status["CapEff"], 16) != 0  # CHOWN is back, everything else stays dropped


def test_relaxed_spec_removes_the_restrictions(make_container):
    container = make_container(
        mem_limit=None,
        cpus=None,
        pids_limit=None,
        no_new_privileges=False,
        cap_drop_all=False,
    )
    sandbox = ContainerSandbox(container)

    status = _proc_status(sandbox)
    container.reload()
    host_config = container.attrs["HostConfig"]

    assert int(status["CapEff"], 16) != 0
    assert status["NoNewPrivs"] == "0"
    assert host_config["Memory"] == 0
    assert host_config["NanoCpus"] == 0
    assert not host_config["PidsLimit"]  # unlimited (0 or None, depending on the Docker version)


# === Lifecycle ==================================================================


def test_delete_returns_well_before_the_stop_grace_period(provisioner, make_container):
    """With init=True, SIGTERM reaches `sleep` and the container stops at once.

    Without an init process, a bare `sleep` as PID 1 ignores SIGTERM and
    delete() waits out the full 5 s grace period. The 4 s threshold sits
    between the two; raise it if your machine is very slow.
    """
    container = make_container()

    start = time.monotonic()
    provisioner.delete(container)
    elapsed = time.monotonic() - start

    assert elapsed < 4.0, f"delete() took {elapsed:.1f}s; is init=True still applied?"
