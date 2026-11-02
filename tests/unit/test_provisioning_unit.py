"""Unit tests for ContainerProvisioner.provision(): no Docker daemon required."""

from __future__ import annotations

import logging
import re
from dataclasses import fields
from unittest.mock import MagicMock

import docker
import pytest

from agent_container_sandbox.errors import ContainerNotRunningError, SpecMismatchError
from agent_container_sandbox.provisioning import (
    _FINGERPRINT_LABEL_KEY,
    _MARKER_LABEL_KEY,
    _MARKER_LABEL_VALUE,
    ContainerProvisioner,
    ContainerSpec,
    DockerImageNotFound,
    _spec_fingerprint,
)

_OTHER_FINGERPRINT = "0" * 64
_MARKER = {_MARKER_LABEL_KEY: _MARKER_LABEL_VALUE}

# One alternative value per field, each different from the default. The
# completeness test below fails when a new ContainerSpec field is missing here.
_ALTERNATIVES = {
    "image": "other:image",
    "volumes": {"/host": {"bind": "/container", "mode": "rw"}},
    "working_dir": "/other",
    "max_lifetime_seconds": 60,
    "labels": {"a": "b"},
    "mem_limit": "1g",
    "cpus": 1.0,
    "network_mode": "bridge",
    "pids_limit": 64,
    "no_new_privileges": False,
    "cap_drop_all": False,
    "cap_add": ("NET_ADMIN",),
}


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture
def mock_container():
    """A fake, freshly-created container. exec_run defaults to a successful
    mkdir (0, b""), since most tests don't care about mkdir's output."""
    container = MagicMock()
    container.short_id = "abc123"
    container.status = "running"
    container.exec_run.return_value = (0, b"")
    return container


@pytest.fixture
def mock_client(mock_container):
    """A fake Docker client. The image check passes by default, and
    containers.create hands out the fake container (whose start() and
    exec_run() succeed by default, see mock_container)."""
    client = MagicMock()
    client.images.get.return_value = MagicMock()  # image "found" by default
    client.containers.create.return_value = mock_container
    return client


@pytest.fixture
def provisioner(mock_client):
    return ContainerProvisioner(mock_client)


@pytest.fixture
def spec():
    return ContainerSpec(image="test:image", name="test-container")


# === ContainerSpec ============================================================


def test_spec_fields_are_keyword_only():
    """Positional arguments would silently shift when fields are added or reordered."""
    with pytest.raises(TypeError):
        ContainerSpec("test:image")  # type: ignore


# === _check_image() ===========================================================


def test_check_image_auto_pull_pulls_when_missing(provisioner, mock_client):
    mock_client.images.get.side_effect = docker.errors.ImageNotFound("missing")

    provisioner._check_image("test:image", auto_pull=True)

    mock_client.images.pull.assert_called_once_with("test:image")


def test_check_image_pull_failure_raises_image_not_found(provisioner, mock_client):
    mock_client.images.get.side_effect = docker.errors.ImageNotFound("missing")
    mock_client.images.pull.side_effect = docker.errors.APIError("registry down")

    with pytest.raises(DockerImageNotFound) as exc_info:
        provisioner._check_image("test:image", auto_pull=True)

    assert exc_info.value.image == "test:image"
    assert "registry down" in str(exc_info.value)


def test_check_image_does_not_pull_by_default(provisioner, mock_client):
    mock_client.images.get.side_effect = docker.errors.ImageNotFound("missing")

    with pytest.raises(DockerImageNotFound):
        provisioner._check_image("test:image")

    mock_client.images.pull.assert_not_called()


# === provision() ==============================================================


def test_provision_happy_path_returns_the_running_container(
    provisioner, mock_client, mock_container, spec
):
    """A successful provision returns exactly what containers.create produced."""
    result = provisioner.provision(spec)

    assert result is mock_container
    mock_client.containers.create.assert_called_once()


def test_provision_applies_marker_label_merged_with_spec_labels(provisioner, mock_client, spec):
    """The fixed marker label is always present, merged with any labels the
    caller supplied — a reaper script must be able to find this container
    regardless of what the caller passed."""
    spec_with_labels = ContainerSpec(image=spec.image, name=spec.name, labels={"team": "x"})

    provisioner.provision(spec_with_labels)

    create_kwargs = mock_client.containers.create.call_args.kwargs
    labels = create_kwargs["labels"]
    assert labels["created-by"] == "agent-container-sandbox"
    assert labels["team"] == "x"


def test_provision_uses_sleep_entrypoint_with_max_lifetime(provisioner, mock_client, spec):
    """The container's main process is `sleep <max_lifetime_seconds>`, so it
    exits on its own even if the caller crashes. The entrypoint is set
    explicitly so an image's own ENTRYPOINT cannot interfere, and no
    `command` is passed that would be appended to it."""
    spec_with_lifetime = ContainerSpec(image=spec.image, name=spec.name, max_lifetime_seconds=1800)

    provisioner.provision(spec_with_lifetime)

    create_kwargs = mock_client.containers.create.call_args.kwargs
    assert create_kwargs["entrypoint"] == ["sleep", "1800"]
    assert "command" not in create_kwargs


def test_provision_runs_keepalive_behind_init_process(provisioner, mock_client, spec):
    """init=True puts Docker's init process (tini) at PID 1, with `sleep` as
    its child, so SIGTERM is forwarded and `stop` doesn't wait out the grace
    period."""
    provisioner.provision(spec)

    create_kwargs = mock_client.containers.create.call_args.kwargs
    assert create_kwargs["init"] is True


def test_provision_deletes_container_when_mkdir_fails(provisioner, mock_container, spec):
    """If creating the working directory fails, the just-created container
    must be cleaned up before the exception propagates — a failed provision
    must not leak a container."""
    mock_container.exec_run.return_value = (1, b"mkdir: permission denied")

    with pytest.raises(RuntimeError, match="mkdir"):
        provisioner.provision(spec)

    mock_container.stop.assert_called_once()
    mock_container.remove.assert_called_once_with(force=True)


def test_provision_generates_name_when_spec_has_none(provisioner, mock_client):
    """name=None means the provisioner picks a readable prefix plus a short
    random suffix."""
    provisioner.provision(ContainerSpec(image="test:image"))

    name = mock_client.containers.create.call_args.kwargs["name"]
    assert re.fullmatch(r"agent-container-sandbox-[0-9a-f]{12}", name)


def test_provision_generated_names_are_unique(provisioner, mock_client):
    """Two containers from the same provisioner and the same nameless spec
    must not collide (regression guard against deriving the name from
    per-provisioner state such as id(self))."""
    nameless = ContainerSpec(image="test:image")

    provisioner.provision(nameless)
    provisioner.provision(nameless)

    names = [c.kwargs["name"] for c in mock_client.containers.create.call_args_list]
    assert len(names) == 2
    assert names[0] != names[1]


def test_provision_generated_name_is_valid_for_images_with_registry_path(provisioner, mock_client):
    """The image string never leaks into the generated name, so slashes and
    digests in image references cannot produce an invalid Docker name."""
    image = "ghcr.io/astral-sh/uv:0.10.0-python3.13-trixie-slim"

    provisioner.provision(ContainerSpec(image=image))

    name = mock_client.containers.create.call_args.kwargs["name"]
    assert re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name)


def test_provision_uses_explicit_name_verbatim(provisioner, mock_client, spec):
    """A caller-supplied name is passed through untouched."""
    provisioner.provision(spec)

    assert mock_client.containers.create.call_args.kwargs["name"] == "test-container"


def test_provision_applies_marker_label_when_name_is_generated(provisioner, mock_client):
    """Orphans with generated names stay findable through the marker label."""
    provisioner.provision(ContainerSpec(image="test:image"))

    labels = mock_client.containers.create.call_args.kwargs["labels"]
    assert labels["created-by"] == "agent-container-sandbox"


def test_provision_deletes_container_when_start_fails(provisioner, mock_container, spec):
    mock_container.start.side_effect = docker.errors.APIError("start failed")

    with pytest.raises(docker.errors.APIError):
        provisioner.provision(spec)

    mock_container.exec_run.assert_not_called()
    mock_container.stop.assert_called_once()
    mock_container.remove.assert_called_once_with(force=True)


def test_provision_does_not_delete_anything_when_create_fails(
    provisioner, mock_client, mock_container, spec
):
    mock_client.containers.create.side_effect = docker.errors.APIError("409 name conflict")
    provisioner.delete = MagicMock()

    with pytest.raises(docker.errors.APIError):
        provisioner.provision(spec)

    provisioner.delete.assert_not_called()
    mock_container.stop.assert_not_called()
    mock_container.remove.assert_not_called()


def test_provision_starts_container_once_before_exec(provisioner, mock_container, spec):
    result = provisioner.provision(spec)

    assert result is mock_container
    mock_container.start.assert_called_once_with()
    names = [c[0] for c in mock_container.mock_calls]
    assert names.index("start") < names.index("exec_run")


def test_provision_logs_warning_and_reraises_original_when_cleanup_fails(
    provisioner, mock_container, spec, caplog
):
    mock_container.start.side_effect = docker.errors.APIError("start failed")
    mock_container.stop.side_effect = RuntimeError("daemon unreachable")

    with caplog.at_level(logging.WARNING):
        with pytest.raises(docker.errors.APIError, match="start failed"):
            provisioner.provision(spec)

    assert "Cleanup of abc123 failed" in caplog.text


# === find() ====================================================================


def test_find_returns_existing_container(provisioner, mock_client, mock_container):
    """An existing container is returned."""
    mock_container.name = "test-container"
    mock_client.containers.list.return_value = [mock_container]

    result = provisioner.find("test-container")

    assert result is mock_container


def test_find_returns_none_when_no_match(provisioner, mock_client):
    """No matching container returns None, rather than raising."""
    mock_client.containers.list.return_value = []

    result = provisioner.find("does-not-exist")

    assert result is None


# === _spec_fingerprint() ======================================================


def test_fingerprint_table_covers_every_spec_field():
    """A new ContainerSpec field must get an alternative value here (name is
    excluded from the fingerprint and tested separately)."""
    assert {f.name for f in fields(ContainerSpec)} == set(_ALTERNATIVES) | {"name"}


@pytest.mark.parametrize("field_name", sorted(_ALTERNATIVES))
def test_fingerprint_changes_when_a_single_field_changes(field_name):
    base = {"image": "test:image", "name": "n"}
    changed = {**base, field_name: _ALTERNATIVES[field_name]}

    assert _spec_fingerprint(ContainerSpec(**changed)) != _spec_fingerprint(ContainerSpec(**base))


def test_fingerprint_ignores_name():
    a = ContainerSpec(image="test:image", name="a")
    b = ContainerSpec(image="test:image", name="b")
    nameless = ContainerSpec(image="test:image")

    assert _spec_fingerprint(a) == _spec_fingerprint(b) == _spec_fingerprint(nameless)


def test_fingerprint_is_stable_across_dict_and_cap_add_order():
    a = ContainerSpec(
        image="test:image",
        volumes={"/x": {"bind": "/a", "mode": "ro"}, "/y": {"bind": "/b", "mode": "rw"}},
        labels={"one": "1", "two": "2"},
        cap_add=("CHOWN", "KILL"),
    )
    b = ContainerSpec(
        image="test:image",
        volumes={"/y": {"mode": "rw", "bind": "/b"}, "/x": {"mode": "ro", "bind": "/a"}},
        labels={"two": "2", "one": "1"},
        cap_add=("KILL", "CHOWN"),
    )

    assert _spec_fingerprint(a) == _spec_fingerprint(b)


def test_provision_sets_fingerprint_label(provisioner, mock_client, spec):
    provisioner.provision(spec)

    labels = mock_client.containers.create.call_args.kwargs["labels"]
    assert labels[_FINGERPRINT_LABEL_KEY] == _spec_fingerprint(spec)
    assert labels[_MARKER_LABEL_KEY] == _MARKER_LABEL_VALUE


def test_user_labels_cannot_override_marker_or_fingerprint(provisioner, mock_client):
    spec = ContainerSpec(
        image="test:image",
        name="test-container",
        labels={_FINGERPRINT_LABEL_KEY: "forged", _MARKER_LABEL_KEY: "forged"},
    )

    provisioner.provision(spec)

    labels = mock_client.containers.create.call_args.kwargs["labels"]
    assert labels[_FINGERPRINT_LABEL_KEY] == _spec_fingerprint(spec)
    assert labels[_MARKER_LABEL_KEY] == _MARKER_LABEL_VALUE


# === get_or_create() ===========================================================


def test_get_or_create_returns_existing_container_without_provisioning(
    provisioner, mock_container, spec
):
    """If find() locates a matching, running container, provision() must never run."""
    mock_container.labels = {**_MARKER, _FINGERPRINT_LABEL_KEY: _spec_fingerprint(spec)}
    mock_container.status = "running"
    provisioner.find = MagicMock(return_value=mock_container)
    provisioner.provision = MagicMock()

    result = provisioner.get_or_create(spec)

    assert result is mock_container
    provisioner.provision.assert_not_called()


def test_get_or_create_provisions_when_not_found(provisioner, mock_container, spec):
    """If find() finds nothing, provision() is called and its result returned."""
    provisioner.find = MagicMock(return_value=None)
    provisioner.provision = MagicMock(return_value=mock_container)

    result = provisioner.get_or_create(spec)

    assert result is mock_container
    provisioner.provision.assert_called_once_with(spec)


@pytest.mark.parametrize("bad_name", [None, ""])
def test_get_or_create_rejects_spec_without_name(provisioner, bad_name):
    """A nameless spec has no lookup key, so get_or_create refuses it before
    searching or provisioning anything."""
    provisioner.find = MagicMock()
    provisioner.provision = MagicMock()

    with pytest.raises(ValueError, match="name"):
        provisioner.get_or_create(ContainerSpec(image="test:image", name=bad_name))

    provisioner.find.assert_not_called()
    provisioner.provision.assert_not_called()


def _assert_container_untouched(provisioner, container):
    provisioner.provision.assert_not_called()
    provisioner.delete.assert_not_called()
    container.start.assert_not_called()
    container.stop.assert_not_called()
    container.remove.assert_not_called()


@pytest.mark.parametrize(
    ("labels", "status", "reason"),
    [
        ({}, "running", "not_created_by_library"),
        (None, "running", "not_created_by_library"),
        ({"other": "x"}, "running", "not_created_by_library"),
        (_MARKER, "running", "no_fingerprint"),
        ({**_MARKER, _FINGERPRINT_LABEL_KEY: _OTHER_FINGERPRINT}, "running", "different_spec"),
        # Identity before state: a stopped container that is the wrong one
        # must raise SpecMismatchError, not ContainerNotRunningError.
        ({}, "exited", "not_created_by_library"),
        (_MARKER, "exited", "no_fingerprint"),
        ({**_MARKER, _FINGERPRINT_LABEL_KEY: _OTHER_FINGERPRINT}, "exited", "different_spec"),
    ],
)
def test_get_or_create_raises_spec_mismatch(
    provisioner, mock_container, spec, labels, status, reason
):
    mock_container.labels = labels
    mock_container.status = status
    provisioner.find = MagicMock(return_value=mock_container)
    provisioner.provision = MagicMock()
    provisioner.delete = MagicMock()

    with pytest.raises(SpecMismatchError) as exc_info:
        provisioner.get_or_create(spec)

    assert exc_info.value.reason == reason
    assert exc_info.value.container is mock_container
    _assert_container_untouched(provisioner, mock_container)


@pytest.mark.parametrize("status", ["exited", "created", "paused"])
def test_get_or_create_raises_when_matching_container_is_not_running(
    provisioner, mock_container, spec, status
):
    mock_container.labels = {**_MARKER, _FINGERPRINT_LABEL_KEY: _spec_fingerprint(spec)}
    mock_container.status = status
    provisioner.find = MagicMock(return_value=mock_container)
    provisioner.provision = MagicMock()
    provisioner.delete = MagicMock()

    with pytest.raises(ContainerNotRunningError) as exc_info:
        provisioner.get_or_create(spec)

    assert exc_info.value.status == status
    assert exc_info.value.container is mock_container
    _assert_container_untouched(provisioner, mock_container)


def test_find_does_not_verify_the_container(provisioner, mock_client, mock_container):
    """find() is a plain lookup: it returns containers without the marker or
    fingerprint labels, which keeps 'bring your own container' working and
    lets callers recover from a ContainerReuseError."""
    mock_container.name = "hand-made"
    mock_container.labels = {}
    mock_container.status = "exited"
    mock_client.containers.list.return_value = [mock_container]

    assert provisioner.find("hand-made") is mock_container


# === delete() ===================================================================


def test_delete_stops_then_removes_container(provisioner, mock_container):
    """The normal case: stop() then remove()."""
    provisioner.delete(mock_container)

    mock_container.stop.assert_called_once()
    mock_container.remove.assert_called_once()


@pytest.mark.parametrize("failing_method", ["stop", "remove"])
def test_delete_tolerates_container_already_gone(provisioner, mock_container, failing_method):
    """A NotFound from either stop() or remove() means the container is
    already gone -- that's a successful end state, not a failure."""
    getattr(mock_container, failing_method).side_effect = docker.errors.NotFound("gone")

    provisioner.delete(mock_container)  # must not raise


def test_delete_propagates_unexpected_errors(provisioner, mock_container):
    """Anything other than NotFound is a real failure and must not be
    swallowed -- unlike the original close(), which logged and continued."""
    mock_container.stop.side_effect = RuntimeError("daemon unreachable")

    with pytest.raises(RuntimeError, match="daemon unreachable"):
        provisioner.delete(mock_container)


# === provision(): working_dir ===================================================


def test_provision_uses_working_dir_for_container_and_mkdir(
    provisioner, mock_client, mock_container
):
    """The spec's working_dir becomes the container's working directory and is
    the directory that gets created."""
    custom = ContainerSpec(image="test:image", name="n", working_dir="/my dir")

    provisioner.provision(custom)

    assert mock_client.containers.create.call_args.kwargs["working_dir"] == "/my dir"
    mkdir_cmd = mock_container.exec_run.call_args.kwargs["cmd"]
    assert mkdir_cmd == ["sh", "-c", "mkdir -p '/my dir'"]


# === hardening ===================================================


def test_provision_applies_hardening_defaults(provisioner, mock_client, spec):
    """A bare spec yields a hardened container."""
    provisioner.provision(spec)

    kw = mock_client.containers.create.call_args.kwargs
    assert kw["mem_limit"] == "2g"
    assert kw["memswap_limit"] == "2g"
    assert kw["nano_cpus"] == 2_000_000_000
    assert kw["network_mode"] == "none"
    assert kw["pids_limit"] == 512
    assert kw["security_opt"] == ["no-new-privileges:true"]
    assert kw["cap_drop"] == ["ALL"]
    assert "cap_add" not in kw


def test_provision_omits_hardening_when_disabled(provisioner, mock_client):
    """Turning every setting off leaves the container unrestricted.

    Disabled settings are omitted from the run arguments, except pids_limit:
    it is sent as -1 ("unlimited"), because omitting it would let the
    daemon's own default (for example 2048 on Podman) apply.
    """
    relaxed = ContainerSpec(
        image="test:image",
        name="test-container",
        mem_limit=None,
        cpus=None,
        network_mode=None,
        pids_limit=None,
        no_new_privileges=False,
        cap_drop_all=False,
    )

    provisioner.provision(relaxed)

    kw = mock_client.containers.create.call_args.kwargs
    for key in (
        "mem_limit",
        "memswap_limit",
        "nano_cpus",
        "network_mode",
        "security_opt",
        "cap_drop",
        "cap_add",
    ):
        assert key not in kw
    # Not omitted like the others: an explicit "unlimited" overrides the daemon default.
    assert kw["pids_limit"] == -1


def test_provision_passes_cap_add_after_dropping_all(provisioner, mock_client):
    """Capabilities can be added back selectively."""
    spec_with_caps = ContainerSpec(
        image="test:image",
        name="test-container",
        cap_add=("CHOWN", "DAC_OVERRIDE"),
    )

    provisioner.provision(spec_with_caps)

    kw = mock_client.containers.create.call_args.kwargs
    assert kw["cap_drop"] == ["ALL"]
    assert kw["cap_add"] == ["CHOWN", "DAC_OVERRIDE"]


def test_provision_converts_cpus_to_nano_cpus(provisioner, mock_client):
    spec_half_cpu = ContainerSpec(image="test:image", name="test-container", cpus=0.5)

    provisioner.provision(spec_half_cpu)

    assert mock_client.containers.create.call_args.kwargs["nano_cpus"] == 500_000_000


def test_provision_passes_custom_network_mode(provisioner, mock_client):
    """A custom network mode is forwarded verbatim to Docker."""
    spec_custom_net = ContainerSpec(
        image="test:image", name="test-container", network_mode="bridge"
    )

    provisioner.provision(spec_custom_net)

    assert mock_client.containers.create.call_args.kwargs["network_mode"] == "bridge"
