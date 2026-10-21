"""Tests for the package's public surface: no Docker daemon required."""

from __future__ import annotations

import importlib

import agent_container_sandbox

EXPECTED_EXPORTS = {
    "ContainerSandbox",
    "ContainerProvisioner",
    "ContainerSpec",
    "SandboxError",
    "DockerImageNotFound",
    "MissingTimeoutError",
    "ManagedContainerSandbox",
    "__version__",
}


def test_all_matches_expected_exports():
    """__all__ is exactly the intended public API: nothing missing, nothing extra."""
    assert set(agent_container_sandbox.__all__) == EXPECTED_EXPORTS


def test_every_name_in_all_is_importable():
    """Every name advertised in __all__ actually resolves on the package."""
    for name in agent_container_sandbox.__all__:
        assert hasattr(agent_container_sandbox, name), f"missing export: {name}"


def test_star_import_exposes_public_names():
    """`from agent_container_sandbox import *` yields exactly __all__."""
    namespace: dict = {}
    exec("from agent_container_sandbox import *", namespace)
    exported = {k for k in namespace if not k.startswith("__") or k == "__version__"}
    assert exported == EXPECTED_EXPORTS


def test_exports_are_the_same_objects_as_in_submodules():
    """Top-level names are re-exports, not accidental copies or shadows."""
    from agent_container_sandbox import provisioning, sandbox

    assert agent_container_sandbox.ContainerSandbox is sandbox.ContainerSandbox
    assert agent_container_sandbox.ContainerProvisioner is provisioning.ContainerProvisioner
    assert agent_container_sandbox.ContainerSpec is provisioning.ContainerSpec
    assert agent_container_sandbox.DockerImageNotFound is provisioning.DockerImageNotFound


def test_version_is_a_nonempty_string():
    """__version__ is always set, even from a source checkout without metadata."""
    assert isinstance(agent_container_sandbox.__version__, str)
    assert agent_container_sandbox.__version__


def test_version_falls_back_when_package_metadata_missing(monkeypatch):
    """Without install metadata, __version__ falls back instead of raising."""
    from importlib import metadata

    def raise_not_found(_name):
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(metadata, "version", raise_not_found)
    reloaded = importlib.reload(agent_container_sandbox)
    try:
        assert reloaded.__version__ == "0+unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(agent_container_sandbox)


def test_no_placeholder_docstrings():
    """Catch leftover scaffolding text in public docstrings."""
    for obj in (
        agent_container_sandbox.ContainerSandbox,
        agent_container_sandbox.ContainerProvisioner,
        agent_container_sandbox.ContainerSpec,
    ):
        assert obj.__doc__, f"{obj.__name__} has no docstring"
        assert "your existing docstring" not in obj.__doc__
