"""LangChain's standard sandbox suite, run against ContainerSandbox.

Known deviations are marked strict xfail, so the run fails as soon as one of
them starts passing and the mark can be reviewed.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from deepagents.backends.protocol import SandboxBackendProtocol
from langchain_tests.integration_tests import SandboxIntegrationTests

from agent_container_sandbox import ContainerProvisioner, ContainerSandbox, ContainerSpec

_upstream_glob = pytest.mark.xfail(
    strict=True,
    reason=(
        "langchain-tests 1.1.9 expects relative glob paths; "
        "deepagents 0.7.32 returns absolute paths."
    ),
)
_relative_paths = pytest.mark.xfail(
    strict=True,
    reason=(
        "upload_files/download_files resolve relative paths against working_dir "
        "(documented); the suite expects an invalid_path error."
    ),
)


class TestContainerSandboxStandard(SandboxIntegrationTests):
    # Keep the inherited cases visible. The tests below pin the current contract.
    test_glob = _upstream_glob(SandboxIntegrationTests.test_glob)
    test_glob_basic_pattern = _upstream_glob(SandboxIntegrationTests.test_glob_basic_pattern)
    test_glob_hidden_files_explicitly = _upstream_glob(
        SandboxIntegrationTests.test_glob_hidden_files_explicitly
    )
    test_glob_with_character_class = _upstream_glob(
        SandboxIntegrationTests.test_glob_with_character_class
    )
    test_glob_with_question_mark = _upstream_glob(
        SandboxIntegrationTests.test_glob_with_question_mark
    )
    test_glob_with_directories = pytest.mark.xfail(
        strict=True,
        reason="deepagents 0.7.32 glob returns files only; the suite expects directories.",
    )(SandboxIntegrationTests.test_glob_with_directories)
    test_write_existing_file_fails = pytest.mark.xfail(
        strict=True,
        reason="deepagents 0.7.32 write replaces existing files; the suite expects an error.",
    )(SandboxIntegrationTests.test_write_existing_file_fails)
    test_upload_relative_path_returns_invalid_path = _relative_paths(
        SandboxIntegrationTests.test_upload_relative_path_returns_invalid_path
    )
    test_download_error_invalid_path_relative = _relative_paths(
        SandboxIntegrationTests.test_download_error_invalid_path_relative
    )

    @pytest.fixture(scope="class")
    def sandbox(
        self, provisioner: ContainerProvisioner, image: str, name_prefix: str
    ) -> Iterator[SandboxBackendProtocol]:
        spec = ContainerSpec(
            image=image,
            name=f"{name_prefix}std-{uuid.uuid4().hex[:8]}",
            max_lifetime_seconds=1800,
        )
        container = provisioner.provision(spec)
        try:
            # The inherited read() ignores ExecuteResponse.truncated, so output
            # clipped at the 64 KiB default breaks parsing of larger files
            # (same family as deepagents#7480).
            sb = ContainerSandbox(container, max_output_size=1_048_576)
            sb.verify_timeout()
            sb.verify_python()
            yield sb
        finally:
            provisioner.delete(container)

    def test_current_glob_contract(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        sandbox_backend.execute(f"mkdir -p {sandbox_test_root}/directory")
        path = f"{sandbox_test_root}/file.txt"
        sandbox_backend.write(path, "data")
        result = sandbox_backend.glob("*", path=sandbox_test_root)
        assert result.error is None
        assert result.matches is not None
        assert [m["path"] for m in result.matches] == [path]

    def test_current_write_contract(
        self, sandbox_backend: SandboxBackendProtocol, sandbox_test_root: str
    ) -> None:
        path = f"{sandbox_test_root}/existing.txt"
        assert sandbox_backend.write(path, "before").error is None
        assert sandbox_backend.write(path, "after").error is None
        assert sandbox_backend.download_files([path])[0].content == b"after"
