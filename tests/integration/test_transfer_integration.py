"""Integration tests for ContainerSandbox.upload_files / download_files.

Uploads are verified by reading the file inside the container, and downloads
are set up by writing the file with execute(). A pure upload-then-download
round trip could pass even if both halves were wrong in the same way.
"""

from __future__ import annotations

import hashlib
import random
import shlex

import pytest


def _put(sandbox, path: str, content: bytes):
    """Upload one file and assert that it worked."""
    [response] = sandbox.upload_files([(path, content)])
    assert response.error is None, response.error
    return response


def _sha256_in_container(sandbox, path: str) -> str:
    result = sandbox.execute(f"sha256sum {shlex.quote(path)}")
    assert result.exit_code == 0, result.output
    return result.output.split()[0]


# === Round trips ==============================================================


def test_binary_round_trip_with_all_byte_values(sandbox):
    data = bytes(range(256))

    _put(sandbox, "all-bytes.bin", data)

    assert _sha256_in_container(sandbox, "/workspace/all-bytes.bin") == hashlib.sha256(data).hexdigest()
    [downloaded] = sandbox.download_files(["all-bytes.bin"])
    assert downloaded.error is None
    assert downloaded.content == data


def test_empty_file_round_trip(sandbox):
    _put(sandbox, "empty.bin", b"")

    assert sandbox.execute("wc -c < /workspace/empty.bin").output.strip() == "0"
    [downloaded] = sandbox.download_files(["empty.bin"])
    assert downloaded.error is None
    assert downloaded.content == b""


def test_multi_megabyte_round_trip(sandbox):
    data = random.Random(1234).randbytes(3 * 1024 * 1024)  # deterministic

    _put(sandbox, "big.bin", data)

    assert _sha256_in_container(sandbox, "/workspace/big.bin") == hashlib.sha256(data).hexdigest()
    [downloaded] = sandbox.download_files(["big.bin"])
    assert downloaded.error is None
    assert downloaded.content == data


def test_relative_path_lands_in_the_working_directory(sandbox):
    _put(sandbox, "rel.txt", b"relative")

    assert sandbox.execute("cat /workspace/rel.txt").output == "relative"


def test_absolute_path_is_used_as_given(sandbox):
    _put(sandbox, "/tmp/abs-upload.txt", b"absolute")

    assert sandbox.execute("cat /tmp/abs-upload.txt").output == "absolute"
    [downloaded] = sandbox.download_files(["/tmp/abs-upload.txt"])
    assert downloaded.content == b"absolute"


# === Path handling ============================================================


def test_missing_parent_directories_are_created(sandbox):
    _put(sandbox, "/workspace/a/b/c/deep.txt", b"deep")

    assert sandbox.execute("cat /workspace/a/b/c/deep.txt").output == "deep"


@pytest.mark.parametrize(
    "name",
    [
        "with space.txt",
        "it's.txt",
        'say "hi".txt',
        "héllo ✓.txt",
        "$HOME `id`.txt",
        "dir with space/nested file.txt",
    ],
)
def test_awkward_file_names_round_trip(sandbox, name):
    content = f"content of {name}".encode()

    _put(sandbox, name, content)

    in_container = sandbox.execute(f"cat {shlex.quote('/workspace/' + name)}")
    assert in_container.output == content.decode()
    [downloaded] = sandbox.download_files([name])
    assert downloaded.error is None
    assert downloaded.content == content


def test_uploading_again_overwrites_the_file(sandbox):
    _put(sandbox, "twice.txt", b"first version, longer")
    _put(sandbox, "twice.txt", b"second")

    assert sandbox.execute("cat /workspace/twice.txt").output == "second"  # no leftover tail


def test_responses_report_resolved_path_on_upload_and_requested_path_on_download(sandbox):
    """Characterization: upload reports the resolved absolute path, download
    echoes the path exactly as the caller gave it. If this is changed to be
    consistent, update the test deliberately."""
    [up] = sandbox.upload_files([("rel.txt", b"x")])
    [down] = sandbox.download_files(["rel.txt"])

    assert up.path == "/workspace/rel.txt"
    assert down.path == "rel.txt"


# === Download errors ==========================================================


def test_downloading_a_missing_file_reports_file_not_found(sandbox):
    [response] = sandbox.download_files(["does-not-exist.txt"])

    assert response.content is None
    assert response.error == "file_not_found"


def test_batch_download_keeps_order_and_reports_per_file(sandbox):
    _put(sandbox, "one.txt", b"1")
    _put(sandbox, "two.txt", b"2")

    responses = sandbox.download_files(["one.txt", "missing.txt", "two.txt"])

    assert [r.path for r in responses] == ["one.txt", "missing.txt", "two.txt"]
    assert [r.content for r in responses] == [b"1", None, b"2"]
    assert [r.error for r in responses] == [None, "file_not_found", None]


def test_downloading_a_directory_reports_is_directory(sandbox):
    sandbox.execute("mkdir -p /workspace/somedir && echo x > /workspace/somedir/f.txt")

    [response] = sandbox.download_files(["somedir"])

    assert response.content is None
    assert response.error == "is_directory"


def test_downloading_a_symlink_does_not_raise(sandbox):
    """Characterization. Podman's archive endpoint follows the link and
    returns the target's content. Docker may return the link itself, which
    tarfile cannot resolve, so that surfaces as an error message. Either
    outcome is acceptable; raising is not."""
    sandbox.execute("echo target > /workspace/real.txt && ln -s real.txt /workspace/link.txt")

    [response] = sandbox.download_files(["link.txt"])

    assert response.content == b"target\n" or isinstance(response.error, str)


# === Upload errors ============================================================


def test_upload_below_an_existing_file_reports_a_per_file_error(sandbox):
    sandbox.execute("echo x > /workspace/afile")

    [response] = sandbox.upload_files([("afile/child.txt", b"data")])

    assert response.error is not None
    assert response.error.startswith("Failed to upload")


def test_upload_onto_an_existing_directory_reports_a_per_file_error(sandbox):
    sandbox.execute("mkdir /workspace/adir && echo x > /workspace/adir/f.txt")

    [response] = sandbox.upload_files([("adir", b"data")])

    assert response.error is not None
    assert sandbox.execute("test -d /workspace/adir").exit_code == 0  # still a directory
    assert sandbox.execute("cat /workspace/adir/f.txt").output == "x\n"  # content untouched


def test_batch_upload_reports_per_file_and_keeps_going(sandbox):
    sandbox.execute("echo x > /workspace/afile")

    responses = sandbox.upload_files(
        [("good1.txt", b"1"), ("afile/bad.txt", b"x"), ("good2.txt", b"2")]
    )

    assert [r.path for r in responses] == [
        "/workspace/good1.txt",
        "/workspace/afile/bad.txt",
        "/workspace/good2.txt",
    ]
    assert [r.error is None for r in responses] == [True, False, True]
    assert sandbox.execute("cat /workspace/good1.txt /workspace/good2.txt").output == "12"


# === Dead container ===========================================================


def test_upload_to_a_removed_container_reports_an_error_without_raising(
    container, sandbox, provisioner
):
    provisioner.delete(container)

    [response] = sandbox.upload_files([("x.txt", b"data")])

    assert response.error is not None
    assert response.error.startswith("Failed to upload")


def test_download_from_a_removed_container_does_not_claim_the_file_is_missing(
    container, sandbox, provisioner
):
    provisioner.delete(container)

    [response] = sandbox.download_files(["anything.txt"])

    assert response.content is None
    assert response.error is not None
    assert response.error != "file_not_found"


# === Metadata of uploaded files ===============================================


def test_uploaded_file_has_default_mode_and_owner(sandbox):
    """Characterization: uploads ignore the container's user and use the tar
    defaults (mode 644, owner root). Revisit with the non-root image."""
    _put(sandbox, "meta.txt", b"x")

    result = sandbox.execute("stat -c '%a %U' /workspace/meta.txt")

    assert result.output.strip() == "644 root"


@pytest.mark.xfail(
    strict=True,
    reason="TarInfo.mtime defaults to 0, so uploaded files are dated 1970. "
    "Undecided whether that needs fixing; remove the marker if it is changed.",
)
def test_uploaded_file_has_a_plausible_modification_time(sandbox):
    _put(sandbox, "mtime.txt", b"x")

    result = sandbox.execute("stat -c %Y /workspace/mtime.txt")

    assert int(result.output.strip()) > 1_000_000_000  # after 2001
