"""Unit tests for sandbox.py's pure module-level helpers."""

from __future__ import annotations

import io
import tarfile

from agent_container_sandbox.sandbox import (
    _decode_and_truncate,
    _extract_single_file,
    _is_directory,
)

# -- Helpers to build test fixtures ------------------------------------------


def _make_tar(name: str, content: bytes) -> io.BytesIO:
    """Build a single-file tar archive in memory, matching what
    _build_single_file_tar / put_archive produce."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        info = tarfile.TarInfo(name=name)
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    stream.seek(0)
    return stream


def _make_empty_tar() -> io.BytesIO:
    """A structurally valid tar archive with no members."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w"):
        pass
    stream.seek(0)
    return stream


# -- _decode_and_truncate -----------------------------------------------------


def test_decode_and_truncate_under_limit_not_truncated():
    """Output within the byte limit is returned as-is."""
    output, truncated = _decode_and_truncate(b"hello", limit=100)

    assert truncated is False
    assert output == "hello"


def test_decode_and_truncate_over_limit_is_truncated():
    """Output beyond the byte limit is cut and flagged."""
    output, truncated = _decode_and_truncate(b"x" * 100, limit=10)

    assert truncated is True
    assert output.startswith("x" * 10)
    assert "x" * 11 not in output  # nothing beyond the limit survived
    assert "truncated at 10 bytes" in output


def test_decode_and_truncate_exactly_at_limit_not_truncated():
    """Output exactly at the byte limit is NOT truncated (off-by-one guard)."""
    output, truncated = _decode_and_truncate(b"x" * 10, limit=10)

    assert truncated is False
    assert output == "x" * 10


def test_decode_and_truncate_splits_multibyte_character_cleanly():
    """A multi-byte UTF-8 character cut exactly at the byte limit must not
    produce a stray replacement character from a half-read code point."""
    raw = "é".encode() * 20  # "é" is 2 bytes in UTF-8
    limit = 15  # odd: cuts the 8th "é" in half

    output, truncated = _decode_and_truncate(raw, limit=limit)

    assert truncated is True
    assert "\ufffd" not in output.split("\n\n")[0]  # no replacement char in the kept text
    assert output.startswith("é" * 7)
    assert "truncated at 15 bytes" in output


# -- _extract_single_file -----------------------------------------------------


def test_extract_single_file_returns_content_for_normal_file():
    """A normal file entry is read back with no error."""
    tar_data = _make_tar("workspace/test.txt", b"hello world")

    content, error = _extract_single_file(tar_data)

    assert content == b"hello world"
    assert error is None


def test_extract_single_file_empty_archive_returns_file_not_found():
    """An archive with no members (e.g. an unexpected empty response) is
    treated as a missing file rather than raising."""
    tar_data = _make_empty_tar()

    content, error = _extract_single_file(tar_data)

    assert content is None
    assert error == "file_not_found"


# -- _is_directory ------------------------------------------------------------


def test_is_directory_true_for_real_go_filemode():
    """Header from container.get_archive on a directory: os.ModeDir (bit 31) | 0o755."""
    assert _is_directory({"mode": 2147484141}) is True


def test_is_directory_false_for_regular_file_mode():
    assert _is_directory({"mode": 0o644}) is False


def test_is_directory_ignores_posix_directory_bits():
    """POSIX S_IFDIR is not what Docker sends. Treating it as a directory flag
    is the original bug's blind spot."""
    assert _is_directory({"mode": 0o040755}) is False


def test_is_directory_false_without_usable_header():
    assert _is_directory(None) is False
    assert _is_directory({}) is False
