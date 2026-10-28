"""Header checks reject sidecars without mistaking them for MATLAB v4."""

import struct

import h5py
import numpy as np
import pytest
from scipy.io import savemat

from ctapdash.io.eeglab.validation import set_appears_valid
from ctapdash.io.paths import ObservationData
from ctapdash.io.cache import scan_dataset
from tests.dummy_data import write_eeglab


@pytest.mark.parametrize("format", ["4", "5"])
def test_mat_headers(tmp_path, format):
    path = tmp_path / "p.set"
    savemat(path, {"samples": np.ones((2, 3))}, format=format)
    assert set_appears_valid(path) is True


def test_big_endian_v4_header(tmp_path):
    path = tmp_path / "p.set"
    path.write_bytes(
        struct.pack(">5i", 1000, 1, 1, 0, 2) + b"x\0" + struct.pack(">d", 1)
    )
    assert set_appears_valid(path) is True


def test_v73_signature_in_user_block(tmp_path):
    path = tmp_path / "p.set"
    with h5py.File(path, "w", userblock_size=512) as stream:
        stream["samples"] = [1]
    header = b"MATLAB 7.3 MAT-file".ljust(124, b" ") + b"\x00\x02IM"
    with path.open("r+b") as stream:
        stream.write(header)
    assert set_appears_valid(path) is True
    # A version marker alone must not be enough.
    path.write_bytes(header + b"\0" * 512)
    assert set_appears_valid(path) is False


@pytest.mark.parametrize(
    "contents",
    [
        b"",
        b"MATLAB",
        b"x" * 128,
        struct.pack(">II16sH", 0x00051607, 0x00020000, b"Mac OS X        ", 1)
        + b"\0" * 102,
        struct.pack("<5i", 0, -1, 1, 0, 2) + b"x\0" + b"\0" * 8,
    ],
)
def test_invalid_headers(tmp_path, contents):
    directory = tmp_path / "1_load"
    directory.mkdir()
    path = directory / "p.set"
    path.write_bytes(contents)
    assert set_appears_valid(path) is False
    dataset = ObservationData(tmp_path, "p")
    assert dataset.get_all_steps()["participants"] == []
    assert dataset.get_steps() == []
    assert set(scan_dataset(tmp_path)) == {("stats", str(tmp_path), "")}


def test_hidden_paths_are_ignored_everywhere(tmp_path):
    write_eeglab(tmp_path / "1_load", stem="p")
    write_eeglab(tmp_path / "1_load", stem="._p")
    write_eeglab(tmp_path / ".hidden" / "2_clean", stem="p")
    (tmp_path / "1_load" / "directory.set").mkdir()
    dataset = ObservationData(tmp_path, "p")
    assert dataset.get_all_steps()["participants"] == [("p", 1)]
    assert [number for number, _ in dataset.get_steps()] == [1]
    assert ObservationData(tmp_path, "._p").get_steps() == []
    snapshot = scan_dataset(tmp_path)
    assert len(snapshot) == 5
    assert {key[2] for key in snapshot} == {"", "1_load/p.set"}


def test_missing_file_does_not_appear_valid(tmp_path):
    assert set_appears_valid(tmp_path / "missing.set") is False
