import pickle

import numpy as np
import pytest
from tests.dummy_data import write_eeglab

from ctapdash.io.eeglab import MmapEpochEEGLAB, MmapRawEEGLAB, mmap_eeglab


@pytest.mark.parametrize("n_epochs", [1, 2])
def test_mmap_eeglab_matches_mne(tmp_path, n_epochs):
    import mne
    from mne.io.eeglab.eeglab import CAL

    set_path = write_eeglab(tmp_path, n_epochs)
    if n_epochs == 1:
        eeg = mne.io.read_raw_eeglab(set_path, preload=False, verbose="error")
    else:
        eeg = mne.read_epochs_eeglab(set_path, verbose="error")

    fdt_path = set_path.with_suffix(".fdt")
    original_bytes = fdt_path.read_bytes()
    data = mmap_eeglab(eeg)

    assert isinstance(data, np.memmap)
    assert data.dtype == np.float32
    expected = eeg.get_data()
    if n_epochs > 1:
        expected = expected.transpose(1, 0, 2)
    np.testing.assert_allclose(data, expected / CAL, rtol=1e-6, atol=0)
    assert fdt_path.read_bytes() == original_bytes


def test_mmap_eeglab_xarray(tmp_path):
    import mne

    eeg = mne.io.read_raw_eeglab(
        write_eeglab(tmp_path, 1), preload=False, verbose="error"
    )

    data = mmap_eeglab(eeg, return_xarray=True)

    assert data.dims == ("ch", "time")
    assert isinstance(data.data, np.memmap)
    assert data.ch.values.tolist() == eeg.ch_names
    np.testing.assert_array_equal(data.time, eeg.times)


def test_mmap_eeglab_epochs_xarray(tmp_path):
    import mne
    from mne.io.eeglab.eeglab import CAL

    eeg = mne.read_epochs_eeglab(
        write_eeglab(tmp_path, 2), verbose="error"
    )

    data = mmap_eeglab(eeg, return_xarray=True)

    assert data.dims == ("ch", "epoch", "time")
    assert isinstance(data.data, np.memmap)
    np.testing.assert_array_equal(data.epoch, eeg.selection)
    np.testing.assert_allclose(data, eeg.get_data().transpose(1, 0, 2) / CAL, rtol=1e-6, atol=0)


def test_mmap_raw_eeglab(tmp_path):
    eeg = MmapRawEEGLAB(
        write_eeglab(tmp_path, 1), preload=False, verbose="error"
    )

    data = eeg.mmap()

    assert isinstance(data, np.memmap)
    np.testing.assert_allclose(data, mmap_eeglab(eeg), rtol=1e-6, atol=0)


def test_mmap_epoch_eeglab(tmp_path, monkeypatch):
    set_path = write_eeglab(tmp_path, 2)

    def fail_fromfile(*args, **kwargs):
        raise AssertionError("the .fdt file was loaded during construction")

    monkeypatch.setattr(np, "fromfile", fail_fromfile)
    eeg = MmapEpochEEGLAB(set_path, verbose="error")

    data = eeg.mmap(return_xarray=True)

    assert eeg.preload is False
    assert eeg._data is None
    assert data.dims == ("ch", "epoch", "time")
    assert isinstance(data.data, np.memmap)
    expected = np.arange(16, dtype=np.float32).reshape(2, 2, 4).transpose(1, 0, 2)
    np.testing.assert_allclose(data, expected, rtol=1e-6, atol=0)
    with pytest.raises(RuntimeError, match=r"use MmapEpochEEGLAB\.mmap"):
        eeg.get_data()
    with pytest.raises(RuntimeError, match=r"use MmapEpochEEGLAB\.mmap"):
        eeg.load_data()
    with pytest.raises(RuntimeError, match=r"use MmapEpochEEGLAB\.mmap"):
        next(iter(eeg))


@pytest.mark.parametrize(
    ("n_epochs", "eeglab_class"),
    [(1, MmapRawEEGLAB), (2, MmapEpochEEGLAB)],
)
def test_mmap_eeglab_pickle_is_metadata_only(
    tmp_path, monkeypatch, n_epochs, eeglab_class
):
    eeg = eeglab_class(write_eeglab(tmp_path, n_epochs), verbose="error")
    reduction = eeg.__reduce__()
    payload = pickle.dumps(eeg)

    assert len(reduction) == 3
    assert isinstance(reduction[2], dict)
    assert not {"_data", "_raw", "_init_kwargs"} & reduction[2].keys()

    def fail_load_mat(*args, **kwargs):
        raise AssertionError("the .set file was read during unpickling")

    monkeypatch.setattr("ctapdash.io.eeglab._check_load_mat", fail_load_mat)
    restored = pickle.loads(payload)
    monkeypatch.undo()

    assert type(restored) is eeglab_class
    assert restored.preload is False
    assert getattr(restored, "_data", None) is None
    assert restored.ch_names == eeg.ch_names
    assert restored.info["sfreq"] == eeg.info["sfreq"]
    np.testing.assert_allclose(restored.mmap(), eeg.mmap(), rtol=0, atol=0)
