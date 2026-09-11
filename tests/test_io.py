import pickle

import numpy as np
import pytest
from tests.dummy_data import write_eeglab

from ctapdash.io.eeglab import CtapEpochEEGLAB, CtapRawEEGLAB, mmap_eeglab


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
    eeg = CtapRawEEGLAB(
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
    eeg = CtapEpochEEGLAB(set_path, verbose="error")

    data = eeg.mmap(return_xarray=True)

    assert eeg.preload is False
    assert eeg._data is None
    assert data.dims == ("ch", "epoch", "time")
    assert isinstance(data.data, np.memmap)
    expected = np.arange(16, dtype=np.float32).reshape(2, 2, 4).transpose(1, 0, 2)
    np.testing.assert_allclose(data, expected, rtol=1e-6, atol=0)
    with pytest.raises(RuntimeError, match=r"use CtapEpochEEGLAB\.mmap"):
        eeg.get_data()
    with pytest.raises(RuntimeError, match=r"use CtapEpochEEGLAB\.mmap"):
        eeg.load_data()
    with pytest.raises(RuntimeError, match=r"use CtapEpochEEGLAB\.mmap"):
        next(iter(eeg))


@pytest.mark.parametrize(
    ("n_epochs", "eeglab_class"),
    [(1, CtapRawEEGLAB), (2, CtapEpochEEGLAB)],
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

    monkeypatch.setattr("ctapdash.io.eeglab.readers._check_load_mat", fail_load_mat)
    restored = pickle.loads(payload)
    monkeypatch.undo()

    assert type(restored) is eeglab_class
    assert restored.preload is False
    assert getattr(restored, "_data", None) is None
    assert restored.ch_names == eeg.ch_names
    assert restored.info["sfreq"] == eeg.info["sfreq"]
    np.testing.assert_allclose(restored.mmap(), eeg.mmap(), rtol=0, atol=0)


@pytest.mark.parametrize("n_epochs,reader", [(1, CtapRawEEGLAB), (2, CtapEpochEEGLAB)])
@pytest.mark.parametrize("labels", [("Custom Sensor", " EOG "), ("", "EEG"), None])
def test_ctap_channel_types(tmp_path, monkeypatch, n_epochs, reader, labels):
    from scipy.io import loadmat, savemat
    from mne import BaseEpochs
    from mne.io import BaseRaw

    path = write_eeglab(tmp_path, n_epochs)
    metadata = loadmat(path, simplify_cells=True)
    metadata = {key: value for key, value in metadata.items() if not key.startswith("__")}
    if labels is None:
        metadata["chanlocs"] = np.empty(0)
    else:
        metadata["chanlocs"] = np.rec.fromarrays(
            [["A", "B"], labels, [0., 0.], [80., -80.], [60., 60.]],
            names=["labels", "type", "X", "Y", "Z"],
        )
    savemat(path, metadata)

    def fail_fromfile(*args, **kwargs):
        raise AssertionError("sample data must not be loaded")

    monkeypatch.setattr(np, "fromfile", fail_fromfile)
    eeg = reader(path, montage_units="mm", verbose="error")
    assert reader.__bases__ == ((BaseRaw,) if n_epochs == 1 else (BaseEpochs,))
    expected = [label.strip() or "EEG" for label in labels] if labels else ["EEG", "EEG"]
    assert eeg.raw_ch_types == expected
    assert pickle.loads(pickle.dumps(eeg)).raw_ch_types == expected
    assert eeg.get_channel_types() == (["eeg", "eog"] if labels and labels[1].strip() == "EOG" else ["eeg", "eeg"])
    assert isinstance(eeg.mmap(), np.memmap)
    if labels:
        assert eeg.get_montage() is not None


@pytest.mark.parametrize("n_epochs,reader", [(1, CtapRawEEGLAB), (2, CtapEpochEEGLAB)])
def test_ctap_rejects_embedded_samples(tmp_path, n_epochs, reader):
    from scipy.io import loadmat, savemat

    path = write_eeglab(tmp_path, n_epochs)
    metadata = loadmat(path, simplify_cells=True)
    metadata = {key: value for key, value in metadata.items() if not key.startswith("__")}
    metadata["data"] = np.zeros((2, 4, n_epochs))
    savemat(path, metadata)
    with pytest.raises(ValueError, match="external .fdt"):
        reader(path, verbose="error")
