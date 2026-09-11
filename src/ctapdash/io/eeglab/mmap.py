import os
import xarray as xr


import numpy as np
from mne import BaseEpochs
from mne.io import BaseRaw
from mne.io.eeglab.eeglab import (
    CAL,
    _check_eeglab_fname,
    _check_load_mat,
)


def _eeglab_memmap(data_fname, shape, order):
    expected_size = np.prod(shape, dtype=np.int64) * np.dtype("<f4").itemsize
    actual_size = os.path.getsize(data_fname)
    if actual_size != expected_size:
        raise ValueError(
            f"EEGLAB data file has {actual_size} bytes; expected {expected_size} "
            f"bytes for shape {shape}."
        )

    return np.memmap(data_fname, dtype="<f4", mode="r", shape=shape, order=order)


def mmap_eeglab(eeg, *, return_xarray=False, data_fname=None, ctapdash_order=False):
    """Return float32 data from an MNE EEGLAB object.

    External EEGLAB ``.fdt`` data is returned in its native stored units so the
    result can remain a direct memory map. Embedded ``.set`` data cannot be
    memory-mapped and is returned as an ordinary float32 array in the same
    units.

    Parameters
    ----------
    eeg : mne.io.BaseRaw | mne.BaseEpochs
        An MNE object read from an EEGLAB ``.set`` file. The object's channels,
        time range, and epochs must not have been selected after reading.
    return_xarray : bool
        Return an ``xarray.DataArray`` instead of a NumPy array. Raw data uses
        the same ``("ch", "time")`` dimensions and coordinates as
        :func:`ctapdash.pyramid.convert_to_xarray`. Epoch data additionally
        has an ``"epoch"`` dimension between ``"ch"`` and ``"time"``.

    Returns
    -------
    numpy.ndarray | xarray.DataArray
        Float32 data with shape ``(channel, time)`` for raw data or
        ``(channel, epoch, time)`` for epochs. For external data the NumPy
        object (and the xarray object's backing data) is a ``numpy.memmap``.
    """
    if not isinstance(eeg, BaseRaw | BaseEpochs):
        raise TypeError("eeg must be an MNE Raw or Epochs object")

    if isinstance(eeg, BaseRaw):
        order = "C" if ctapdash_order else "F"
        if len(eeg.filenames) != 1:
            raise ValueError("An EEGLAB Raw object must refer to exactly one file")
        if data_fname is None:
            data_fname = os.fspath(eeg.filenames[0])
            if not data_fname.lower().endswith((".fdt", ".set")):
                raise ValueError("The Raw object was not read from an EEGLAB file")
            use_mmap = data_fname.lower().endswith(".fdt")
        else:
            use_mmap = True

        if use_mmap:
            orig_nchan = eeg._raw_extras[0]["orig_nchan"]
            if orig_nchan != len(eeg.ch_names):
                raise ValueError("mmap_eeglab does not support picked Raw channels")
            n_total = os.path.getsize(data_fname) // (orig_nchan * 4)
            data = _eeglab_memmap(
                data_fname, (orig_nchan, n_total), order=order
            )
            data = data[:, eeg.first_samp : eeg.last_samp + 1]
        else:
            data = np.asarray(eeg.get_data(), dtype=np.float32)
            data = data / CAL
    else:
        set_fname = getattr(eeg, "filename", None)
        if set_fname is None or not os.fspath(set_fname).lower().endswith(".set"):
            raise ValueError("The Epochs object was not read from an EEGLAB .set file")

        eeglab = _check_load_mat(os.fspath(set_fname), None, preload=False)
        use_mmap = False
        if data_fname is None:
            if isinstance(eeglab.data, str):
                data_fname = _check_eeglab_fname(os.fspath(set_fname), eeglab.data)
                use_mmap = True
        else:
            use_mmap = True
        if use_mmap:
            if eeglab.nbchan != len(eeg.ch_names):
                raise ValueError("mmap_eeglab does not support picked Epochs channels")
            selection = np.asarray(eeg.selection)
            if not np.array_equal(selection, np.arange(eeglab.trials)):
                raise ValueError("mmap_eeglab does not support selected or dropped epochs")
            if len(eeg.times) != eeglab.pnts:
                raise ValueError("mmap_eeglab does not support cropped epochs")
            if ctapdash_order:
                data = _eeglab_memmap(
                    data_fname,
                    (eeglab.nbchan, eeglab.trials, eeglab.pnts),
                    order="C",
                )
            else:
                data = _eeglab_memmap(
                    data_fname,
                    (eeglab.nbchan, eeglab.pnts, eeglab.trials),
                    order="F",
                ).transpose(0, 2, 1)
        else:
            data = np.asarray(eeg.get_data(), dtype=np.float32)
            data = data / CAL

    if return_xarray:
        if isinstance(eeg, BaseRaw):
            dims = ("ch", "time")
            coords = (eeg.ch_names, eeg.times)
        else:
            dims = ("ch", "epoch", "time")
            coords = (eeg.ch_names, np.asarray(eeg.selection), eeg.times)
        return xr.DataArray(data, coords=coords, dims=dims)
    return data