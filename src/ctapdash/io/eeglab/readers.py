"""Metadata-only EEGLAB readers for ctapdash's external sample maps."""

import os
import warnings
from copy import copy

import numpy as np
from mne import BaseEpochs, read_events
from mne.io import BaseRaw
from mne._fiff.pick import _PICK_TYPES_KEYS
from mne.io.eeglab.eeglab import (
    _bunchify,
    _check_boundary,
    _check_eeglab_fname,
    _check_latencies,
    _check_load_mat,
    _dol_to_lod,
    _get_info as _mne_get_info,
    _read_annotations_eeglab,
    _set_dig_montage_in_init,
)

from .mmap import mmap_eeglab


def _external_data_fname(input_fname, eeg, reader_name):
    if isinstance(eeg.data, str):
        data_fname = _check_eeglab_fname(input_fname, eeg.data)
        if data_fname.lower().endswith(".fdt"):
            return data_fname
    raise ValueError(f"{reader_name} requires data stored in an external .fdt file")


def _get_info(eeg, *, eog, montage_units):
    """Keep EEGLAB type labels separately from MNE's restricted channel types."""
    eeg = copy(eeg)
    chanlocs = eeg.chanlocs
    if isinstance(chanlocs, dict):
        chanlocs = [chanlocs] if eeg.nbchan == 1 else _dol_to_lod(chanlocs)
    raw_ch_types = []
    mne_chanlocs = []
    for chanloc in chanlocs:
        chanloc = chanloc.copy()
        label = chanloc.get("type")
        label = label.strip() if isinstance(label, str) else ""
        raw_ch_types.append(label or "EEG")
        ch_type = label.lower()
        chanloc["type"] = ch_type if ch_type in _PICK_TYPES_KEYS else "eeg"
        mne_chanlocs.append(chanloc)
    # MNE wraps single-channel metadata itself.
    eeg.chanlocs = np.asarray(mne_chanlocs, dtype=object)
    info, montage, update_ch_names = _mne_get_info(
        eeg, eog=eog, montage_units=montage_units
    )
    return info, raw_ch_types or ["EEG"] * eeg.nbchan, montage, update_ch_names


class CtapRawEEGLAB(BaseRaw):
    """Continuous EEGLAB metadata; access external samples through ``mmap()``."""

    def __init__(self, input_fname, eog=(), preload=False, *,
                 uint16_codec=None, montage_units="auto", verbose=None):
        if preload:
            raise ValueError("Preload has been disabled. Use .mmap(...)")
        input_fname = os.fspath(input_fname)
        eeg = _check_load_mat(input_fname, uint16_codec, preload=False)
        if eeg.trials != 1:
            raise ValueError("Raw files must contain exactly one trial")
        data_fname = _external_data_fname(input_fname, eeg, type(self).__name__)
        info, self.raw_ch_types, montage, _ = _get_info(
            eeg, eog=eog, montage_units=montage_units
        )
        super().__init__(info, preload=False, filenames=[data_fname],
                         last_samps=[eeg.pnts - 1], orig_format="double",
                         verbose=verbose)
        annotations = _read_annotations_eeglab(eeg)
        self.set_annotations(annotations)
        _check_boundary(annotations, None)
        _set_dig_montage_in_init(self, montage)
        _check_latencies(np.round(annotations.onset * info["sfreq"]))

    def mmap(self, *, return_xarray=False, data_fname=None, ctapdash_order=False):
        return mmap_eeglab(self, return_xarray=return_xarray, data_fname=data_fname,
                           ctapdash_order=ctapdash_order)

    def __reduce__(self):
        return _restore_mmap_eeglab, (type(self),), _mmap_eeglab_state(self)

    def _read_segment_file(self, *args, **kwargs):
        raise RuntimeError("Normal MNE data loading is disabled; use CtapRawEEGLAB.mmap()")


class CtapEpochEEGLAB(BaseEpochs):
    """
    Metadata-only EEGLAB epochs backed by an external ``.fdt`` file with support for arbitrary channel types.
    """

    def __init__(
        self,
        input_fname,
        events=None,
        event_id=None,
        tmin=0,
        baseline=None,
        reject=None,
        flat=None,
        reject_tmin=None,
        reject_tmax=None,
        eog=(),
        uint16_codec=None,
        montage_units="auto",
        verbose=None,
    ):
        unsupported = {
            "tmin": tmin != 0,
            "baseline": baseline is not None,
            "reject": reject is not None,
            "flat": flat is not None,
            "reject_tmin": reject_tmin is not None,
            "reject_tmax": reject_tmax is not None,
        }
        unsupported = [name for name, supplied in unsupported.items() if supplied]
        if unsupported:
            raise NotImplementedError(
                "CtapEpochEEGLAB does not support " + ", ".join(unsupported)
            )

        input_fname = os.fspath(input_fname)
        eeg = _check_load_mat(input_fname, uint16_codec, preload=False)
        if eeg.trials <= 1:
            raise ValueError(
                "The file does not contain epochs (trials must be greater than 1)"
            )
        _external_data_fname(input_fname, eeg, type(self).__name__)

        if not (
            (events is None and event_id is None)
            or (events is not None and event_id is not None)
        ):
            raise ValueError("Both `events` and `event_id` must be None or not None")

        if events is None:
            events, event_id = self._events_from_eeglab(eeg)
        elif isinstance(events, str | os.PathLike):
            events = read_events(events)

        info, raw_ch_types, eeg_montage, _ = _get_info(
            eeg, eog=eog, montage_units=montage_units
        )
        self.raw_ch_types = raw_ch_types
        assert events is not None
        assert event_id is not None
        BaseEpochs.__init__(
            self,
            info,
            None,
            events,
            event_id,
            eeg.xmin,
            eeg.xmax,
            baseline=None,
            filename=input_fname,
            verbose=verbose,
        )
        self._bad_dropped = True
        _set_dig_montage_in_init(self, eeg_montage)

    @staticmethod
    def _events_from_eeglab(eeg):
        """Construct MNE events without touching the external data file."""
        epochs = _bunchify(eeg.get("epoch", []))
        eeg_events = _bunchify(eeg.get("event", []))
        if len(epochs) == 0 or len(eeg_events) == 0:
            warnings.warn(
                "The EEGLAB file contains no event information. All epochs "
                "will be assigned to a single 'unknown' event."
            )
            return (
                np.column_stack(
                    (
                        np.arange(eeg.trials),
                        np.zeros(eeg.trials, dtype=int),
                        np.ones(eeg.trials, dtype=int),
                    )
                ),
                {"unknown": 1},
            )

        event_names = []
        event_latencies = []
        unique_events = []
        event_index = 0
        multiple_events = False
        for epoch in epochs:
            if isinstance(epoch.eventtype, int | float):
                epoch.eventtype = str(epoch.eventtype)
            if isinstance(epoch.eventtype, str):
                event_type = epoch.eventtype
                event_index_increment = 1
            else:
                event_type = "/".join(str(value) for value in epoch.eventtype)
                event_index_increment = len(epoch.eventtype)
                multiple_events = True
            event_names.append(event_type)
            event_latencies.append(eeg_events[event_index].latency - 1)
            event_index += event_index_increment
            if event_type not in unique_events:
                unique_events.append(event_type)

        if multiple_events:
            warnings.warn(
                "At least one epoch has multiple events. Only the latency of "
                "the first event will be retained."
            )

        event_id = {
            event_name: index + 1
            for index, event_name in enumerate(unique_events)
        }
        events = np.zeros((eeg.trials, 3), dtype=int)
        for index, (event_name, latency) in enumerate(
            zip(event_names, event_latencies, strict=True)
        ):
            previous_stimulus = 0
            if index > 0 and latency - event_latencies[index - 1] == 1:
                previous_stimulus = event_id[event_names[index - 1]]
            events[index] = latency, previous_stimulus, event_id[event_name]
        return events, event_id

    def mmap(self, *, return_xarray=False, data_fname=None, ctapdash_order=False):
        return mmap_eeglab(self, return_xarray=return_xarray, data_fname=data_fname, ctapdash_order=ctapdash_order)

    def __reduce__(self):
        return _restore_mmap_eeglab, (type(self),), _mmap_eeglab_state(self)

    def _data_disabled(self):
        raise RuntimeError(
            "Normal MNE data loading is disabled; use CtapEpochEEGLAB.mmap()"
        )

    def get_data(self, *args, **kwargs):
        self._data_disabled()

    def _get_data(self, *args, **kwargs):
        self._data_disabled()

    def _get_epoch_from_raw(self, *args, **kwargs):
        self._data_disabled()

    def load_data(self, *args, **kwargs):
        self._data_disabled()


def _mmap_eeglab_state(eeg):
    """Return metadata state without sample data or reconstructible duplicates."""
    state = eeg.__dict__.copy()
    state.pop("_data", None)
    state.pop("_raw", None)
    state.pop("_init_kwargs", None)

    # MNE can add an embedded-data cache to raw extras after construction.
    # Never allow such a cache to leak into the pickle.
    if "_raw_extras" in state:
        state["_raw_extras"] = [extra.copy() for extra in state["_raw_extras"]]
        for extra in state["_raw_extras"]:
            extra.pop("cached_data", None)
    return state


def _restore_mmap_eeglab(eeglab_class):
    """Allocate without invoking an EEGLAB constructor or reading its .set file."""
    return eeglab_class.__new__(eeglab_class)
