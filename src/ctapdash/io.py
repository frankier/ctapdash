import os
import warnings
import re
import xarray as xr


from natsort import natsorted
import numpy as np
from mne import BaseEpochs, read_epochs, read_events
from mne.io import BaseRaw, read_raw_eeglab, read_epochs_eeglab, read_raw_fif
from mne.io.eeglab.eeglab import (
    CAL,
    EpochsEEGLAB,
    RawEEGLAB,
    _bunchify,
    _check_eeglab_fname,
    _check_load_mat,
    _get_info,
    _set_dig_montage_in_init,
)
import pickle
from scipy import stats


SCALP_REGEX = re.compile("(?P<stem>[^-]+)-badChan-scalp.png")
CH_REGEX = re.compile("(?P<stem>.+)-chs(?P<ch_start>[0-9]+)-(?P<ch_end>[0-9]+).png")


def is_epoched(path):
    import pymatreader
    return len(pymatreader.read_mat(path, "epoch").get("epoch", ())) > 0


def _eeglab_memmap(data_fname, shape, order):
    expected_size = np.prod(shape, dtype=np.int64) * np.dtype("<f4").itemsize
    actual_size = os.path.getsize(data_fname)
    if actual_size != expected_size:
        raise ValueError(
            f"EEGLAB data file has {actual_size} bytes; expected {expected_size} "
            f"bytes for shape {shape}."
        )

    return np.memmap(data_fname, dtype="<f4", mode="r", shape=shape, order=order)


def mmap_eeglab(eeg, *, return_xarray=False):
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
        has a leading ``"epoch"`` dimension.

    Returns
    -------
    numpy.ndarray | xarray.DataArray
        Float32 data with shape ``(channel, time)`` for raw data or
        ``(epoch, channel, time)`` for epochs. For external data the NumPy
        object (and the xarray object's backing data) is a ``numpy.memmap``.
    """
    if not isinstance(eeg, BaseRaw | BaseEpochs):
        raise TypeError("eeg must be an MNE Raw or Epochs object")

    if isinstance(eeg, BaseRaw):
        if len(eeg.filenames) != 1:
            raise ValueError("An EEGLAB Raw object must refer to exactly one file")
        data_fname = os.fspath(eeg.filenames[0])
        if not data_fname.lower().endswith((".fdt", ".set")):
            raise ValueError("The Raw object was not read from an EEGLAB file")

        if data_fname.lower().endswith(".fdt"):
            orig_nchan = eeg._raw_extras[0]["orig_nchan"]
            if orig_nchan != len(eeg.ch_names):
                raise ValueError("mmap_eeglab does not support picked Raw channels")
            n_total = os.path.getsize(data_fname) // (orig_nchan * 4)
            data = _eeglab_memmap(
                data_fname, (orig_nchan, n_total), order="F"
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
        if isinstance(eeglab.data, str):
            data_fname = _check_eeglab_fname(os.fspath(set_fname), eeglab.data)
            if eeglab.nbchan != len(eeg.ch_names):
                raise ValueError("mmap_eeglab does not support picked Epochs channels")
            selection = np.asarray(eeg.selection)
            if not np.array_equal(selection, np.arange(eeglab.trials)):
                raise ValueError("mmap_eeglab does not support selected or dropped epochs")
            if len(eeg.times) != eeglab.pnts:
                raise ValueError("mmap_eeglab does not support cropped epochs")
            data = _eeglab_memmap(
                data_fname,
                (eeglab.nbchan, eeglab.pnts, eeglab.trials),
                order="F",
            ).transpose(2, 0, 1)
        else:
            data = np.asarray(eeg.get_data(), dtype=np.float32)
            data = data / CAL

    if return_xarray:
        if isinstance(eeg, BaseRaw):
            dims = ("ch", "time")
            coords = (eeg.ch_names, eeg.times)
        else:
            dims = ("epoch", "ch", "time")
            coords = (np.asarray(eeg.selection), eeg.ch_names, eeg.times)
        return xr.DataArray(data, coords=coords, dims=dims)
    return data


class MmapRawEEGLAB(RawEEGLAB):
    def __init__(
        self,
        input_fname,
        eog=(),
        preload=False,
        *,
        uint16_codec=None,
        montage_units="auto",
        verbose=None,
    ):
        if preload:
            raise ValueError("Preload has been disabled. Use .mmap(...)")
        super().__init__(
            input_fname,
            eog,
            preload,
            uint16_codec=uint16_codec,
            montage_units=montage_units,
            verbose=verbose,
        )

    def mmap(self, *, return_xarray=False):
        return mmap_eeglab(self, return_xarray=return_xarray)

    def __reduce__(self):
        return _restore_mmap_eeglab, (type(self),), _mmap_eeglab_state(self)

    def _read_segment_file(self, *args, **kwargs):
        raise ValueError(
            "This method has been disabled since it would load data into "
            "memory rather than using mmap"
        )


class MmapEpochEEGLAB(EpochsEEGLAB):
    """Metadata-only EEGLAB epochs backed by an external ``.fdt`` file."""

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
                "MmapEpochEEGLAB does not support " + ", ".join(unsupported)
            )

        input_fname = os.fspath(input_fname)
        eeg = _check_load_mat(input_fname, uint16_codec, preload=False)
        if eeg.trials <= 1:
            raise ValueError(
                "The file does not contain epochs (trials must be greater than 1)"
            )
        if not isinstance(eeg.data, str):
            raise ValueError(
                "MmapEpochEEGLAB requires data stored in an external .fdt file"
            )
        # Validate the referenced data file, but deliberately do not open it.
        _check_eeglab_fname(input_fname, eeg.data)

        if not (
            (events is None and event_id is None)
            or (events is not None and event_id is not None)
        ):
            raise ValueError("Both `events` and `event_id` must be None or not None")

        if events is None:
            events, event_id = self._events_from_eeglab(eeg)
        elif isinstance(events, str | os.PathLike):
            events = read_events(events)

        info, eeg_montage, _ = _get_info(
            eeg, eog=eog, montage_units=montage_units
        )
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

    def mmap(self, *, return_xarray=False):
        return mmap_eeglab(self, return_xarray=return_xarray)

    def __reduce__(self):
        return _restore_mmap_eeglab, (type(self),), _mmap_eeglab_state(self)

    def _data_disabled(self):
        raise RuntimeError(
            "Normal MNE data loading is disabled; use MmapEpochEEGLAB.mmap()"
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


def cached_path(path):
    return path.parent / ("." + path.stem + ".pkl")


def pickle_load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _try_cache(load_func, path, base_stat, warm=False, force_cache=False):
    if path.exists():
        invalid = False
        cached_raw_stat = os.stat(path)
        mtime_valid = cached_raw_stat.st_mtime >= base_stat.st_mtime
        if mtime_valid:
            try:
                result = load_func(path)
            except ValueError as err:
                if force_cache:
                    raise err
                invalid = True
            else:
                if warm:
                    return False
                else:
                    return result
        else:
            if force_cache:
                raise ValueError("Cache is outdated, but force_cache=True")
            invalid = True
        if invalid:
            path.unlink()
    return None


def read_eeglab(path, use_cache=True, warm=False, force_cache=False, mmap=True):
    base_stat = os.stat(path)
    cache_path = cached_path(path)
    if not mmap and (use_cache or force_cache or warm):
        raise ValueError("Cache is not implemented for mmap=False")
    if use_cache or force_cache:
        cached = _try_cache(pickle_load, cache_path, base_stat, warm=warm, force_cache=force_cache)
        if cached is not None:
            return cached
    if force_cache:
        raise ValueError("Wasn't able to load from cache when force_cache=True")
    with warnings.catch_warnings(action="ignore"):
        if mmap:
            if is_epoched(path):
                eeg = MmapEpochEEGLAB(path)
            else:
                eeg = MmapRawEEGLAB(path)
            if not use_cache:
                return eeg
            with open(cached_path(path), "wb") as f:
                pickle.dump(eeg, f)
            if warm:
                return True
            return eeg
        else:
            if is_epoched(path):
                eeg = read_epochs_eeglab(path)
            else:
                eeg = read_raw_eeglab(path, preload=False)
            return eeg


def collect_logs(source_path, participant):
    logs = []
    log_dir = source_path / "logs"
    for root, dirs, files in log_dir.walk():
        for file in files:
            if not file.startswith(participant):
                continue
            file_path = root / file
            rel_path = file_path.relative_to(source_path)
            logs.append(str(rel_path))
    return logs


def collect_qc(source_path, participant):
    qc = []
    qc_dir = source_path / "quality_control"
    for root, dirs, files in qc_dir.walk():
        for file in files:
            if not file.startswith(participant):
                continue
            file_path = root / file
            rel_path = file_path.relative_to(qc_dir)
            qc.append(rel_path)
    qc = natsorted(qc)
    return qc


def qc_to_tree(qcs):
    tree = {}
    for qc in qcs:
        tree.setdefault(qc.parts[0], []).append(qc)

    def group_channels(values):
        groups = {}
        rest = {}
        for value, path in values:
            match = SCALP_REGEX.match(value)
            if match:
                stem = match.group("stem")
                groups.setdefault(stem, {})["scalp"] = (value, path)
                continue
            match = CH_REGEX.match(value)
            if match:
                stem = match.group("stem")
                ch_start = int(match.group("ch_start"))
                ch_end = int(match.group("ch_end"))
                groups.setdefault(stem, {}).setdefault("chs", []).append((ch_start, ch_end, value, path))
                groups[stem]["chs"].sort()
                continue
            rest[value] = path
        return groups, rest

    def form_sets(peek_list):
        peek_dict = {}
        rest = {}
        for directory in peek_list:
            first_seg = directory.parts[1]
            if first_seg.startswith("set"):
                peek_dict.setdefault(directory.parts[1], []).append((directory.parts[-1], directory))
            else:
                sub_peek_dict, rest_dict = form_sets(directory)
                peek_dict.update(sub_peek_dict)
                rest.update(rest_dict)
        return peek_dict, rest

    new_tree = {}
    for root, peek_list in tree.items():
        peek_dict, rest = form_sets(peek_list)

        assert len(rest) == 0
        peek_dict = {k: group_channels(v) for k, v in peek_dict.items()}
        new_tree[root] = peek_dict

    return new_tree


def get_steps_for_participant(root_path, participant):
    steps = []
    for subdir in root_path.iterdir():
        if not subdir.name[0].isnumeric():
            continue
        path = subdir / (participant + ".set")
        if not path.exists():
            continue
        step_num = int(subdir.name.split("_", 1)[0])
        steps.append((step_num, subdir))
    steps = natsorted(steps)
    return steps


def collect_steps(root_path):
    from collections import Counter

    steps = []
    files = Counter()

    for subdir in root_path.iterdir():
        if subdir.name[0].isnumeric():
            step_num = int(subdir.name.split("_", 1)[0])
            steps.append((step_num, subdir))
            for filename in subdir.iterdir():
                if not filename.suffix == ".set":
                    continue
                files[filename.stem] += 1

    steps.sort()
    files = [(-count, filename) for filename, count in files.items()]
    files.sort()

    return {
        "participants": [(filename, -count) for count, filename in files],
        "steps": steps,
    }


class ObservationData:
    def __init__(self, source_path, participant=None, source=None):
        self.source_path = source_path
        self.participant = participant
        self.source = source

    @classmethod
    def from_source(cls, source, participant=None):
        from ctapdash.config import SETTINGS
        from pathlib import Path

        source_path = Path(SETTINGS.sources[source])
        return cls(source_path, participant, source)

    @classmethod
    def from_request(cls, request):
        source = request.query_params.get("source")
        if source is None:
            raise ValueError("Source must be specified in the request.")
        participant = request.query_params.get("participant")
        return cls.from_source(source, participant)

    @classmethod
    def from_bokeh_doc(cls, doc):
        def get_arg(name):
            args = doc.session_context.request.arguments
            val = args.get(name)
            if not val:
                return None
            return val[-1].decode()
        source = get_arg("source")
        participant = get_arg("participant")
        return cls.from_source(source, participant)

    def require_participant(self):
        if self.participant is None:
            raise ValueError("Participant must be specified for this operation.")

    def get_logs(self):
        self.require_participant()
        return collect_logs(self.source_path, self.participant)

    def get_qc(self):
        self.require_participant()
        return collect_qc(self.source_path, self.participant)

    def get_steps(self):
        self.require_participant()
        return get_steps_for_participant(self.source_path, self.participant)

    def get_all_steps(self):
        return collect_steps(self.source_path)
