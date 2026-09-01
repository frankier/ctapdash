import os
import warnings
import re

from natsort import natsorted
import numpy as np
from mne import read_epochs, BaseEpochs
from mne.io import BaseRaw, read_raw_eeglab, read_epochs_eeglab, read_raw_fif
from mne.io.eeglab.eeglab import CAL, _check_eeglab_fname, _check_load_mat
from scipy import stats


SCALP_REGEX = re.compile("(?P<stem>[^-]+)-badChan-scalp.png")
CH_REGEX = re.compile("(?P<stem>.+)-chs(?P<ch_start>[0-9]+)-(?P<ch_end>[0-9]+).png")


def _scaled_eeglab_memmap(data_fname, shape, order):
    """Map an EEGLAB float file and apply MNE's volts calibration."""
    expected_size = np.prod(shape, dtype=np.int64) * np.dtype("<f4").itemsize
    actual_size = os.path.getsize(data_fname)
    if actual_size != expected_size:
        raise ValueError(
            f"EEGLAB data file has {actual_size} bytes; expected {expected_size} "
            f"bytes for shape {shape}."
        )

    return np.memmap(data_fname, dtype="<f4", shape=shape, order=order)


def mmap_eeglab(eeg, return_xarray=False):
    """Return float32 data from an MNE EEGLAB object.

    External EEGLAB ``.fdt`` data is memory-mapped copy-on-write and calibrated
    to volts, matching :meth:`mne.io.BaseRaw.get_data`. Embedded ``.set`` data
    cannot be memory-mapped and is returned as an ordinary float32 array.

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
            data = _scaled_eeglab_memmap(
                data_fname, (orig_nchan, n_total), order="F"
            )
            data = data[:, eeg.first_samp : eeg.last_samp + 1]
        else:
            data = np.asarray(eeg.get_data(), dtype=np.float32)
            data = data / CAL
        dims = ("ch", "time")
        coords = (eeg.ch_names, eeg.times)
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
            data = _scaled_eeglab_memmap(
                data_fname,
                (eeglab.nbchan, eeglab.pnts, eeglab.trials),
                order="F",
            ).transpose(2, 0, 1)
        else:
            data = np.asarray(eeg.get_data(), dtype=np.float32)
            data = data / CAL
        dims = ("epoch", "ch", "time")
        coords = (np.asarray(eeg.selection), eeg.ch_names, eeg.times)

    if return_xarray:
        import xarray as xr

        return xr.DataArray(data, coords=coords, dims=dims)
    return data


def cached_path(path, raw=True):
    if raw:
        return path.parent / ("." + path.stem + "-raw.fif")
    else:
        return path.parent / ("." + path.stem + "-epo.fif")


def _try_cache(load_func, path, base_stat, preload=False, warm=False, force_cache=False):
    if path.exists():
        invalid = False
        cached_raw_stat = os.stat(path)
        mtime_valid = cached_raw_stat.st_mtime >= base_stat.st_mtime
        if mtime_valid:
            try:
                result = load_func(path, preload=preload)
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


def read_eeglab(path, use_cache=True, warm=False, force_cache=False):
    base_stat = os.stat(path)
    cached_raw_path = cached_path(path, raw=True)
    cached_epo_path = cached_path(path, raw=False)
    if use_cache or force_cache:
        raw_cached = _try_cache(read_raw_fif, cached_raw_path, base_stat, preload=False, warm=warm, force_cache=force_cache)
        if raw_cached is not None:
            return raw_cached
        epo_cached = _try_cache(read_epochs, cached_epo_path, base_stat, preload=False, warm=warm, force_cache=force_cache)
        if epo_cached is not None:
            return epo_cached
    if force_cache:
        raise ValueError("Wasn't able to load from cache when force_cache=True")
    with warnings.catch_warnings(action="ignore"):
        try:
            base_eeg = read_epochs_eeglab(path)
        except ValueError:
            base_eeg = read_raw_eeglab(path, preload=False)
        if not use_cache:
            return base_eeg
        if isinstance(base_eeg, BaseEpochs):
            base_eeg.save(cached_epo_path)
        else:
            base_eeg.save(cached_raw_path)
        if warm:
            return True
        return read_eeglab(path, force_cache=True)


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
