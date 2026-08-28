import os
import warnings
import re

from natsort import natsorted
import numpy as np
from mne import read_epochs, BaseEpochs
from mne.io import BaseRaw, read_raw_eeglab, read_epochs_eeglab, read_raw_fif
from scipy import stats


SCALP_REGEX = re.compile("(?P<stem>[^-]+)-badChan-scalp.png")
CH_REGEX = re.compile("(?P<stem>.+)-chs(?P<ch_start>[0-9]+)-(?P<ch_end>[0-9]+).png")


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
