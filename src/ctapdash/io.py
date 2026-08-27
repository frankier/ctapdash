import os
import warnings

import numpy as np
from mne import read_epochs, BaseEpochs
from mne.io import BaseRaw, read_raw_eeglab, read_epochs_eeglab, read_raw_fif
from scipy import stats


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
