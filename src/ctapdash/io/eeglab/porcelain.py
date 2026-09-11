import os
import warnings


from mne.io import read_raw_eeglab, read_epochs_eeglab
import pickle

from ctapdash.io.eeglab.readers import CtapEpochEEGLAB, CtapRawEEGLAB
from ctapdash.io.paths import metadata_file_path


def cached_path(path):
    return metadata_file_path(path)


def pickle_load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def is_epoched(path):
    import pymatreader
    return len(pymatreader.read_mat(path, "epoch").get("epoch", ())) > 0


def _try_cache(load_func, path, base_stat, warm=False, force_cache=False):
    if path.exists():
        invalid = False
        cached_raw_stat = os.stat(path)
        mtime_valid = cached_raw_stat.st_mtime >= base_stat.st_mtime
        if mtime_valid:
            try:
                result = load_func(path)
            except Exception as err:
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



def read_eeglab(path, use_cache=True, warm=False, force_cache=False, mmap=True, *, validated=False):
    # Registered callers have already checked freshness and awaited readiness.
    if validated:
        if not mmap or not use_cache:
            raise ValueError("validated loading requires the metadata cache")
        return pickle_load(cached_path(path))
    base_stat = os.stat(path) if use_cache or force_cache else None
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
                eeg = CtapEpochEEGLAB(path)
            else:
                eeg = CtapRawEEGLAB(path)
            if not use_cache:
                return eeg
            from ctapdash.io.utils import atomic_write
            with atomic_write(cached_path(path), overwrite=True) as staging:
                with staging.open("wb") as f:
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