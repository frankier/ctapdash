import os
import warnings


from mne.io import read_raw_eeglab, read_epochs_eeglab
from mne.io.eeglab.eeglab import _check_eeglab_fname
import pickle

from ctapdash.io.eeglab.readers import (
    CtapEpochEEGLAB,
    CtapRawEEGLAB,
    sample_file_signature,
)
from ctapdash.io.paths import metadata_file_path
from ctapdash.io.utils import atomic_write


_METADATA_CACHE_HEADER = b"CTAPDASH metadata 1\n"


def cached_path(path):
    return metadata_file_path(path)


def _read_sample_source(stream):
    if stream.read(len(_METADATA_CACHE_HEADER)) != _METADATA_CACHE_HEADER:
        raise ValueError("Outdated EEGLAB metadata cache")
    return pickle.load(stream)


def _load_metadata(path):
    with open(path, "rb") as stream:
        _read_sample_source(stream)
        return pickle.load(stream)


def metadata_cache_current(path, set_path):
    # Read only the sample dependency during registration, without loading MNE data.
    try:
        with path.open("rb") as stream:
            data_name, sample_path, signature = _read_sample_source(stream)
        with warnings.catch_warnings(action="ignore"):
            resolved = _check_eeglab_fname(os.fspath(set_path), data_name)
        return (
            os.path.abspath(resolved) == sample_path
            and sample_file_signature(resolved) == signature
        )
    except OSError, ValueError, TypeError, EOFError, pickle.UnpicklingError:
        return False


def write_metadata_cache(path, eeg):
    with atomic_write(path, overwrite=True) as staging:
        with staging.open("wb") as stream:
            stream.write(_METADATA_CACHE_HEADER)
            pickle.dump(eeg.sample_source, stream)
            pickle.dump(eeg, stream)


def is_epoched(path):
    import pymatreader

    return len(pymatreader.read_mat(path, "epoch").get("epoch", ())) > 0


def _try_cache(path, set_path, base_stat, warm=False, force_cache=False):
    if path.exists():
        invalid = False
        cached_raw_stat = os.stat(path)
        if (
            cached_raw_stat.st_mtime_ns >= base_stat.st_mtime_ns
            and metadata_cache_current(path, set_path)
        ):
            try:
                result = _load_metadata(path)
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


def read_eeglab(
    path, use_cache=True, warm=False, force_cache=False, mmap=True, *, validated=False
):
    # Registered callers have already checked freshness and awaited readiness.
    invalid = False
    if validated:
        if not mmap or not use_cache:
            raise ValueError("validated loading requires the metadata cache")
        cached = cached_path(path)
        try:
            result = _load_metadata(cached)
            if hasattr(result, "raw_montage"):
                return result
        except Exception:
            pass
        invalid = True
        cached.unlink(missing_ok=True)
    if not mmap and (use_cache or force_cache or warm):
        raise ValueError("Cache is not implemented for mmap=False")
    if not invalid:
        base_stat = os.stat(path) if use_cache or force_cache else None
        cache_path = cached_path(path)
        if use_cache or force_cache:
            cached = _try_cache(
                cache_path, path, base_stat, warm=warm, force_cache=force_cache
            )
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
            write_metadata_cache(cached_path(path), eeg)
            if warm:
                return True
            return eeg
        else:
            if is_epoched(path):
                eeg = read_epochs_eeglab(path)
            else:
                eeg = read_raw_eeglab(path, preload=False)
            return eeg
