import os
import warnings

import numpy as np
import xarray as xr
from mne import read_epochs, BaseEpochs
from mne.io import BaseRaw, read_raw_eeglab, read_epochs_eeglab, read_raw_fif
from scipy import stats


DESCRIPTIVE_STATISTICS = (
    "nobs",
    "min",
    "max",
    "mean",
    "variance",
    "skewness",
    "kurtosis",
)


def _channel_samples(instance):
    if isinstance(instance, BaseRaw):
        return instance.get_data()
    if isinstance(instance, BaseEpochs):
        data = instance.get_data(copy=False)
        return data.swapaxes(0, 1).reshape(len(instance.ch_names), -1)
    raise TypeError(
        "describe() inputs must be MNE Raw or Epochs instances, "
        f"got {type(instance).__name__}"
    )


def _channel_values(value, n_channels):
    values = np.asarray(value)
    if values.ndim == 0:
        return np.full(n_channels, values.item())
    return values


def describe(*instances):
    """Return SciPy descriptive statistics for each MNE channel.

    Raw observations are time samples. Epochs observations combine every epoch
    and time sample for a channel. Multiple inputs are indexed by ``recording``;
    xarray aligns their channel-name union and fills absent channels with NaN.
    """
    if not instances:
        raise ValueError("describe() requires at least one Raw or Epochs instance")

    summaries = []
    for instance in instances:
        samples = _channel_samples(instance)
        result = stats.describe(samples, axis=-1, nan_policy="omit")
        values = (
            result.nobs,
            result.minmax[0],
            result.minmax[1],
            result.mean,
            result.variance,
            result.skewness,
            result.kurtosis,
        )
        summaries.append(
            xr.Dataset(
                {
                    name: ("channel", _channel_values(value, len(instance.ch_names)))
                    for name, value in zip(DESCRIPTIVE_STATISTICS, values, strict=True)
                },
                coords={"channel": instance.ch_names},
            )
        )

    return xr.concat(
        summaries,
        dim=xr.IndexVariable("recording", np.arange(len(summaries))),
        join="outer",
    )


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
