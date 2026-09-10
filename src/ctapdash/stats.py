from pathlib import Path

import numpy as np
from mne import BaseEpochs
from mne.io import BaseRaw
import xarray as xr
import numba
from collections import namedtuple
from math import isnan
from ctapdash.io.eeglab import MmapEpochEEGLAB, MmapRawEEGLAB
from ctapdash.io.paths import DatasetPaths, stats_file_path
from ctapdash.io.recording import RecordingData
from ctapdash.io.xarray import save_xarray


DESCRIPTIVE_STATISTICS = (
    "nobs",
    "min",
    "max",
    "mean",
    "variance",
    "skewness",
    "kurtosis",
)


def _channel_samples(instance: BaseRaw | BaseEpochs) -> np.ndarray:
    if isinstance(instance, MmapRawEEGLAB):
        return instance.mmap()
    if isinstance(instance, MmapEpochEEGLAB):
        data = instance.mmap()
        return data.reshape(len(instance.ch_names), -1)
    if isinstance(instance, BaseRaw):
        return instance.get_data()
    if isinstance(instance, BaseEpochs):
        data = instance.get_data(copy=False)
        return data.swapaxes(0, 1).reshape(len(instance.ch_names), -1)


def _channel_values(value, n_channels: int) -> np.ndarray:
    values = np.asarray(value)
    if values.ndim == 0:
        return np.full(n_channels, values.item())
    return values


def describe_mne(instance: BaseRaw | BaseEpochs, samples=None) -> xr.Dataset:
    if samples is None:
        samples = _channel_samples(instance)
    result = describe(samples, axis=-1)
    values = (
        result.nobs,
        result.minmax[0],
        result.minmax[1],
        result.mean,
        result.variance,
        result.skewness,
        result.kurtosis,
    )
    return xr.Dataset(
        {
            name: ("channel", _channel_values(value, len(instance.ch_names)))
            for name, value in zip(DESCRIPTIVE_STATISTICS, values, strict=True)
        },
        coords={"channel": instance.ch_names},
    )


def describe_dataset(dataset_dir: Path, *, recordings=None, metadata_validated=False) -> xr.Dataset:
    """Return SciPy descriptive statistics for each MNE channel.

    Raw observations are time samples. Epochs observations combine every epoch
    and time sample for a channel. Multiple inputs are indexed by ``recording``;
    xarray aligns their channel-name union and fills absent channels with NaN.
    """
    summaries = []
    recording_ids = []
    paths = DatasetPaths(dataset_dir)
    if recordings is None:
        recordings = [p for p in paths.root.glob("*/*.set")
                      if p.parent.name[0].isnumeric()]
    for path in recordings:
        recording = RecordingData(paths.recording(path), metadata_validated)
        path = recording.paths.set
        step_num = int(path.parent.name.split("_", 1)[0])
        instance = recording.read_metadata()
        samples = recording.open_transpose()
        if samples.ndim == 3:
            samples = samples.reshape(len(instance.ch_names), -1)
        result = describe(samples, axis=-1)
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
        recording_ids.append((step_num, path.stem))

    if not summaries:
        return xr.Dataset(coords={"recording": [], "step": ("recording", []),
                                  "participant": ("recording", [])})
    return xr.concat(
        summaries,
        dim=xr.IndexVariable("recording", np.arange(len(summaries))),
        join="outer",
    ).assign_coords(
        step=("recording", [step_num for step_num, _ in recording_ids]),
        participant=("recording", [participant for _, participant in recording_ids]),
    )


def precompute_descriptive_statistics(dataset_dir, *, recordings=None, metadata_validated=False):
    from ctapdash.io.utils import atomic_write

    dest = stats_file_path(dataset_dir)
    stats = describe_dataset(dataset_dir, recordings=recordings,
                             metadata_validated=metadata_validated)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(dest, dir=True, overwrite=True) as staging:
        # save_xarray creates its own destination.
        staging.rmdir()
        save_xarray(stats, staging)


@numba.njit
def single_scan_stats(a):
    n = 0
    mn = float("inf")
    mx = float("-inf")
    tot = 0
    for x in a:
        n += 1
        tot += x
        if x < mn:
            mn = x
        if x > mx:
            mx = x
    mean = tot / n if n > 0 else float("nan")
    return (n, mn, mx, mean)


@numba.njit
def moments(a, mean, n, orders, results):
    for x in a:
        demean = x - mean
        for i, order in enumerate(orders):
            results[i] += demean ** order
    results /= n


DescribeResult = namedtuple('DescribeResult',
                            ('nobs', 'minmax', 'mean', 'variance', 'skewness',
                             'kurtosis'))


@numba.njit
def _describe_1d(a):
    mo = np.zeros((3,), dtype=a.dtype)
    (n, mn, mx, mean) = single_scan_stats(a)
    if isnan(mean):
        m2 = m3 = m4 = sk = kurt = float("nan")
    else:
        moments(a, mean, n, (2, 3, 4), mo)
        m2, m3, m4 = mo
        if m2 == 0.0:
            sk = kurt = float("nan")
        else:
            sk = m3 / m2**1.5
            kurt = m4 / m2**2.0 - 3

    return DescribeResult(n, (mn, mx), mean, m2, sk, kurt)


@numba.njit(parallel=True, cache=True)
def describe(a, axis=-1):
    if axis < 0:
        axis += a.ndim
    reduced = np.ascontiguousarray(np.moveaxis(a, axis, -1))
    rows = reduced.reshape(-1, reduced.shape[-1])
    shape = (rows.shape[0],)
    nobs = np.empty(shape, dtype=np.int64)
    mn = np.empty(shape)
    mx = np.empty(shape)
    mean = np.empty(shape)
    variance = np.empty(shape)
    skewness = np.empty(shape)
    kurtosis = np.empty(shape)
    for i in numba.prange(rows.shape[0]):
        result = _describe_1d(rows[i])
        nobs[i] = result.nobs
        mn[i] = result.minmax[0]
        mx[i] = result.minmax[1]
        mean[i] = result.mean
        variance[i] = result.variance
        skewness[i] = result.skewness
        kurtosis[i] = result.kurtosis
    return DescribeResult(
        nobs, (mn, mx), mean, variance, skewness, kurtosis
    )
