from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _sample_interval(times: NDArray[np.float64], identity: str) -> float:
    if len(times) < 2:
        raise ValueError(f"{identity} must contain at least two samples")
    if not np.all(np.isfinite(times)):
        index = int(np.flatnonzero(~np.isfinite(times))[0])
        raise ValueError(f"{identity} has a non-finite time coordinate at index {index}")
    deltas = np.diff(times)
    if np.any(deltas <= 0):
        index = int(np.flatnonzero(deltas <= 0)[0] + 1)
        raise ValueError(f"{identity} has non-monotonic time at index {index}")
    interval = float(np.median(deltas))
    tolerance = max(abs(interval) * 1e-6, np.finfo(np.float64).eps * 16)
    mismatch = np.flatnonzero(np.abs(deltas - interval) > tolerance)
    if len(mismatch):
        index = int(mismatch[0] + 1)
        raise ValueError(f"{identity} has irregular time coordinates at index {index}")
    return interval


def resolve_channel(channel_names: Sequence[str], requested: str) -> tuple[int, str]:
    """Resolve a channel by exact name or zero-based integer index."""
    if requested in channel_names:
        index = channel_names.index(requested)
        return index, channel_names[index]
    try:
        index = int(requested)
    except ValueError:
        index = -1
    if not 0 <= index < len(channel_names):
        available = ", ".join(channel_names)
        raise ValueError(f"unknown channel {requested!r}; available channels: {available}")
    return index, channel_names[index]


class RangeSeries(Protocol):
    """Uniform time series exposing raw samples and range-pyramid levels."""

    dataset_version: str
    channel_identity: str
    dtype: np.dtype
    sample_count: int
    time_start: float
    sample_interval: float

    @property
    def source_factors(self) -> tuple[int, ...]: ...

    @property
    def times(self) -> NDArray[np.floating]: ...

    def level_length(self, source_factor: int) -> int: ...

    def slice_ranges(
        self, source_factor: int, start: int, stop: int
    ) -> NDArray[np.floating]: ...

    def finite_extrema(self, stop: int | None = None) -> tuple[float, float]: ...


class ArrayRangeSeries:
    """A RangeSeries backed by NumPy arrays.

    ``levels`` maps aggregation factors to ``(n, 2)`` min/max arrays.  Factor
    one may be omitted; raw samples are always exposed as degenerate ranges.
    """

    def __init__(
        self,
        values: ArrayLike,
        times: ArrayLike,
        *,
        channel_identity: str,
        dataset_version: str = "memory",
        levels: dict[int, ArrayLike] | None = None,
    ) -> None:
        raw = np.asarray(values)
        time = np.asarray(times, dtype=np.float64)
        if raw.ndim != 1 or time.ndim != 1 or raw.shape != time.shape:
            raise ValueError("values and times must be equal-length 1-D arrays")
        self._raw = raw
        self._times = time
        self.channel_identity = channel_identity
        self.dataset_version = dataset_version
        self.dtype = raw.dtype
        self.sample_count = len(raw)
        self.time_start = float(time[0]) if len(time) else 0.0
        self.sample_interval = _sample_interval(time, channel_identity)
        self._levels: dict[int, NDArray[np.floating]] = {}
        for factor, ranges in (levels or {}).items():
            if factor <= 1:
                raise ValueError("stored range factors must be greater than one")
            level = np.asarray(ranges)
            if level.ndim != 2 or level.shape[1] != 2:
                raise ValueError("range levels must have shape (n, 2)")
            self._levels[int(factor)] = level

    @property
    def source_factors(self) -> tuple[int, ...]:
        return (1, *sorted(self._levels))

    @property
    def times(self) -> NDArray[np.float64]:
        return self._times

    def level_length(self, source_factor: int) -> int:
        if source_factor == 1:
            return self.sample_count
        return min(len(self._levels[source_factor]), self.sample_count // source_factor)

    def slice_ranges(
        self, source_factor: int, start: int, stop: int
    ) -> NDArray[np.floating]:
        length = self.level_length(source_factor)
        start = max(0, min(start, length))
        stop = max(start, min(stop, length))
        if source_factor == 1:
            raw = np.asarray(self._raw[start:stop])
            return np.column_stack((raw, raw))
        return np.asarray(self._levels[source_factor][start:stop])

    def finite_extrema(self, stop: int | None = None) -> tuple[float, float]:
        data = np.asarray(self._raw[:stop])
        finite = data[np.isfinite(data)]
        if not len(finite):
            raise ValueError(f"{self.channel_identity} contains no finite samples")
        return float(finite.min()), float(finite.max())


class RecordingRangeSeries:
    """Lazy RangeSeries adapter for one channel of an EEGLAB recording."""

    def __init__(self, path: Path, requested_channel: str) -> None:
        from ctapdash.io import read_eeglab
        from ctapdash.pyramid import load_pyramid

        self.path = Path(path).resolve()
        recording = read_eeglab(self.path)
        if not hasattr(recording, "mmap"):
            raise ValueError(f"{self.path} cannot provide memory-mapped raw samples")
        if recording.__class__.__name__.lower().find("epoch") >= 0:
            raise ValueError(
                f"{self.path} is epoched; directly addressable epoch selection is not implemented"
            )
        channel_index, channel_name = resolve_channel(recording.ch_names, requested_channel)
        raw_array = recording.mmap(return_xarray=True)
        self._raw = raw_array.isel(ch=channel_index)
        self._times = np.asarray(raw_array["time"].data, dtype=np.float64)
        self.channel_identity = f"{self.path}:{channel_name}"
        self.sample_count = int(self._raw.sizes["time"])
        self.time_start = float(self._times[0])
        self.sample_interval = _sample_interval(self._times, self.channel_identity)
        self.dtype = np.dtype(self._raw.dtype)
        stat = self.path.stat()
        self.dataset_version = f"{self.path}:{stat.st_size}:{stat.st_mtime_ns}:{channel_name}"
        self._levels: dict[int, object] = {}
        pyramid_path = Path(f"{self.path}.rangepyramid")
        if pyramid_path.exists():
            tree, groups = load_pyramid(self.path, range=True)
            for group in groups:
                factor = int(group.rsplit("factor_", 1)[1])
                dataset = tree[group].ds
                data_array = next(iter(dataset.data_vars.values()))
                if "ch" in data_array.dims:
                    data_array = data_array.sel(ch=channel_name)
                self._levels[factor] = data_array
            pstat = pyramid_path.stat()
            self.dataset_version += f":{pstat.st_mtime_ns}"

    @property
    def source_factors(self) -> tuple[int, ...]:
        return (1, *sorted(self._levels))

    @property
    def times(self) -> NDArray[np.float64]:
        return self._times

    def level_length(self, source_factor: int) -> int:
        if source_factor == 1:
            return self.sample_count
        level = self._levels[source_factor]
        return min(int(level.sizes["time"]), self.sample_count // source_factor)  # type: ignore[attr-defined]

    def slice_ranges(
        self, source_factor: int, start: int, stop: int
    ) -> NDArray[np.floating]:
        length = self.level_length(source_factor)
        start = max(0, min(start, length))
        stop = max(start, min(stop, length))
        if source_factor == 1:
            raw = np.asarray(self._raw.isel(time=slice(start, stop)).data)
            return np.column_stack((raw, raw))
        level = self._levels[source_factor]
        return np.asarray(level.isel(time=slice(start, stop)).data)  # type: ignore[attr-defined]

    def finite_extrema(self, stop: int | None = None) -> tuple[float, float]:
        stop = self.sample_count if stop is None else min(stop, self.sample_count)
        minimum, maximum = np.inf, -np.inf
        for start in range(0, stop, 1_000_000):
            block = np.asarray(
                self._raw.isel(time=slice(start, min(start + 1_000_000, stop))).data
            )
            finite = block[np.isfinite(block)]
            if len(finite):
                minimum = min(minimum, float(finite.min()))
                maximum = max(maximum, float(finite.max()))
        if not np.isfinite(minimum):
            raise ValueError(f"{self.channel_identity} contains no finite samples")
        return minimum, maximum