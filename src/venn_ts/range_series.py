from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

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

    def __init__(self, path: Path, requested_channel: str, *, recording=None) -> None:
        from ctapdash.io.eeglab import read_eeglab
        from ctapdash.io.pyramid import load_pyramid

        self.path = Path(path).resolve()
        if recording is None:
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
        self._extrema_cache: dict[int, tuple[float, float]] = {}
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
        if stop in self._extrema_cache:
            return self._extrema_cache[stop]
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
        result = (minimum, maximum)
        self._extrema_cache[stop] = result
        return result


class RecordingTileSource:
    """One lazily sliced, multichannel recording and its optional pyramids.

    The MNE object and its xarray-backed memory map are created once.  All
    viewer reads then select a bounded channel/time rectangle from that shared
    array or from a lazily opened Zarr pyramid.
    """

    def __init__(self, path: Path, *, recording=None) -> None:
        from ctapdash.io.eeglab import read_eeglab
        from ctapdash.io.pyramid import load_pyramid

        self.path = Path(path).resolve()
        if recording is None:
            recording = read_eeglab(self.path, mmap=True)
        if not hasattr(recording, "mmap"):
            raise ValueError(f"{self.path} does not support bounded memory-mapped reads")
        if "epoch" in recording.__class__.__name__.lower():
            raise ValueError(f"{self.path} is epoched; the comparison viewer supports continuous data only")

        # MmapRawEEGLAB must materialize embedded MATLAB data.  Reject that
        # case before calling mmap; external .fdt recordings remain zero-copy.
        filenames = [Path(name) for name in getattr(recording, "filenames", ()) if name]
        if (
            recording.__class__.__name__ == "MmapRawEEGLAB"
            and filenames
            and filenames[0].suffix.lower() != ".fdt"
        ):
            raise ValueError(
                f"{self.path} embeds its samples in the .set file and cannot be read "
                "in bounded windows; export it with an external .fdt file"
            )

        self.recording = recording
        self.raw = recording.mmap(return_xarray=True)
        if self.raw.dims != ("ch", "time"):
            raise ValueError(f"{self.path} is not a continuous channel/time recording")
        self.channels = tuple(str(value) for value in self.raw["ch"].values)
        self.channel_index = {channel: index for index, channel in enumerate(self.channels)}
        self.times = np.asarray(self.raw["time"].values, dtype=np.float64)
        identity = str(self.path)
        self.sample_interval = _sample_interval(self.times, identity)
        self.sample_count = len(self.times)
        self.time_start = float(self.times[0])
        stat = self.path.stat()
        self.dataset_version = f"{self.path}:{stat.st_size}:{stat.st_mtime_ns}"

        self.line_tree = None
        self.line_groups: dict[int, str] = {}
        self.range_tree = None
        self.range_groups: dict[int, str] = {}
        for is_range, attribute, groups_attribute in (
            (False, "line_tree", "line_groups"),
            (True, "range_tree", "range_groups"),
        ):
            pyramid_path = Path(f"{self.path}.{'rangepyramid' if is_range else 'pyramid'}")
            if not pyramid_path.exists():
                continue
            tree, groups = load_pyramid(self.path, range=is_range)
            setattr(self, attribute, tree)
            setattr(
                self,
                groups_attribute,
                {int(group.rsplit("factor_", 1)[1]): group for group in groups},
            )
            pstat = pyramid_path.stat()
            self.dataset_version += f":{pstat.st_mtime_ns}"

    @property
    def line_factors(self) -> tuple[int, ...]:
        return (1, *sorted(factor for factor in self.line_groups if factor != 1))

    @property
    def range_factors(self) -> tuple[int, ...]:
        return (1, *sorted(factor for factor in self.range_groups if factor != 1))

    def indices(self, channels: Sequence[str]) -> tuple[int, ...]:
        missing = [channel for channel in channels if channel not in self.channel_index]
        if missing:
            raise ValueError(f"Channels absent from {self.path}: {', '.join(missing)}")
        return tuple(self.channel_index[channel] for channel in channels)

    @staticmethod
    def _sole_array(tree, group):
        dataset = tree[group].ds
        if len(dataset.data_vars) != 1:
            raise ValueError(f"Expected one data array in pyramid group {group!r}")
        return next(iter(dataset.data_vars.values()))

    def level_length(self, layer: str, factor: int) -> int:
        if factor == 1:
            return self.sample_count
        groups = self.range_groups if layer == "venn" else self.line_groups
        tree = self.range_tree if layer == "venn" else self.line_tree
        return int(self._sole_array(tree, groups[factor]).sizes["time"])

    def slice_ranges(
        self, factor: int, channel_indices: Sequence[int], start: int, stop: int
    ) -> NDArray[np.floating]:
        if factor == 1 or factor not in self.range_groups:
            raw_start, raw_stop = start * factor, min(stop * factor, self.sample_count)
            values = np.asarray(
                self.raw.isel(ch=list(channel_indices), time=slice(raw_start, raw_stop)).data
            )
            if factor != 1:
                count = (values.shape[1] + factor - 1) // factor
                ranges = np.full((values.shape[0], count, 2), np.nan, dtype=values.dtype)
                for index in range(count):
                    block = values[:, index * factor : (index + 1) * factor]
                    ranges[:, index, 0] = np.nanmin(block, axis=1)
                    ranges[:, index, 1] = np.nanmax(block, axis=1)
                return ranges
            return np.stack((values, values), axis=-1)
        array = self._sole_array(self.range_tree, self.range_groups[factor])
        return np.asarray(array.isel(ch=list(channel_indices), time=slice(start, stop)).data)

    def slice_lines(
        self, factor: int, channel_indices: Sequence[int], start: int, stop: int
    ) -> tuple[NDArray[np.float64], NDArray[np.floating]]:
        if factor == 1 or factor not in self.line_groups:
            raw_start, raw_stop = start * factor, min(stop * factor, self.sample_count)
            positions = np.arange(raw_start, raw_stop, factor, dtype=np.int64)
            values = np.asarray(self.raw.isel(ch=list(channel_indices), time=positions).data)
            return self.times[positions], values
        array = self._sole_array(self.line_tree, self.line_groups[factor])
        selected = array.isel(ch=list(channel_indices), time=slice(start, stop))
        return np.asarray(selected["time"].values, dtype=np.float64), np.asarray(selected.data)

    def finite_extrema(
        self, channel_indices: Sequence[int], stop: int | None = None
    ) -> NDArray[np.float64]:
        stop = self.sample_count if stop is None else min(stop, self.sample_count)
        output = np.full((len(channel_indices), 2), (np.inf, -np.inf), dtype=np.float64)
        usable = [factor for factor in self.range_groups if factor > 1]
        if usable:
            factor = max(usable)
            array = self._sole_array(self.range_tree, self.range_groups[factor])
            count = min(int(array.sizes["time"]), (stop + factor - 1) // factor)
            ranges = np.asarray(
                array.isel(ch=list(channel_indices), time=slice(0, count)).data
            )
            output[:, 0] = np.nanmin(ranges[..., 0], axis=1)
            output[:, 1] = np.nanmax(ranges[..., 1], axis=1)
        else:
            # Bounded-memory fallback.  This touches the mmap sequentially but
            # never constructs a full in-memory recording.
            for start in range(0, stop, 262_144):
                block = np.asarray(
                    self.raw.isel(
                        ch=list(channel_indices),
                        time=slice(start, min(start + 262_144, stop)),
                    ).data
                )
                output[:, 0] = np.minimum(output[:, 0], np.nanmin(block, axis=1))
                output[:, 1] = np.maximum(output[:, 1], np.nanmax(block, axis=1))
        if not np.all(np.isfinite(output)):
            raise ValueError(f"{self.path} contains a channel with no finite samples")
        return output


def validate_tile_source_alignment(a: RecordingTileSource, b: RecordingTileSource) -> int:
    """Validate two recordings once and return their common sample count."""
    common = min(a.sample_count, b.sample_count)
    if common < 2:
        raise ValueError("The selected steps need at least two common samples")
    tolerance = max(
        abs(a.sample_interval) * 1e-5,
        abs(b.sample_interval) * 1e-5,
        np.finfo(np.float64).eps * 16,
    )
    if abs(a.sample_interval - b.sample_interval) > tolerance:
        raise ValueError("The selected steps have incompatible sampling intervals")
    mismatch = np.flatnonzero(np.abs(a.times[:common] - b.times[:common]) > tolerance)
    if len(mismatch):
        raise ValueError(f"The selected steps differ at time sample {int(mismatch[0])}")
    return common
