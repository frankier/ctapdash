import xarray as xr
import numpy as np
from tsdownsample import MinMaxLTTBDownsampler
from pathlib import Path

from ctapdash.io.xarray import load_xarray, save_xarray

_LEVEL_VARIABLE = "__xarray_dataarray_variable__"


def convert_to_xarray(eeg):
    # Extract coordinates for the specified dimensions
    arr, times = eeg.get_data(return_times=True)

    return xr.DataArray(arr, coords=(eeg.ch_names, times), dims=("ch", "time"))


def help_downsample(data, time, n_out):
    if n_out < 3:
        # LTTB needs at least three outputs. Small final levels use endpoints.
        return data[np.linspace(0, len(data) - 1, n_out, dtype=int)]
    indices = MinMaxLTTBDownsampler().downsample(time, data, n_out=n_out)
    return data[indices]


GRID_VERSION = "1"


def grid_cache_current(path):
    try:
        return (Path(path) / "sample-grid-version").read_text() == GRID_VERSION
    except FileNotFoundError, NotADirectoryError:
        return False


def _aligned_pyramid(arr, destination, factors, *, ranges):
    """Reduce on the sample-zero grid, retaining each epoch's partial buckets.

    Epochs remain separate even when they overlap. A level stores the first
    global bucket's sample offset and the number of valid buckets per epoch;
    unused slots in the rectangular backing array are NaN.
    """
    epoched = "epoch" in arr.dims
    if epoched and "epoch_sample_start" not in arr.coords:
        raise ValueError("Epoched pyramids require epoch_sample_start coordinates")
    rate = arr.attrs.get("sample_rate")
    if rate is None:
        rate = 1 / float(arr.time[1] - arr.time[0])
    rate = float(rate)
    starts = (
        np.asarray(arr.epoch_sample_start.values, dtype=np.int64)
        if epoched
        else np.array([round(float(arr.time[0]) * rate)], dtype=np.int64)
    )
    levels = {}
    effective = 1
    # Work one epoch at a time; do not copy the complete raw recording.
    current = [
        arr.isel(epoch=i).values if epoched else arr.values for i in range(len(starts))
    ]
    for factor in factors:
        if factor < 2:
            raise ValueError("Pyramid factors must be at least two")
        effective *= factor
        reduced = []
        for epoch, values in enumerate(current):
            phase = int(starts[epoch] % factor)
            length = values.shape[1]
            boundaries = np.arange(-phase, length, factor)
            boundaries[0] = 0
            if ranges:
                lower = values[..., 0] if values.ndim == 3 else values
                upper = values[..., 1] if values.ndim == 3 else values
                result = np.stack(
                    (
                        np.fmin.reduceat(lower, boundaries, axis=1),
                        np.fmax.reduceat(upper, boundaries, axis=1),
                    ),
                    axis=-1,
                )
            else:
                # Keep the existing LTTB line reduction, with bucket slots
                # aligned by extending the two endpoint values for partial bins.
                count = len(boundaries)
                padded = np.pad(
                    values,
                    ((0, 0), (phase, count * factor - phase - length)),
                    mode="edge",
                )
                times = np.arange(padded.shape[1], dtype=np.float64)
                result = np.stack(
                    [
                        help_downsample(np.ascontiguousarray(row), times, count)
                        for row in padded
                    ]
                )
            reduced.append(result)
        starts = starts // factor
        counts = np.array([v.shape[1] for v in reduced], dtype=np.int64)
        shape = (arr.sizes["ch"], len(starts), int(counts.max()))
        if ranges:
            shape += (2,)
        output = np.full(shape, np.nan, dtype=arr.dtype)
        for epoch, values in enumerate(reduced):
            output[:, epoch, : counts[epoch]] = values
        coords = dict(ch=arr.ch, time=np.arange(counts.max()) * effective / rate)
        dims = ("ch", "epoch", "time")
        if epoched:
            coords.update(
                epoch=arr.epoch,
                bucket_sample_start=("epoch", starts * effective),
                bucket_count=("epoch", counts),
            )
        else:
            output = output[:, 0]
            dims = ("ch", "time")
            coords["time"] += starts[0] * effective / rate
            coords.update(
                bucket_sample_start=int(starts[0] * effective),
                bucket_count=int(counts[0]),
            )
        if ranges:
            dims += ("range",)
            coords["range"] = ["min", "max"]
        level = xr.DataArray(output, dims=dims, coords=coords, name=_LEVEL_VARIABLE)
        levels[f"factor_{effective}"] = level.to_dataset()
        current = reduced
    save_xarray(xr.DataTree.from_dict(levels), destination)
    (Path(destination) / "sample-grid-version").write_text(GRID_VERSION)


def mne_to_pyramid(arr, pyramid_path, factors):
    _aligned_pyramid(arr, pyramid_path, factors, ranges=False)


def mne_to_rangepyramid(arr, rangepyramid_path, factors):
    _aligned_pyramid(arr, rangepyramid_path, factors, ranges=True)


def _pyramid_groups(ts_dt):
    """Pyramid levels of a DataTree sorted from finest to coarsest."""
    # DataTree includes the (usually empty) root group in ``groups``. Only
    # nodes containing pyramid samples are levels.
    return tuple(
        sorted(
            (group for group in ts_dt.groups if "time" in ts_dt[group].ds),
            key=lambda group: ts_dt[group].ds["time"].size,
            reverse=True,
        )
    )


def load_pyramid(base, range=False):
    """Open an explicit artifact path or RecordingPaths/RecordingData handle."""
    from ctapdash.io.paths import RecordingPaths

    if hasattr(base, "paths"):
        base = base.paths
    if isinstance(base, RecordingPaths):
        base = base.rangepyramid if range else base.pyramid
    path = Path(base)
    if path.suffix not in (".pyramid", ".rangepyramid"):
        raise ValueError(
            "Pass a RecordingPaths handle or an explicit pyramid artifact path"
        )
    if not grid_cache_current(path):
        raise FileNotFoundError(
            f"Pyramid requires rebuilding on the sample grid: {path}"
        )
    dt = load_xarray(path)
    return dt, _pyramid_groups(dt)
