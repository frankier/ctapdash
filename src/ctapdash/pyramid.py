import xarray as xr
import numpy as np
from tsdownsample import MinMaxLTTBDownsampler
from zarr.codecs import BloscCodec
import numba
from pathlib import Path


def convert_to_xarray(eeg):
    # Extract coordinates for the specified dimensions
    arr, times = eeg.get_data(return_times=True)

    return xr.DataArray(
        arr,
        coords=(eeg.ch_names, times),
        dims=("ch", "time")
    )


def help_downsample(data, time, n_out):
    indices = MinMaxLTTBDownsampler().downsample(time, data, n_out=n_out)
    return data[indices]


def apply_downsample(arr, factor):
    """
    Apply downsampling to a time series dataset.
    """
    n_out = arr.shape[-1] // factor
    downsampled = xr.apply_ufunc(
        help_downsample,
        arr,
        arr["time"],
        kwargs=dict(n_out=n_out),
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["time"]],
        exclude_dims=set(("time",)),
        vectorize=True,
    )
    slicer = slice(0, n_out * factor, factor)
    new_time = arr["time"].isel(time=slicer).copy()
    downsampled["time"] = new_time
    return downsampled


def mne_to_pyramid(arr, pyramid_path, factors):
    from shutil import rmtree
    rmtree(pyramid_path, ignore_errors=True)

    effective_factor = 1
    cur_arr = arr
    for factor in factors:
        effective_factor *= factor
        name = "factor_" + str(effective_factor)
        cur_arr = apply_downsample(cur_arr, factor=factor)
        cur_arr.to_zarr(pyramid_path, group=name, mode="a", consolidated=False)


@numba.njit
def _range_downsample(x, factor, res):
    if x.ndim == 2:
        for i in range(len(x)):
            bucket = i // factor
            if bucket >= len(res):
                break
            res[bucket, 0] = min(x[i, 0], res[bucket, 0])
            res[bucket, 1] = max(x[i, 1], res[bucket, 1])
        return res
    else:
        for i in range(len(x)):
            bucket = i // factor
            if bucket >= len(res):
                break
            res[bucket, 0] = min(x[i], res[bucket, 0])
            res[bucket, 1] = max(x[i], res[bucket, 1])
        return res


@numba.njit(parallel=True, cache=True)
def range_downsample_with_ch(x, factor, out):
    for ch in numba.prange(x.shape[0]):
        _range_downsample(x[ch], factor, out[ch])


@numba.njit(parallel=True, cache=True)
def range_downsample_with_epochs_ch(x, factor, out):
    for epoch in numba.prange(x.shape[0]):
        for ch in range(x.shape[1]):
            _range_downsample(x[epoch, ch], factor, out[epoch, ch])


def mne_to_rangepyramid(arr, rangepyramid_path, factors):
    from shutil import rmtree
    rmtree(rangepyramid_path, ignore_errors=True)

    leading_shape = arr.shape[:-1]
    leading_coords = list(arr.coords.values())[:-1]
    leading_dims = arr.dims[:-1]
    effective_factor = 1
    cur_arr = arr.data
    cur_len = arr.shape[-1]
    for factor in factors:
        effective_factor *= factor
        cur_len = cur_len // factor
        out = np.zeros((*leading_shape, cur_len, 2), dtype=arr.dtype)
        out[..., 0] = np.inf
        out[..., 1] = -np.inf
        name = "factor_" + str(effective_factor)
        if len(leading_dims) == 1:
            range_downsample_with_ch(cur_arr, factor, out)
        else:
            assert len(leading_dims) == 2
            range_downsample_with_epochs_ch(cur_arr, factor, out)
        cur_arr = out
        cur_xarr = xr.DataArray(
            cur_arr,
            coords=(
                *leading_coords,
                arr["time"].isel(time=slice(0, cur_len * effective_factor, effective_factor)),
                ["min", "max"]
            ),
            dims=(*leading_dims, "time", "range"),
        )
        cur_xarr.to_zarr(rangepyramid_path, group=name, mode="a", consolidated=False)


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
    ext = "rangepyramid" if range else "pyramid"
    if not isinstance(base, (str, Path)):
        base = base.filenames[0]
        if base.suffix == ".fdt":
            base = base.with_suffix(".set")
        if base.suffix != ".set":
            raise ValueError(f"Expected .set file, got {base}")
    pyramid_path = f"{base}.{ext}"
    dt = xr.open_datatree(pyramid_path, engine="zarr", consolidated=False)
    return dt, _pyramid_groups(dt)
