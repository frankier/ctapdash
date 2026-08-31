import xarray as xr
from tsdownsample import MinMaxLTTBDownsampler
from zarr.codecs import BloscCodec


def convert_to_xarray(eeg):
    # Extract coordinates for the specified dimensions
    arr, times = eeg.get_data(return_times=True)

    data_array = xr.DataArray(
        arr,
        coords=(eeg.ch_names, times),
        dims=("ch", "time")
    )
    ds = data_array.to_dataset(name='data')
    return ds


def help_downsample(data, time, n_out):
    """
    Helper function for downsampling and returning as a specific format.
    """
    indices = MinMaxLTTBDownsampler().downsample(time, data, n_out=n_out)
    return data[indices], indices


def apply_downsample(ts_ds, factor, dims):
    """
    Apply downsampling to a time series dataset.
    """
    dim = dims[0]
    n_out = ts_ds["data"].shape[-1] // factor
    ts_ds_downsampled, indices = xr.apply_ufunc(
        help_downsample,
        ts_ds["data"],
        ts_ds[dim],
        kwargs=dict(n_out=n_out),
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim], ["indices"]],
        exclude_dims=set((dim,)),
        vectorize=True,
    )
    ts_ds_downsampled[dim] = ts_ds[dim].isel(time=indices.values[0])
    ds = ts_ds_downsampled.rename("data")
    return ds


compressor = BloscCodec(cname="zstd", clevel=9)


def mne_to_pyramid(mne_raw, pyramid_path, factors):
    from shutil import rmtree
    ts_ds = convert_to_xarray(mne_raw)
    rmtree(pyramid_path, ignore_errors=True)

    encoding = {
        "data": {"compressor": compressor},
    }
    for factor in factors:
        name = "factor_" + str(factor)
        data = apply_downsample(ts_ds, factor=factor, dims=["time"])
        data.to_zarr(pyramid_path, group=name, mode="a", encoding=encoding)
