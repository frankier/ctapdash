import numba
import numpy as np


@numba.njit
def inranges(range_array, target):
    results = np.ones((len(range_array),), dtype=np.bool)
    results &= range_array[:, 0] <= target
    results &= target <= range_array[:, 1]
    return results


@numba.njit(cache=True)
def draw(arr, ranges, overall_range):
    y_min, y_max = overall_range
    height, width,  _ = arr.shape
    cur_ranges = np.zeros((2, 2), dtype=ranges[0].dtype)
    num_buckets = len(ranges[0])
    scale_factor = num_buckets / width
    for x in range(width):
        start_bucket_idx = int(scale_factor * x)
        if start_bucket_idx >= num_buckets:
            break
        next_bucket_idx = int(scale_factor * (x + 1))
        if start_bucket_idx == next_bucket_idx:
            end_bucket_idx = start_bucket_idx + 1
        else:
            end_bucket_idx = min(next_bucket_idx, num_buckets)
        cur_ranges[:, 0] = np.inf
        cur_ranges[:, 1] = -np.inf
        for i in range(2):
            cur_ranges[i, 0] = min(cur_ranges[i, 0], ranges[i][start_bucket_idx:end_bucket_idx, 0].min())
            cur_ranges[i, 1] = max(cur_ranges[i, 1], ranges[i][start_bucket_idx:end_bucket_idx, 1].max())
        #print(cur_ranges)
        for y in range(height):
            data_y = y_min + (y_max - y_min) * (1 - y / height)
            insides = inranges(cur_ranges, data_y)
            if insides[0] and insides[1]:
                col = (0, 0, 0)
            elif not insides[0] and not insides[1]:
                col = (255, 255, 255)
            elif insides[0] and not insides[1]:
                col = (255, 0, 0)
            elif not insides[0] and insides[1]:
                col = (0, 0, 255)
            else:
                assert False
            arr[y, x, :] = col


def venn_time_series(eegs, width_px, height_px, time_range):
    import time
    from matplotlib.image import imsave

    das = []
    min_size = float("inf")
    for eeg in eegs:
        ts_dt, groups = eeg.open_pyramid(range=True)
        coarsest_name = groups[-1]
        da = ts_dt[coarsest_name]["__xarray_dataarray_variable__"]
        das.append(da)
        if len(da["time"]) < min_size:
            min_size = len(da["time"])
    coarsests = []
    for da in das:
        coarsests.append(da.data[0, :min_size])
    overall_range = (min([c[:, 0].min() for c in coarsests]), max([c[:, 1].max() for c in coarsests]))
    arr = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    assert len(coarsests) == 2
    start = time.time()
    draw(arr, coarsests, overall_range)
    print("Time taken: {:.2f} seconds".format(time.time() - start))
    imsave('name.png', arr)
