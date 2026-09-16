"""Global-grid pyramid construction, partial buckets, and cache-only tile reads."""

import numpy as np
import pytest

from ctapdash.io.eeglab import read_eeglab
from ctapdash.io.paths import DatasetPaths
from ctapdash.io.recording import RecordingData
from ctapdash.io.pyramid import mne_to_pyramid, mne_to_rangepyramid, load_pyramid
from ctapdash.io.cache import scan_dataset
from venn_ts.domain import ComparisonDomain
from venn_ts.range_series import RecordingTileSource
from tests.dummy_data import write_eeglab


def cached_source(root, step, *, starts=None, count=507):
    path = write_eeglab(
        root / step, len(starts) if starts else 1, n_times=count, epoch_starts=starts
    )
    recording = read_eeglab(path, use_cache=False)
    data = RecordingData(DatasetPaths(root).recording(path))
    arr = recording.mmap(return_xarray=True)
    for builder, dest in (
        (mne_to_pyramid, data.paths.pyramid),
        (mne_to_rangepyramid, data.paths.rangepyramid),
    ):
        builder(arr, dest, [8, 8])
    return RecordingTileSource(path, recording=recording, recording_data=data)


@pytest.mark.parametrize(
    "starts,count", [([54, 562, 1069], 507), ([0, 7], 1), ([-3, 10], 9), (None, 509)]
)
def test_levels_match_global_buckets_and_keep_partial_ends(tmp_path, starts, count):
    # The reader requires at least two samples for a viewer; builders also handle
    # short epochs, tested below without constructing a viewer source.
    if count == 1:
        import xarray as xr

        arr = xr.DataArray(
            np.array([[[2.0], [3.0]]], dtype=np.float32),
            dims=("ch", "epoch", "time"),
            coords={
                "ch": ["A"],
                "epoch": [0, 1],
                "time": [0.0],
                "epoch_sample_start": ("epoch", starts),
            },
            attrs={"sample_rate": 100.0},
        )
        dest = tmp_path / "short.rangepyramid"
        mne_to_rangepyramid(arr, dest, [8, 8])
        tree, _ = load_pyramid(dest, range=True)
        np.testing.assert_array_equal(
            tree["factor_64"].ds.to_array().values.ravel(), [2, 2, 3, 3]
        )
        return
    s = cached_source(tmp_path, "1_load", starts=starts, count=count)
    for factor, group in s.range_groups.items():
        array = s._sole_array(s.range_tree, group)
        for epoch, offset in enumerate(starts or [0]):
            level = array.isel(epoch=epoch) if starts else array
            raw = s.raw.isel(epoch=epoch).values if starts else s.raw.values
            first = offset // factor * factor
            assert int(level.bucket_sample_start) == first
            expected = []
            for bucket in range(first, offset + count, factor):
                block = raw[
                    :, max(0, bucket - offset) : min(count, bucket + factor - offset)
                ]
                expected.append(
                    np.stack((block.min(axis=1), block.max(axis=1)), axis=-1)
                )
            assert int(level.bucket_count) == len(expected)
            np.testing.assert_array_equal(
                level.values[:, : len(expected)], np.stack(expected, axis=1)
            )
            line = s._sole_array(s.line_tree, s.line_groups[factor])
            if starts:
                line = line.isel(epoch=epoch)
            assert int(line.bucket_sample_start) == first
            assert int(line.bucket_count) == len(expected)
            assert np.isfinite(line.values[:, : len(expected)]).all()


@pytest.mark.parametrize("mode", ["union", "intersection"])
def test_zoomed_out_domains_read_cached_buckets_without_raw(tmp_path, mode):
    a = cached_source(tmp_path, "1_load", count=1600)
    b = cached_source(tmp_path, "2_clean", starts=[54, 562, 1069])
    c = cached_source(tmp_path, "3_clean", starts=[75, 430, 1150])

    class NoRawReads:
        def __getattr__(self, name):
            pytest.fail(f"Zoomed-out tile accessed raw {name}")

    for source in (a, b, c):
        source.raw = NoRawReads()
    for pair in ((a, b), (b, c)):
        domain = ComparisonDomain(pair, mode)
        for factor in (8, 64):
            pages = domain.segment_page_counts([factor], 2)[0]
            for page in range(pages):
                _, segment, start, stop, _, _ = domain.page(factor, 2, page)
                first_bucket = domain.sample_start(segment) // factor * factor
                for side, source in enumerate(pair):
                    for layer in ("venn", "lines"):
                        _, values = domain.read(
                            segment, side, [0, 1], factor, start, stop, layer
                        )
                        selected = segment.selected(side)
                        if selected is None:
                            assert np.isnan(values).all()
                            continue
                        tree = (
                            source.range_tree if layer == "venn" else source.line_tree
                        )
                        level = source._sole_array(tree, f"/factor_{factor}")
                        epoch = source.segments[selected].epoch
                        if epoch is not None:
                            level = level.isel(epoch=epoch)
                        index = (
                            first_bucket - int(level.bucket_sample_start)
                        ) // factor
                        np.testing.assert_array_equal(
                            values, level.values[:, index + start : index + stop]
                        )


def test_old_pyramids_are_invalidated_even_with_fresh_mtime(tmp_path):
    s = cached_source(tmp_path, "1_load", starts=[54, 600])
    paths = DatasetPaths(tmp_path).recording(s.path)
    (paths.rangepyramid / "sample-grid-version").unlink()
    jobs = scan_dataset(tmp_path)
    states = {job.key[0]: job.state for job in jobs.values()}
    assert states["rangepyramid"] == "pending"
    assert states["pyramid"] == "ready"
    with pytest.raises(FileNotFoundError, match="rebuilding"):
        load_pyramid(paths.rangepyramid, range=True)
