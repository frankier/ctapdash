"""Original-time alignment and bounded domain tiles from real EEGLab fixtures."""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from bokeh.document import Document
from scipy.io import loadmat, savemat

from ctapdash.io.eeglab import CtapEpochEEGLAB, CtapRawEEGLAB
from venn_ts.domain import ComparisonDomain
from venn_ts.range_series import RecordingSegment, RecordingTileSource
from venn_ts.renderer import TileCoordinator, VennTimeSeriesRenderer
from tests.dummy_data import write_eeglab


def source(tmp_path, name, *, starts=None, count=100):
    path = write_eeglab(
        tmp_path / name,
        len(starts) if starts else 1,
        n_times=count,
        epoch_starts=starts,
    )
    reader = CtapEpochEEGLAB if starts else CtapRawEEGLAB
    return RecordingTileSource(path, recording=reader(path, verbose="error"))


def edit(path, change):
    data = {
        k: v
        for k, v in loadmat(path, simplify_cells=True).items()
        if not k.startswith("__")
    }
    change(data)
    savemat(path, data)


def coordinator(domain, page_size=4):
    renderer = VennTimeSeriesRenderer(
        sample_count=domain.sample_count,
        page_size=page_size,
        sample_interval=domain.interval,
        time_start=domain.start,
        time_end=domain.end,
        channel_names=["A"],
    )
    return TileCoordinator(Document(), renderer, domain.sources, ["A"], domain=domain)


def test_original_epoch_timing_and_pickle(tmp_path):
    data = source(tmp_path, "epochs", starts=[10, 300])
    assert [s.time_start for s in data.segments] == [0.1, 3.0]
    restored = pickle.loads(pickle.dumps(data.recording))
    np.testing.assert_array_equal(restored.events[:, 0], [10, 300])
    assert RecordingTileSource(data.path, recording=restored).segments == data.segments
    assert data.raw.dims == ("ch", "epoch", "time")


def test_union_one_sided_data_and_intersection_compaction(tmp_path):
    raw = source(tmp_path, "raw", count=400)
    epochs = source(tmp_path, "epochs", starts=[50, 250])
    union = ComparisonDomain((raw, epochs))
    assert union.sample_count == 400
    assert [s.count for s in union.segments] == [50, 100, 100, 100, 50]
    assert union.start == 0 and union.end == pytest.approx(4)
    tiles = coordinator(union)
    try:
        tile = tiles._range_tile(1, 0, 0)
        assert tile["valid_a"].all() and not tile["valid_b"].any()
        assert tile["clip_start"] == 0 and tile["clip_end"] == pytest.approx(0.04)
    finally:
        tiles.close()
    intersection = ComparisonDomain((raw, epochs), "intersection")
    assert intersection.sample_count == 200
    assert [s.original_start for s in intersection.segments] == pytest.approx(
        [0.5, 2.5]
    )
    assert [s.display_start for s in intersection.segments] == pytest.approx([0, 1])
    assert list(intersection.epoch_starts()) == pytest.approx([0, 1])
    assert list(intersection.epoch_starts().values()) == [{1}, {1}]
    times, data = intersection.read(intersection.segments[1], 0, [0], 1, 0, 4, "lines")
    np.testing.assert_allclose(times, [1, 1.01, 1.02, 1.03])
    np.testing.assert_array_equal(data[0], [250, 251, 252, 253])


def test_unequal_continuous_domains_and_union_gaps(tmp_path):
    a, b = source(tmp_path, "a", count=10), source(tmp_path, "b", count=20)
    assert ComparisonDomain((a, b)).sample_count == 20
    assert ComparisonDomain((a, b), "intersection").sample_count == 10
    b.segments = (RecordingSegment(1, 20),)
    union = ComparisonDomain((a, b))
    assert union.end == pytest.approx(0.3)
    assert union.segments[1].display_start == 0.1
    assert union.segments[1].original_start == 1
    with pytest.raises(ValueError, match="empty intersection"):
        ComparisonDomain((a, b), "intersection")


def test_overlap_pair_order_and_independent_union_choices(tmp_path):
    a = source(tmp_path, "a", starts=[100, 50], count=100)
    b = source(tmp_path, "b", starts=[75, 100], count=100)
    intersection = ComparisonDomain((a, b), "intersection")
    assert [
        (round(s.original_start, 6), s.selected(0), s.selected(1))
        for s in intersection.segments
    ] == [(0.75, 1, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1)]
    union = ComparisonDomain((a, b))
    region = next(
        i
        for i, s in enumerate(union.segments)
        if len(s.choices_a) == len(s.choices_b) == 2
    )
    chosen = ComparisonDomain((a, b), selections={(region, 0): 1})
    s = chosen.segments[region]
    assert s.selected(0) == 1 and s.selected(1) == 0
    _, data = chosen.read(s, 0, [0], 1, 0, 2, "lines")
    np.testing.assert_array_equal(data[0], [250, 251])
    assert chosen.end == union.end


def test_sampling_grid_validation(tmp_path):
    a, b = source(tmp_path, "a"), source(tmp_path, "b", starts=[0, 10])
    b.segments = (RecordingSegment(0.005, 100, epoch=0),)
    with pytest.raises(ValueError, match="sampling grids"):
        ComparisonDomain((a, b))
    b.sample_interval = 0.02
    with pytest.raises(ValueError, match="sampling intervals"):
        ComparisonDomain((a, b))


def test_continuous_boundary_events_do_not_split_domain(tmp_path):
    a = source(tmp_path, "a", count=10)

    def boundary(d):
        d["event"] = np.rec.fromarrays(
            [[5.5], ["boundary"], [10]], names=["latency", "type", "duration"]
        )

    edit(a.path, boundary)
    raw = CtapRawEEGLAB(a.path, verbose="error")
    assert RecordingTileSource(a.path, recording=raw).segments == (
        RecordingSegment(0, 10),
    )


def test_tiles_clip_final_bucket_and_do_not_cross_epochs(tmp_path):
    a = source(tmp_path, "a", starts=[10, 100], count=7)
    domain = ComparisonDomain((a, a), "intersection")
    tiles = coordinator(domain, page_size=2)
    try:
        assert domain.page_counts([1, 4], 2) == [7, 2]
        assert domain.segment_page_counts([1, 4], 2) == [8, 3]
        tile = tiles._range_tile(4, 0, 0)
        np.testing.assert_array_equal(tile["minimum_a"], [0, 2, 6])
        np.testing.assert_array_equal(tile["maximum_a"], [1, 5, 6])
        assert tile["clip_start"] == pytest.approx(0.1)
        assert tile["clip_end"] == pytest.approx(0.16)
        tail = tiles._range_tile(4, 1, 0)
        assert tail["clip_end"] == pytest.approx(0.17)
        assert tile["segment_id"] == 0
        other = tiles._range_tile(4, 2, 0)
        np.testing.assert_array_equal(other["minimum_a"], [14, 18])
        assert other["clip_start"] == pytest.approx(0.17)
        line = tiles._line_tile(1, 3, 0)[0]
        assert line["time"][-1] < other["clip_start"]
    finally:
        tiles.close()


def test_pyramid_reuse_and_unaligned_boundary_reads(tmp_path):
    import xarray as xr

    a = source(tmp_path, "a", starts=[0, 20], count=16)
    values = np.asarray(a.raw.data)
    blocks = values.reshape(2, 2, 4, 4)
    ranges = np.stack((blocks.min(axis=-1), blocks.max(axis=-1)), axis=-1)
    array = xr.DataArray(
        ranges,
        dims=("ch", "epoch", "time", "minmax"),
        coords={
            "bucket_sample_start": ("epoch", [0, 20]),
            "bucket_count": ("epoch", [4, 4]),
        },
    )
    a.range_groups = {4: "factor_4"}
    a.range_tree = {"factor_4": SimpleNamespace(ds=array.to_dataset(name="ranges"))}
    for offset, count, factor in [(0, 16, 4), (1, 13, 8), (2, 7, 4)]:
        phase = (20 + offset) % factor
        n = (phase + count + factor - 1) // factor
        actual = a.read_segment(a.segments[1], [0], offset, count, factor, 0, n, "venn")
        expected = []
        for start in range(offset - phase, offset + count, factor):
            chunk = values[0, 1, max(0, start) : min(start + factor, 16)]
            expected.append([chunk.min(), chunk.max()])
        np.testing.assert_array_equal(actual[0], expected)


def test_metadata_cache_needs_no_timing_version(tmp_path):
    from ctapdash.io.eeglab import read_eeglab

    path = write_eeglab(tmp_path, 2, epoch_starts=[50, 300])
    recording = read_eeglab(path)
    assert not hasattr(recording, "original_time_version")
    restored = read_eeglab(path, validated=True)
    np.testing.assert_array_equal(restored.events, recording.events)
    source = RecordingTileSource(path, recording=restored)
    assert [s.time_start for s in source.segments] == [0.5, 3]


def test_closed_coordinator_discards_inflight_domain_response(tmp_path):
    from concurrent.futures import Future

    a = source(tmp_path, "a", starts=[0, 50])
    original = coordinator(ComparisonDomain((a, a)))
    replacement = coordinator(ComparisonDomain((a, a), "intersection"))
    request = (("venn", 1, 0, 0),)
    future = Future()
    future.set_result(original._load(1, request))
    original.close()
    original._loaded(future, 1, request)
    try:
        for callback in list(original.document.session_callbacks):
            callback.callback()
        assert original.renderer.response_seq == 0
        assert replacement.renderer.response_seq == 0
        assert not original._pending_responses
    finally:
        replacement.close()


def test_epoch_offsets_ignore_other_event_metadata(tmp_path):
    a = source(tmp_path, "a", starts=[25, 400], count=100)

    def unrelated_metadata(d):
        d["urevent"] = np.rec.fromarrays([[999.0, 9999.0]], names=["latency"])

    edit(a.path, unrelated_metadata)
    recording = CtapEpochEEGLAB(a.path, verbose="error")
    assert recording.times[0] < 0
    # Even stale derived timing attributes must not override recording.events.
    recording.original_time_segments = (RecordingSegment(999, 100),)
    recording.original_time_error = "unusable urevent metadata"
    restored = RecordingTileSource(a.path, recording=recording)
    assert [s.time_start for s in restored.segments] == [0.25, 4]


def test_epoch_switch_changes_only_selected_segment(tmp_path):
    raw = source(tmp_path, "raw", count=400)
    epochs = source(tmp_path, "epochs", starts=[50, 100], count=100)
    domain = ComparisonDomain((raw, epochs))
    tiles = coordinator(domain)
    tiles.renderer.segment_revisions = [0] * len(domain.segments)
    segment_id = next(i for i, s in enumerate(domain.segments) if len(s.choices_b) == 2)
    before = list(domain.segments)
    try:
        old_times, old_values = domain.read(
            before[segment_id], 1, [0], 1, 0, 4, "lines"
        )
        tiles.select_epoch(segment_id, 1, 1)
        new_times, new_values = domain.read(
            domain.segments[segment_id], 1, [0], 1, 0, 4, "lines"
        )
        np.testing.assert_array_equal(old_times, new_times)
        assert not np.array_equal(old_values, new_values)
        # A worker already reading the old immutable segment keeps its snapshot.
        np.testing.assert_array_equal(
            domain.read(before[segment_id], 1, [0], 1, 0, 4, "lines")[1], old_values
        )
        for i, segment in enumerate(domain.segments):
            assert tiles.renderer.segment_revisions[i] == int(i == segment_id)
            if i != segment_id:
                assert segment is before[i]
    finally:
        tiles.close()


def test_marker_models_are_bounded_and_update_in_place():
    from types import SimpleNamespace
    from bokeh.plotting import figure
    from venn_ts.domain_axis import configure_domain_axis, update_epoch_markers
    from venn_ts.range_series import RecordingSegment

    source = SimpleNamespace(
        sample_interval=0.01,
        segments=tuple(RecordingSegment(i * 2, 100, epoch=i) for i in range(1000)),
    )
    domain = ComparisonDomain((source, source), "intersection")
    plot = figure()
    markers = configure_domain_axis(plot, domain, epoch_visible=True)
    assert len(plot.renderers) == 2
    breaks, epochs = plot.renderers
    assert len(breaks.positions) == 999
    assert breaks.labels == ["//"] * 999
    assert len(epochs.positions) == 1000
    assert epochs.labels == ["AB"] * 1000
    update_epoch_markers(plot, domain, False, markers)
    assert plot.renderers == [breaks, epochs]
    assert not epochs.visible
