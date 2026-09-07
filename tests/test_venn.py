from concurrent.futures import Future

import numpy as np
import pytest
from bokeh.document import Document

from ctapdash.plotting.venn_ts.range_series import (
    RecordingTileSource,
    resolve_channel,
    validate_tile_source_alignment,
)
from ctapdash.plotting.venn_ts.renderer import (
    PageMailbox,
    TileCoordinator,
    VennTimeSeriesRenderer,
)
from ctapdash.plotting.venn_ts.venn import (
    build_manifest,
    load_page,
    pack_pages,
    parse_series_args,
    reference_raster,
    validate_alignment,
)


from tests.dummy_data import FakeMmapRecording, series


def test_cli_selection_requires_exactly_two_and_resolves_names_or_indices(tmp_path):
    selections = parse_series_args(
        [[str(tmp_path / "a.set"), "Cz"], [str(tmp_path / "b.set"), "1"]]
    )
    assert selections[0].channel == "Cz"
    assert resolve_channel(["Fz", "Cz"], "Cz") == (1, "Cz")
    assert resolve_channel(["Fz", "Cz"], "0") == (0, "Fz")
    with pytest.raises(ValueError, match="exactly twice"):
        parse_series_args([["a.set", "Cz"]])
    with pytest.raises(ValueError, match="unknown channel"):
        resolve_channel(["Fz"], "Cz")


def test_alignment_truncates_tails_and_reports_first_mismatch():
    a = series(np.arange(5), name="a:Cz")
    b = series(np.arange(3), name="b:Cz")
    assert validate_alignment(a, b) == 3

    bad_times = np.arange(5, dtype=float) * 0.25 + 0.01
    with pytest.raises(ValueError, match=r"time mismatch at sample 0.*a:Cz.*b:Cz"):
        validate_alignment(a, series(np.arange(5), bad_times, name="b:Cz"))


def test_time_coordinates_are_finite_monotonic_and_regular():
    with pytest.raises(ValueError, match="non-finite.*index 1"):
        series([1, 2], [0, np.nan])
    with pytest.raises(ValueError, match="non-monotonic.*index 2"):
        series([1, 2, 3], [0, 1, 0.5])
    with pytest.raises(ValueError, match="irregular.*index 1"):
        series([1, 2, 3], [0, 1, 2.1])


def test_manifest_uses_common_length_shared_levels_and_padded_equal_extrema():
    level = np.array([[2, 2], [2, 2]])
    a = series([2] * 5, levels={2: level})
    b = series([2] * 4, levels={2: level})
    manifest = build_manifest(a, b, page_size=1)
    assert manifest.sample_count == 4
    assert manifest.source_factors == (1, 2)
    assert manifest.page_counts == (4, 2)
    assert manifest.initial_y_start < 2 < manifest.initial_y_end


def test_pages_have_stable_cores_one_entry_gutters_and_degenerate_raw_ranges():
    values = np.arange(8, dtype=float)
    s = series(values)
    middle = load_page(
        s, series_id=0, source_factor=1, page_index=1, common_length=8, page_size=3
    )
    assert middle.metadata.core_start == 1
    assert middle.metadata.core_length == 3
    np.testing.assert_array_equal(middle.minimum, [2, 3, 4, 5, 6])
    np.testing.assert_array_equal(middle.maximum, middle.minimum)

    first = load_page(
        s, series_id=0, source_factor=1, page_index=0, common_length=8, page_size=3
    )
    assert first.metadata.core_start == 0
    assert first.metadata.length == 4


def test_nan_is_invalid_but_infinity_is_rejected():
    page = load_page(
        series([1, np.nan, 3]),
        series_id=0,
        source_factor=1,
        page_index=0,
        common_length=3,
        page_size=3,
    )
    np.testing.assert_array_equal(page.valid, [1, 0, 1])
    with pytest.raises(ValueError, match="infinite range"):
        load_page(
            series([1, np.inf, 3]),
            series_id=0,
            source_factor=1,
            page_index=0,
            common_length=3,
            page_size=3,
        )


def test_packing_keeps_buffer_and_metadata_column_lengths_independent():
    s = series(np.arange(8))
    pages = [
        load_page(s, series_id=i, source_factor=1, page_index=0, common_length=8, page_size=3)
        for i in (0, 1)
    ]
    data, metadata = pack_pages(pages)
    assert {len(column) for column in data.values()} == {8}
    assert {len(column) for column in metadata.values()} == {2}
    np.testing.assert_array_equal(metadata["offset"], [0, 4])


def test_reference_raster_aggregates_before_occupancy_and_is_transparent_elsewhere():
    # Disjoint ranges in each input land in one output x pixel.  Extrema are
    # aggregated first, so both occupy the whole interval and overlap is black.
    a = np.array([[0, 0], [2, 2]], dtype=float)
    b = np.array([[1, 1], [3, 3]], dtype=float)
    image = reference_raster(a, b, width=1, height=4, y_start=0, y_end=4)
    np.testing.assert_array_equal(
        image[:, 0],
        [[0, 0, 0, 0], [0, 0, 255, 255], [0, 0, 0, 255], [0, 0, 0, 255]],
    )


def test_reference_raster_supports_reversed_y_ranges():
    ranges = np.array([[0, 1]], dtype=float)
    normal = reference_raster(ranges, ranges, width=1, height=4, y_start=0, y_end=2)
    reversed_ = reference_raster(ranges, ranges, width=1, height=4, y_start=2, y_end=0)
    np.testing.assert_array_equal(normal[:, 0, 3], [0, 0, 255, 255])
    np.testing.assert_array_equal(reversed_[:, 0, 3], [255, 255, 255, 0])


def test_bokeh_model_manifest_and_mailbox_protocol():
    a = series(np.arange(8), name="a:Cz")
    b = series(np.arange(8) + 10, name="b:Cz")
    manifest = build_manifest(a, b, page_size=3)
    renderer = VennTimeSeriesRenderer(manifest=manifest)
    assert renderer.level == "image"
    assert renderer.composition_mode == "pending"
    assert renderer.page_counts == [3]
    assert renderer.page_source.column_names == ["minimum", "maximum", "valid"]

    mailbox = PageMailbox(Document(), renderer, (a, b), workers=1)
    seq, (data, metadata) = mailbox._load(7, ((1, 1),))
    assert seq == 7
    assert len(metadata["series_id"]) == 2
    assert len(data["minimum"]) == 10
    np.testing.assert_array_equal(metadata["series_id"], [0, 1])
    mailbox.close()


def test_recording_tile_source_reuses_one_mmap_and_reads_rectangles(tmp_path):
    path = tmp_path / "recording.set"
    path.touch()
    recording = FakeMmapRecording()
    source = RecordingTileSource(path, recording=recording)

    assert recording.mmap_calls == 1
    np.testing.assert_array_equal(
        source.slice_lines(1, [1, 2], 3, 6)[1],
        np.arange(36, dtype=np.float32).reshape(3, 12)[[1, 2], 3:6],
    )
    ranges = source.slice_ranges(1, [1], 4, 7)
    assert ranges.shape == (1, 3, 2)
    np.testing.assert_array_equal(ranges[..., 0], ranges[..., 1])
    np.testing.assert_array_equal(source.finite_extrema([1, 2]), [[12, 23], [24, 35]])
    assert recording.mmap_calls == 1


def test_recording_tile_source_rejects_embedded_set_before_loading(tmp_path):
    path = tmp_path / "embedded.set"
    path.touch()
    Embedded = type(
        "MmapRawEEGLAB",
        (),
        {
            "filenames": [path],
            "mmap": lambda self, **kwargs: pytest.fail("embedded data was eagerly loaded"),
        },
    )
    with pytest.raises(ValueError, match="external .fdt"):
        RecordingTileSource(path, recording=Embedded())


def test_tile_source_alignment_is_shared_across_channels(tmp_path):
    paths = [tmp_path / name for name in ("a.set", "b.set")]
    for path in paths:
        path.touch()
    a = RecordingTileSource(paths[0], recording=FakeMmapRecording())
    b = RecordingTileSource(paths[1], recording=FakeMmapRecording(offset=10))
    assert validate_tile_source_alignment(a, b) == 12

    bad_times = np.arange(12, dtype=float) * 0.25 + 0.1
    bad = RecordingTileSource(paths[1], recording=FakeMmapRecording(times=bad_times))
    with pytest.raises(ValueError, match="time sample 0"):
        validate_tile_source_alignment(a, bad)


def test_unified_tile_coordinator_packs_channel_and_time_tiles(tmp_path):
    paths = [tmp_path / name for name in ("a.set", "b.set")]
    for path in paths:
        path.touch()
    sources = tuple(
        RecordingTileSource(path, recording=FakeMmapRecording(offset=index * 100))
        for index, path in enumerate(paths)
    )
    renderer = VennTimeSeriesRenderer(
        dataset_version="test",
        sample_count=12,
        time_start=0,
        time_end=2.75,
        sample_interval=0.25,
        page_size=3,
        channel_tile_size=2,
        channel_names=["Fz", "Cz", "Pz"],
        amplitude_scales=[1, 1, 1],
        amplitude_offsets=[0, 1, 2],
        channel_y_mins=[0, 1, 2],
        channel_y_maxs=[1, 2, 3],
        range_factors=[1],
        range_page_counts=[4],
        line_factors=[1],
        line_page_counts=[4],
    )
    coordinator = TileCoordinator(
        Document(), renderer, sources, ["Fz", "Cz", "Pz"], workers=1
    )
    seq, (range_data, range_metadata), (line_data, line_metadata) = coordinator._load(
        4, (("venn", 1, 1, 0), ("lines", 1, 1, 1))
    )
    assert seq == 4
    assert range_metadata["channel_count"].tolist() == [2]
    assert range_metadata["core_start"].tolist() == [1]
    assert len(range_data["minimum_a"]) == 10  # two channels, 3 core + 2 gutters
    assert line_metadata["channel_index"].tolist() == [2]
    assert len(line_data["time"]) == 5
    coordinator.close()


def test_tile_coordinator_queues_responses_until_the_browser_acknowledges(tmp_path):
    class ImmediateDocument:
        def add_next_tick_callback(self, callback):
            callback()

    paths = [tmp_path / name for name in ("a.set", "b.set")]
    for path in paths:
        path.touch()
    sources = tuple(
        RecordingTileSource(path, recording=FakeMmapRecording(offset=index * 100))
        for index, path in enumerate(paths)
    )
    renderer = VennTimeSeriesRenderer(
        dataset_version="response-queue-test",
        sample_count=12,
        time_start=0,
        time_end=2.75,
        sample_interval=0.25,
        page_size=3,
        channel_tile_size=2,
        channel_names=["Fz", "Cz", "Pz"],
        amplitude_scales=[1, 1, 1],
        amplitude_offsets=[0, 1, 2],
        channel_y_mins=[0, 1, 2],
        channel_y_maxs=[1, 2, 3],
        range_factors=[1],
        range_page_counts=[4],
        line_factors=[1],
        line_page_counts=[4],
    )
    coordinator = TileCoordinator(
        ImmediateDocument(), renderer, sources, ["Fz", "Cz", "Pz"], workers=1
    )
    first_request = (("venn", 1, 0, 0),)
    second_request = (("venn", 1, 1, 0),)
    first, second = Future(), Future()
    first.set_result(coordinator._load(1, first_request))
    second.set_result(coordinator._load(2, second_request))

    # Completing the second worker first must not overwrite or skip response 1.
    coordinator._loaded(second, 2, second_request)
    assert renderer.response_seq == 0
    coordinator._loaded(first, 1, first_request)
    assert renderer.response_seq == 1
    assert renderer.response_tiles == list(first_request)

    renderer.response_ack = 1
    assert renderer.response_seq == 2
    assert renderer.response_tiles == list(second_request)
    coordinator.close()


def test_tile_coordinator_publishes_failed_tile_keys_for_acknowledgement(tmp_path):
    class ImmediateDocument:
        def add_next_tick_callback(self, callback):
            callback()

    paths = [tmp_path / name for name in ("a.set", "b.set")]
    for path in paths:
        path.touch()
    sources = tuple(
        RecordingTileSource(path, recording=FakeMmapRecording()) for path in paths
    )
    renderer = VennTimeSeriesRenderer(dataset_version="failed-response-test")
    coordinator = TileCoordinator(
        ImmediateDocument(), renderer, sources, ["Fz"], workers=1
    )
    request = (("lines", 1, 0, 0),)
    failed = Future()
    failed.set_exception(RuntimeError("broken tile"))
    coordinator._loaded(failed, 1, request)

    assert renderer.response_seq == 1
    assert renderer.response_tiles == list(request)
    assert "broken tile" in renderer.response_error
    coordinator.close()
