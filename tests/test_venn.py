import numpy as np
import pytest
from bokeh.document import Document

from ctapdash.venn import (
    ArrayRangeSeries,
    build_manifest,
    load_page,
    pack_pages,
    parse_series_args,
    reference_raster,
    resolve_channel,
    validate_alignment,
)
from ctapdash.vennrender import PageMailbox, VennTimeSeriesRenderer


def series(values, times=None, *, name="recording:channel", levels=None):
    values = np.asarray(values, dtype=np.float64)
    if times is None:
        times = np.arange(len(values), dtype=np.float64) * 0.25
    return ArrayRangeSeries(values, times, channel_identity=name, levels=levels)


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
    assert renderer.page_counts == [3]
    assert renderer.page_source.column_names == ["minimum", "maximum", "valid"]

    mailbox = PageMailbox(Document(), renderer, (a, b), workers=1)
    seq, (data, metadata) = mailbox._load(7, ((1, 1),))
    assert seq == 7
    assert len(metadata["series_id"]) == 2
    assert len(data["minimum"]) == 10
    np.testing.assert_array_equal(metadata["series_id"], [0, 1])
    mailbox.close()
