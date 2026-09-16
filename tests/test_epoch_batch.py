"""Epoch batch wire data, exact grids, independent choices and chunking."""

import json
import struct
from concurrent.futures import Future

import numpy as np
import pytest

from venn_ts.domain import ComparisonDomain
from venn_ts.epoch_batch import build_tile, pack_tiles
from tests.test_domain import coordinator
from tests.test_pyramid_grid import cached_source


@pytest.mark.parametrize("mode", ["union", "intersection"])
@pytest.mark.parametrize("factor", [1, 8, 64])
@pytest.mark.parametrize("layer", ["venn", "lines"])
def test_batches_preserve_every_choice_and_original_bucket(
    tmp_path, mode, factor, layer, monkeypatch
):
    a = cached_source(tmp_path, "1_a", starts=[-3, 40, 80], count=101)
    b = cached_source(tmp_path, "2_b", starts=[7, 35, 160], count=117)
    domain = ComparisonDomain((a, b), mode)
    reads = []
    for side, source in enumerate((a, b)):
        original = source.read_segment

        def read(*args, original=original, side=side, **kwargs):
            reads.append((side, args[0].epoch))
            return original(*args, **kwargs)

        monkeypatch.setattr(source, "read_segment", read)
        if factor > 1:

            class NoRaw:
                def __getattr__(self, name):
                    pytest.fail(f"Cached batch accessed raw {name}")

            source.raw = NoRaw()
    end = domain.start
    for page in range(domain.page_counts([factor], 2)[0]):
        reads.clear()
        tile, data = build_tile(
            domain, ([0, 1], [0, 1]), (0, 2), factor, 2, page, 0, layer
        )
        assert len(reads) == len(set(reads)), "Each source epoch is read once per page"
        assert tile["start"] == pytest.approx(end)
        for row in tile["fragments"]:
            segment = domain.segments[row["segment"]]
            assert row["start"] == pytest.approx(end)
            end = row["end"]
            first = round(
                (row["time"] - domain.bucket_geometry(segment, factor)[0])
                / tile["step"]
            )
            for side in (0, 1):
                for choice, selected in enumerate(segment.choices(side)):
                    source = domain.sources[side]
                    ss = source.segments[selected]
                    offset = round(
                        (segment.original_start - ss.time_start) / domain.interval
                    )
                    expected = source.read_segment(
                        ss,
                        [0, 1],
                        offset,
                        segment.count,
                        factor,
                        first,
                        first + row["count"],
                        layer,
                    )
                    base, stride = row["options"][side][choice]
                    components = 2 if layer == "venn" else 1
                    for ch in range(2):
                        actual = data[
                            base + ch * stride : base
                            + ch * stride
                            + row["count"] * components
                        ]
                        np.testing.assert_array_equal(actual, expected[ch].ravel())
        packet = pack_tiles([(tile, data)])
        header_size = struct.unpack("<I", packet[:4])[0]
        header = json.loads(packet[4 : 4 + header_size])[0]
        assert header["length"] == data.size
        np.testing.assert_array_equal(
            np.frombuffer(packet[4 + header_size :], "<f4"), data
        )
    assert end == pytest.approx(domain.end)


def test_page_count_does_not_grow_with_epoch_count(tmp_path):
    raw = cached_source(tmp_path, "1_raw", count=10000)
    epochs = cached_source(
        tmp_path, "2_epochs", starts=list(range(5, 9900, 15)), count=101
    )
    domain = ComparisonDomain((raw, epochs))
    assert len(domain.segments) > 1000
    assert domain.page_counts([1, 8, 64], 2048) == [5, 1, 1]


def test_chunk_acknowledgments_and_selection_independent_payload(tmp_path, monkeypatch):
    import venn_ts.epoch_batch as batch

    a = cached_source(tmp_path, "1_a", starts=[10, 20], count=100)
    domain = ComparisonDomain((a, a))
    tiles = coordinator(domain, page_size=2048)
    tiles.renderer.segment_revisions = [0] * len(domain.segments)
    monkeypatch.setattr(
        batch, "TRANSPORT_BYTES", 127
    )  # Also split inside floats/header.
    requests = (("venn", 8, 0, 0), ("lines", 8, 0, 0))
    try:
        before = tiles._load(1, requests)
        segment = next(i for i, s in enumerate(domain.segments) if len(s.choices_a) > 1)
        tiles.select_epoch(segment, 0, 1)
        tiles.select_epoch(segment, 1, 1)
        assert tiles._load(1, requests) == before
        future = Future()
        future.set_result(before)
        tiles._loaded(future, 1, requests)
        for callback in list(tiles.document.session_callbacks):
            callback.callback()
        chunks = []
        while True:
            r = tiles.renderer
            chunks.append(bytes(r.epoch_batch_source.data["payload"]))
            assert len(chunks[-1]) <= 127
            assert r.response_seq == 1
            final = r.response_final
            r.response_ack = r.response_generation
            if final:
                break
        assert len(chunks) > 1
        assert b"".join(chunks) == before[1]
        assert tiles._publishing_response is None
        assert tiles._next_response_seq == 2
    finally:
        tiles.close()


def test_epoch_switch_retains_unaffected_marker_models(tmp_path):
    from bokeh.plotting import figure
    from venn_ts.domain_axis import configure_domain_axis, update_epoch_markers

    raw = cached_source(tmp_path, "1_raw", count=400)
    epochs = cached_source(tmp_path, "2_epochs", starts=[50, 100], count=100)
    domain = ComparisonDomain((raw, epochs))
    plot = figure()
    markers = configure_domain_axis(plot, domain, epoch_visible=True)
    original = markers[0]
    original_positions = list(original.positions)
    tiles = coordinator(domain)
    tiles.renderer.segment_revisions = [0] * len(domain.segments)
    segment = next(i for i, s in enumerate(domain.segments) if len(s.choices_b) == 2)
    try:
        tiles.select_epoch(segment, 1, 1)
        update_epoch_markers(plot, domain, True, markers)
        assert markers == [original]
        assert set(original.positions) == set(domain.epoch_starts())
        tiles.select_epoch(segment, 1, 0)
        update_epoch_markers(plot, domain, False, markers)
        assert markers == [original]
        assert original.positions == original_positions
        assert all(not m.visible for m in markers)
    finally:
        tiles.close()


@pytest.mark.parametrize("count", [1, 3])
@pytest.mark.parametrize("layer", ["venn", "lines"])
def test_variable_step_batches(tmp_path, count, layer):
    sources = tuple(
        cached_source(tmp_path, f"{i}_step", starts=[i * 4, 40], count=30)
        for i in range(count)
    )
    domain = ComparisonDomain(sources)
    tile, values = build_tile(domain, ([0],) * count, (0, 1), 1, 128, 0, 0, layer)
    for fragment in tile["fragments"]:
        assert len(fragment["options"]) == count
        segment = domain.segments[fragment["segment"]]
        first = round((fragment["time"] - segment.display_start) / domain.interval)
        for side, options in enumerate(fragment["options"]):
            if not options:
                continue
            offset, stride = options[0]
            _, expected = domain.read(
                segment, side, [0], 1, first, first + fragment["count"], layer
            )
            np.testing.assert_array_equal(
                values[offset : offset + expected.size], expected.ravel()
            )
